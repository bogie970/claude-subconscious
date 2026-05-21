"""Fast query-dependent retrieval for UserPromptSubmit hook injection.

Standalone subprocess entry point: embeds the user's prompt, searches LanceDB
via TripleScoredRetriever, and outputs ranked results as XML to stdout.

Usage:
    python -m memory.query_retrieve "user prompt text" --k 10 --format xml

Performance budget: <300ms after cold start (first call loads embedding model).
Timing is logged to stderr for profiling.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
import pathlib

log = logging.getLogger("memory.query_retrieve")

# ---------------------------------------------------------------------------
# sys.path setup — ensure python/ dir is on path for `from memory.xxx` imports
# ---------------------------------------------------------------------------
_PYTHON_ROOT = pathlib.Path(__file__).resolve().parent.parent
_python_str = str(_PYTHON_ROOT)
if _python_str not in sys.path:
    sys.path.insert(0, _python_str)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_K = 10
MAX_CHARS = 60000  # ~15k token budget (1 token ≈ 4 chars)
DEFAULT_NAMESPACE = "hermes"
EMPTY_RESULT = '<retrieved_memories count="0"/>'


def _escape_xml(text: str) -> str:
    """Escape XML special characters."""
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&apos;")
    )


def _format_xml(scored_memories: list, query_time_ms: int) -> str:
    """Format ScoredMemory list as XML for hook injection."""
    if not scored_memories:
        return EMPTY_RESULT

    lines = [f'<retrieved_memories count="{len(scored_memories)}" query_time_ms="{query_time_ms}">']
    for sm in scored_memories:
        r = sm.record
        created = r.created_at.strftime("%Y-%m-%d") if r.created_at else "unknown"
        tags_attr = ""
        if r.tags:
            tags_attr = f' tags="{_escape_xml(",".join(r.tags))}"'
        # Surface all 5 doctrine §7.3 attrs per injection-audit (2026-05-19):
        # tier, provenance, agent, writer, source_ref — agent can use these to
        # classify memories per doctrine §2 Class A/B/C/D before acting on them.
        # Default empty string if absent (pre-v2 rows or missing column).
        tier = r.metadata.get("_tier", "")
        provenance = r.metadata.get("_provenance", "")
        writer = r.metadata.get("_writer", "")
        source_ref = r.metadata.get("_source_ref", "")
        # agent comes from the top-level `source` field (who wrote the memory)
        agent = r.source or ""
        tier_attr = f' tier="{_escape_xml(tier)}"' if tier else ""
        prov_attr = f' provenance="{_escape_xml(provenance)}"' if provenance else ""
        agent_attr = f' agent="{_escape_xml(agent)}"' if agent else ""
        writer_attr = f' writer="{_escape_xml(writer)}"' if writer else ""
        source_ref_attr = f' source_ref="{_escape_xml(source_ref)}"' if source_ref else ""
        lines.append(
            f'<memory id="{_escape_xml(r.id)}" '
            f'importance="{r.importance:.2f}" '
            f'type="{r.memory_type.value}" '
            f'category="{_escape_xml(r.category)}" '
            f'relevance="{sm.relevance:.2f}" '
            f'score="{sm.combined_score:.2f}" '
            f'created="{created}"'
            f'{tier_attr}'
            f'{prov_attr}'
            f'{agent_attr}'
            f'{writer_attr}'
            f'{source_ref_attr}'
            f'{tags_attr}>'
        )
        lines.append(_escape_xml(r.content))
        lines.append("</memory>")
    lines.append("</retrieved_memories>")
    return "\n".join(lines)


def _truncate_to_budget(scored_memories: list, max_chars: int) -> list:
    """Drop lowest-relevance memories until total content fits budget."""
    total = 0
    kept = []
    for sm in scored_memories:
        # Approximate: content + XML overhead (~200 chars per entry)
        entry_size = len(sm.record.content) + 200
        if total + entry_size > max_chars and kept:
            break
        total += entry_size
        kept.append(sm)
    return kept


def retrieve(query: str, k: int = DEFAULT_K, max_chars: int = MAX_CHARS) -> str:
    """Run retrieval and return XML string. Main entry point for subprocess use."""
    t_start = time.perf_counter()

    try:
        from memory.config import LANCEDB_PATH
        from memory.store import MemoryStore
        from memory.retrieval import TripleScoredRetriever

        t_import = time.perf_counter()
        print(f"[query_retrieve] import: {(t_import - t_start)*1000:.0f}ms", file=sys.stderr)

        store = MemoryStore(db_path=LANCEDB_PATH)
        t_store = time.perf_counter()
        print(f"[query_retrieve] store connect: {(t_store - t_import)*1000:.0f}ms", file=sys.stderr)

        retriever = TripleScoredRetriever(store)
        # track_access=False for speed — access tracking has a known
        # LanceDB API compat issue and adds latency we can't afford here.
        # The MCP server's memory_search already tracks access on direct queries.
        results = retriever.retrieve(
            query,
            k=k,
            namespaces=[DEFAULT_NAMESPACE],
            include_archived=False,
            expand_links=True,
            track_access=False,
        )
        t_retrieve = time.perf_counter()
        print(
            f"[query_retrieve] retrieve ({len(results)} results): "
            f"{(t_retrieve - t_store)*1000:.0f}ms",
            file=sys.stderr,
        )

        # Truncate to token budget (results already sorted by combined_score desc)
        results = _truncate_to_budget(results, max_chars)

        query_time_ms = int((time.perf_counter() - t_start) * 1000)
        xml = _format_xml(results, query_time_ms)

        print(
            f"[query_retrieve] total: {query_time_ms}ms, "
            f"returned {len(results)} memories, {len(xml)} chars",
            file=sys.stderr,
        )
        return xml

    except Exception as e:
        print(f"[query_retrieve] error: {e}", file=sys.stderr)
        return EMPTY_RESULT


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Query-dependent memory retrieval for hook injection"
    )
    parser.add_argument("query", help="The user's prompt text to search against")
    parser.add_argument("--k", type=int, default=DEFAULT_K, help="Number of results")
    parser.add_argument(
        "--max-chars",
        type=int,
        default=MAX_CHARS,
        help="Max total characters for output",
    )
    parser.add_argument(
        "--format",
        choices=["xml"],
        default="xml",
        help="Output format (currently only xml)",
    )
    args = parser.parse_args()

    xml = retrieve(args.query, k=args.k, max_chars=args.max_chars)
    print(xml)


if __name__ == "__main__":
    main()
