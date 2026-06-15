"""Conversation-to-memory consolidation engine (Phase 2).

Extracts semantic chunks from conversation transcripts, enriches them
with A-MEM style metadata, deduplicates against existing memories,
and stores new/merged memories in LanceDB L2.

Called by runner.py AFTER the existing block-update loop completes.
Runs in a separate process (spawned by Stop hook), so claude-cli is safe.
"""

from __future__ import annotations

import json
import logging
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

log = logging.getLogger("subconscious.consolidation")

# Batch size for LLM extraction calls
EXTRACTION_BATCH_SIZE = 8

EXTRACTION_PROMPT = """\
You are a memory extraction system. Given the following conversation messages, \
extract standalone semantic facts, decisions, observations, and procedures.

RULES:
1. Extract, don't summarize. Each chunk must be a standalone fact.
2. Preserve specifics: file paths, function names, error messages, numbers.
3. User corrections of the AI are importance >= 0.8.
4. Decisions are importance >= 0.7. Capture the decision AND the reasoning.
5. Don't extract obvious things. "User asked about X" is NOT useful. \
"User decided to use approach Y because Z" IS useful.
6. Procedural knowledge (how-to, workflows) should be type "procedural".
7. Specific events/sessions should be type "episodic".
8. General facts/rules/insights should be type "semantic".
9. Skip trivial greetings, acknowledgments, and filler.
10. Each chunk must make sense without the surrounding conversation.

Output a JSON array (no markdown fences). Each element:
{{
  "content": "The semantic fact or observation — standalone, specific",
  "type": "semantic|episodic|procedural",
  "category": "project|feedback|preference|decision|procedure|reference",
  "importance": <float 0.3-0.9>,
  "tags": ["tag1", "tag2"]
}}

DO NOT EXTRACT (return nothing for content that is only these):
- Raw command/tool output, shell output, logs, stack traces (verbatim pastes)
- Single-point telemetry (GPU temp now, disk usage right now, "Updated task #N", heartbeats)
- Source-code or file-content pastes (facts ABOUT code are fine, the code itself is not)
- Tool/system placeholders and truncated fragments that do not carry a complete claim
- Single transient events that should not persist (one boot timestamp, one TLS error, session IDs)
If the entire input consists only of the above, return an empty array: []

If there is nothing worth extracting, return an empty array: []

CONVERSATION:
---
{messages}
---

Respond with ONLY the JSON array."""


@dataclass
class SemanticChunk:
    """A single extracted semantic unit from conversation."""
    content: str
    memory_type: str  # semantic | episodic | procedural
    category: str
    importance: float
    tags: list[str] = field(default_factory=list)


@dataclass
class DedupResult:
    """Result of deduplication check."""
    is_duplicate: bool
    existing_id: str | None = None
    similarity: float = 0.0
    action: str = "insert"  # insert | skip | merge


@dataclass
class ConsolidationResult:
    """Summary of a consolidation run."""
    extracted: int = 0
    stored: int = 0
    merged: int = 0
    skipped: int = 0
    errors: list[str] = field(default_factory=list)


def parse_transcript_messages(transcript_xml: str) -> list[dict[str, str]]:
    """Parse transcript XML into a list of message dicts.

    Each dict has keys: role, content, timestamp (optional).
    Handles malformed XML gracefully with regex fallback.
    """
    messages: list[dict[str, str]] = []

    # Try proper XML parsing first
    try:
        # Wrap in root if needed
        xml_text = transcript_xml.strip()
        if not xml_text.startswith("<transcript"):
            xml_text = f"<transcript>{xml_text}</transcript>"
        root = ET.fromstring(xml_text)
        for msg_elem in root.findall(".//message"):
            role = msg_elem.get("role", "unknown")
            timestamp = msg_elem.get("timestamp", "")
            content = "".join(msg_elem.itertext())
            if content.strip():
                messages.append({
                    "role": role,
                    "content": content.strip(),
                    "timestamp": timestamp,
                })
        if messages:
            return messages
    except ET.ParseError:
        pass

    # Regex fallback for malformed XML
    pattern = r'<message\s+role="([^"]*)"(?:\s+timestamp="([^"]*)")?\s*>(.*?)</message>'
    for match in re.finditer(pattern, transcript_xml, re.DOTALL):
        role, timestamp, content = match.groups()
        content = content.strip()
        if content:
            messages.append({
                "role": role,
                "content": content,
                "timestamp": timestamp or "",
            })

    return messages


def _format_messages_for_prompt(messages: list[dict[str, str]]) -> str:
    """Format a batch of messages as text for the extraction prompt."""
    lines = []
    for msg in messages:
        role = msg.get("role", "unknown").upper()
        content = msg.get("content", "")
        # Truncate very long messages to avoid blowing up the prompt
        if len(content) > 2000:
            content = content[:2000] + "... [truncated]"
        lines.append(f"[{role}]: {content}")
    return "\n\n".join(lines)


def _parse_extraction_response(raw: str) -> list[SemanticChunk]:
    """Parse LLM extraction response into SemanticChunk objects."""
    raw = raw.strip()
    # Strip markdown fences if present
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0].strip()

    try:
        items = json.loads(raw)
    except json.JSONDecodeError:
        # Find first '[' and try progressively longer substrings
        items = None
        start = raw.find('[')
        while start >= 0 and items is None:
            for end in range(len(raw), start, -1):
                if raw[end - 1] == ']':
                    try:
                        items = json.loads(raw[start:end])
                        break
                    except json.JSONDecodeError:
                        continue
            if items is None:
                start = raw.find('[', start + 1)
        if items is None:
            log.warning("No JSON array found in extraction response: %s", raw[:300])
            return []

    if not isinstance(items, list):
        log.warning("Extraction response is not a list: %s", type(items))
        return []

    chunks = []
    for item in items:
        if not isinstance(item, dict):
            continue
        content = item.get("content", "").strip()
        if not content:
            continue

        # Validate and clamp importance
        try:
            importance = float(item.get("importance", 0.5))
        except (TypeError, ValueError):
            importance = 0.5
        importance = max(0.3, min(0.9, importance))

        # Validate memory type
        mem_type = item.get("type", "semantic")
        if mem_type not in ("semantic", "episodic", "procedural"):
            mem_type = "semantic"

        # Validate category
        category = item.get("category", "project")
        valid_categories = {"project", "feedback", "preference", "decision", "procedure", "reference"}
        if category not in valid_categories:
            category = "project"

        # Parse tags
        tags = item.get("tags", [])
        if not isinstance(tags, list):
            tags = []
        tags = [str(t).lower().strip() for t in tags if t]

        chunks.append(SemanticChunk(
            content=content,
            memory_type=mem_type,
            category=category,
            importance=importance,
            tags=tags,
        ))

    return chunks


class ConsolidationEngine:
    """Extracts semantic chunks from conversation, enriches, stores in LanceDB."""

    def __init__(self, memory_store: Any, llm: Any):
        """Initialize with a MemoryStore and an LLM caller.

        Args:
            memory_store: MemoryStore instance for LanceDB operations.
            llm: Callable[[str], str] — the LLM extraction function.
        """
        self._store = memory_store
        self._llm = llm

    def process_batch(
        self,
        messages: list[dict[str, str]],
        session_id: str,
    ) -> ConsolidationResult:
        """Process a batch of conversation messages into L2 vector memories.

        Steps:
        1. LLM extraction: Ask LLM to extract semantic chunks from messages
        2. Deduplication: Check if similar memories already exist (cosine sim > 0.85)
        3. Storage: Insert new memories, merge overlapping ones

        Args:
            messages: Parsed conversation messages (role, content, timestamp).
            session_id: Current session identifier for provenance.

        Returns:
            ConsolidationResult with counts of extracted/stored/merged/skipped.
        """
        result = ConsolidationResult()

        if not messages:
            return result

        # Step 1: Extract chunks in batches
        all_chunks: list[SemanticChunk] = []
        for i in range(0, len(messages), EXTRACTION_BATCH_SIZE):
            batch = messages[i:i + EXTRACTION_BATCH_SIZE]
            try:
                chunks = self._extract_chunks(batch)
                all_chunks.extend(chunks)
            except Exception as e:
                log.warning("Extraction failed for batch %d: %s", i, e)
                result.errors.append(f"Extraction batch {i}: {e}")

        result.extracted = len(all_chunks)
        log.info("Extracted %d chunks from %d messages", len(all_chunks), len(messages))

        if not all_chunks:
            return result

        # Step 2 + 3: Dedup and store each chunk
        from memory.schema import MemoryType  # noqa: F401  (kept for chunk validation parity)

        # #69 fix: inserts are routed through the write gate (writer="consolidation_pass")
        # instead of constructing MemoryRecords and persisting them directly via
        # insert_many/insert. The old direct-insert path BYPASSED write_gate entirely,
        # so consolidation's llm_inferred chunks could land at verified (the poisoning
        # vector — ~2,140 rows). The gate structurally caps consolidation_pass at
        # probationary (visible-but-flagged), exactly like steward_distill/dream_reflector.
        chunks_to_insert: list[SemanticChunk] = []

        for chunk in all_chunks:
            try:
                dedup = self._check_duplicates(chunk)

                if dedup.action == "skip":
                    result.skipped += 1
                    log.debug("Skipping duplicate (sim=%.3f): %s",
                              dedup.similarity, chunk.content[:80])
                    continue

                if dedup.action == "merge" and dedup.existing_id:
                    try:
                        existing = self._store.get_by_id(dedup.existing_id)
                        if existing:
                            # Bump importance if new is higher
                            if chunk.importance > existing.importance:
                                self._store.update_importance(dedup.existing_id, chunk.importance)
                            # If new content is meaningfully longer, it likely has more context — replace
                            if len(chunk.content) > len(existing.content) * 1.2:
                                self._store.update_content(dedup.existing_id, chunk.content)
                                log.debug("Merged (replaced content, %d->%d chars): %s",
                                          len(existing.content), len(chunk.content), chunk.content[:80])
                            # Merge tags
                            new_tags = list(set(existing.tags + chunk.tags))
                            if new_tags != existing.tags:
                                self._store.update_tags(dedup.existing_id, new_tags)
                        self._store.update_access(dedup.existing_id)
                        result.merged += 1
                    except Exception as e:
                        log.warning("Merge failed for %s: %s", dedup.existing_id, e)
                        result.errors.append(f"Merge {dedup.existing_id}: {e}")
                    continue

                # action == "insert" — defer to the gated persist loop below.
                chunks_to_insert.append(chunk)

            except Exception as e:
                log.warning("Processing chunk failed: %s — %s", chunk.content[:80], e)
                result.errors.append(f"Chunk processing: {e}")

        # Persist each new chunk THROUGH THE WRITE GATE (#69). The gate assigns the
        # tier (consolidation_pass → probationary), runs secret-scrub + filesystem
        # grounding + write-time dedup, and records an audit entry — none of which the
        # old direct insert_many path did. enrich=False parity is preserved implicitly:
        # write_memory does not A-MEM-enrich; it embeds raw content, which is what the
        # extraction prompt already produces (concise, keyword-rich).
        if chunks_to_insert:
            from memory.write_gate import WriteRejected, write_memory

            for chunk in chunks_to_insert:
                try:
                    write_memory(
                        self._store,
                        content=chunk.content,
                        writer="consolidation_pass",   # #69: capped at probationary in the gate
                        provenance="llm_inferred",     # extraction output, NOT a user statement
                        source_ref=f"consolidation:{session_id}",
                        confidence=chunk.importance,
                        tags=chunk.tags,
                        memory_type=MemoryType(chunk.memory_type),
                        category=chunk.category,
                        source="consolidation",
                    )
                    result.stored += 1
                except WriteRejected as e:
                    log.warning("Gate rejected consolidation chunk: %s — %s",
                                chunk.content[:80], e)
                    result.errors.append(f"Gate rejected: {e}")
                except Exception as e:
                    log.warning("Gated insert failed: %s — %s", chunk.content[:80], e)
                    result.errors.append(f"Gated insert: {e}")

            log.info("Stored %d new memories (via write gate, tier=probationary)",
                     result.stored)

        return result

    def _extract_chunks(self, messages: list[dict[str, str]]) -> list[SemanticChunk]:
        """Use LLM to extract semantic facts/decisions/observations from messages."""
        formatted = _format_messages_for_prompt(messages)
        prompt = EXTRACTION_PROMPT.format(messages=formatted)

        raw = self._llm(prompt)
        if not raw:
            log.warning("LLM returned empty response for extraction")
            return []

        return _parse_extraction_response(raw)

    def _check_duplicates(self, chunk: SemanticChunk) -> DedupResult:
        """Check if a similar memory already exists.

        Uses vector similarity search. If cosine similarity > 0.85,
        treat as duplicate (skip or merge based on importance delta).
        """
        from memory.config import MERGE_SIMILARITY_THRESHOLD

        try:
            results = self._store.search(
                query=chunk.content,
                k=1,
                namespaces=["hermes"],
            )
        except Exception as e:
            log.warning("Dedup search failed: %s", e)
            return DedupResult(is_duplicate=False, action="insert")

        if not results:
            return DedupResult(is_duplicate=False, action="insert")

        top = results[0]
        similarity = top.relevance  # 1.0 - distance

        if similarity >= MERGE_SIMILARITY_THRESHOLD:
            # Very similar — merge if new importance is higher, skip otherwise
            if chunk.importance > top.record.importance:
                return DedupResult(
                    is_duplicate=True,
                    existing_id=top.record.id,
                    similarity=similarity,
                    action="merge",
                )
            else:
                return DedupResult(
                    is_duplicate=True,
                    existing_id=top.record.id,
                    similarity=similarity,
                    action="skip",
                )

        return DedupResult(is_duplicate=False, similarity=similarity, action="insert")
