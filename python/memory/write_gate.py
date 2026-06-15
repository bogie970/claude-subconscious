"""Write gate — single entry point for ALL memory writes.

Enforces the three-tier pipeline:
    candidate (subconscious-only) -> probationary -> verified

Per-write checks:
    1. content + source_ref required
    2. tier assigned by writer (subconscious_haiku ALWAYS candidate)
    3. filesystem grounding: candidate-demote on missing code refs
    4. cosine dedup: identical/near-duplicate content bumps seen_count
    5. audit log entry recorded

Concurrency: relies on MemoryStore's existing FileLock.
"""

from __future__ import annotations

import json
import math
import re
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None  # type: ignore[assignment]

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from memory.grounding import extract_code_refs, filesystem_exists
from memory.schema import MemoryRecord, MemoryType
from memory.secret_scrub import scrub_with_count
from memory.store import AUDIT_TABLE_NAME, MemoryStore


# ---- Tier assignment rules ----

CANDIDATE_ONLY_WRITERS = {"subconscious_haiku"}
# #69 fix (2026-06-15): "manual" REMOVED from the auto-verified set. The
# conversation-consolidation pass builds MemoryRecords with no explicit writer →
# schema default writer="manual" → reached verified UNCONDITIONALLY here with no
# provenance/confidence gate (the poisoning vector). Only writer="user" now takes
# the verified fast-path; "manual" + the explicit consolidation_pass sentinel are
# structurally capped at probationary in _assign_tier.
# NOTE: this is the claude-subconscious WORKER copy (the live worker imports the
# cache synced from here). Must stay identical to aisys/memory/write_gate.py.
USER_WRITERS = {"user"}
KNOWN_WRITERS = CANDIDATE_ONLY_WRITERS | USER_WRITERS | {
    "sonnet_promoter", "opus_auditor", "system", "manual", "consolidation_pass",
}
# #69: low-trust LLM producers capped at probationary (visible-but-flagged, never
# verified, never candidate). consolidation_pass = the consolidation pass; "manual"
# = the legacy schema-default writer it fell through to.
PROBATIONARY_CAPPED_WRITERS = {"consolidation_pass", "manual"}

DEDUP_COSINE_THRESHOLD = 0.92  # cosine sim above which we treat as duplicate

# Default pattern when doctrine.config.yaml is unavailable.
# Matches: transcript:<session-id>:turn_<n>  e.g. transcript:abc123:turn_7
_DEFAULT_USER_TURN_PATTERN = r'^transcript:[a-z0-9-]+:turn_\d+$'

_DOCTRINE_CONFIG_CACHE: dict | None = None
_DOCTRINE_CONFIG_PATH = Path.home() / ".hermes" / "doctrine.config.yaml"


def _load_doctrine_config() -> dict:
    """Load and cache doctrine.config.yaml. Returns empty dict on any failure."""
    global _DOCTRINE_CONFIG_CACHE
    if _DOCTRINE_CONFIG_CACHE is not None:
        return _DOCTRINE_CONFIG_CACHE
    if yaml is None or not _DOCTRINE_CONFIG_PATH.exists():
        _DOCTRINE_CONFIG_CACHE = {}
        return _DOCTRINE_CONFIG_CACHE
    try:
        with _DOCTRINE_CONFIG_PATH.open("r", encoding="utf-8") as fh:
            _DOCTRINE_CONFIG_CACHE = yaml.safe_load(fh) or {}
    except Exception:  # noqa: BLE001
        _DOCTRINE_CONFIG_CACHE = {}
    return _DOCTRINE_CONFIG_CACHE


def _user_turn_pattern() -> str:
    """Return the compiled source_ref regex pattern for user_stated writes."""
    cfg = _load_doctrine_config()
    return cfg.get("memory", {}).get(
        "user_turn_source_ref_pattern", _DEFAULT_USER_TURN_PATTERN
    )


class WriteRejected(Exception):
    """Raised when a memory write fails validation."""


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sanitize_id(value: str) -> str:
    """Escape single quotes for safe LanceDB SQL where-clause use."""
    return str(value).replace("'", "''")


def _cosine(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


def _audit(store: MemoryStore, *, memory_id: str, op: str, who: str, why: str,
           before: str = "", after: str = "") -> None:
    """Append to memory_audit table. Idempotent — ensures table exists."""
    try:
        audit_table = store.db.open_table(AUDIT_TABLE_NAME)
    except (FileNotFoundError, ValueError):
        from memory.store import LANCE_AUDIT_SCHEMA
        audit_table = store.db.create_table(AUDIT_TABLE_NAME, schema=LANCE_AUDIT_SCHEMA)

    audit_table.add([{
        "id": str(uuid.uuid4()),
        "memory_id": memory_id,
        "op": op,
        "who": who,
        "when": _now_iso(),
        "why": why,
        "before": before,
        "after": after,
    }])


def _assign_tier(
    writer: str, provenance: str, source_ref: str
) -> tuple[str, str]:
    """Tier based on writer + provenance + source_ref validation.

    Returns (tier, demotion_reason).  demotion_reason is empty string when no
    demotion occurred.

    Per injection-audit (2026-05-19): unknown writers MUST NOT achieve
    verified tier via provenance='user_stated'. Only known writers can
    claim user_stated; unknown writers default to probationary regardless.

    A6 fix (2026-05-21): when provenance=='user_stated', source_ref MUST
    match memory.user_turn_source_ref_pattern from doctrine.config.yaml.
    Mismatch → demote to probationary (not reject — import merges need leniency).
    """
    if writer in CANDIDATE_ONLY_WRITERS:
        return "candidate", ""

    # #69: consolidation_pass / legacy "manual" — low-trust LLM output structurally
    # capped at probationary (never verified, never candidate). Closes the poisoning
    # vector where "manual" hit the unconditional verified fast-path below.
    if writer in PROBATIONARY_CAPPED_WRITERS:
        return "probationary", ""

    if provenance == "user_stated":
        pattern = _user_turn_pattern()
        if not re.match(pattern, source_ref or ""):
            reason = (
                f"source_ref {source_ref!r} does not match user_turn_source_ref_pattern "
                f"{pattern!r}; demoted from verified to probationary"
            )
            return "probationary", reason

    if writer in USER_WRITERS:
        return "verified", ""
    if writer in KNOWN_WRITERS and provenance == "user_stated":
        return "verified", ""
    return "probationary", ""


def _find_dedup_candidate(store: MemoryStore, vector: list[float]) -> dict | None:
    """Native LanceDB vector search for a near-duplicate. Returns the row or None.

    Considers verified/probationary/candidate tiers (excludes only tombstoned).
    Uses LanceDB's vector index (or linear scan for small stores) — drops
    write-time dedup from O(N) Python cosine to ~O(log N) at scale.

    Embeddings are L2-normalized (per EmbeddingService), so LanceDB's
    default cosine distance == 1 - cos(theta). We threshold on distance.
    """
    distance_threshold = 1.0 - DEDUP_COSINE_THRESHOLD  # 0.08

    try:
        results = (
            store.table.search(vector)
            .where("tier != 'tombstoned'")
            .limit(1)
            .to_list()
        )
    except Exception:
        # If the where-clause or search fails, fall back to no-dedup
        # (write proceeds; duplicate may be caught later by cleanup).
        return None

    if not results:
        return None
    top = results[0]
    distance = top.get("_distance", 1.0)
    if distance > distance_threshold:
        return None
    return top


def write_memory(
    store: MemoryStore,
    *,
    content: str,
    writer: str,
    provenance: str,
    source_ref: str,
    confidence: float,
    tags: list[str] | None = None,
    memory_type: MemoryType = MemoryType.SEMANTIC,
    category: str = "general",
    source: str = "hermes",
) -> str:
    """Write a memory through the gate. Returns the resulting record id.

    On dedup: returns the existing record's id and bumps its seen_count.
    On rejection: raises WriteRejected.
    """
    # 1. Validate input
    if not content or not content.strip():
        raise WriteRejected("content required")
    if not source_ref:
        raise WriteRejected("source_ref required")
    if writer not in KNOWN_WRITERS:
        # Unknown writers cannot claim user_stated provenance — that path
        # would bypass _assign_tier's known-writer check and produce a
        # verified-tier record. Reject explicitly per injection-audit.
        if provenance == "user_stated":
            raise WriteRejected(
                f"unknown writer {writer!r} cannot claim provenance='user_stated'"
            )
        # Otherwise allow; _assign_tier will return probationary.

    # 1a. Scrub credentials before they hit L2 — pasted API keys / tokens
    #     would otherwise resurface via retrieval months later.
    content, secrets_redacted = scrub_with_count(content)

    # 2. Tier assignment (includes source_ref pattern check for user_stated)
    tier, tier_demotion_reason = _assign_tier(writer, provenance, source_ref)
    effective_confidence = confidence

    # 3. Filesystem grounding for code refs (demotion)
    code_refs = extract_code_refs(content)
    missing_refs = [r for r in code_refs if not filesystem_exists(r)]
    if missing_refs and tier == "verified":
        tier = "candidate"
        effective_confidence *= 0.5

    # 4. Compute embedding (slow; do this outside the critical section)
    vector = store._embedder.embed_one(content)

    # 5. Critical section: dedup-check + insert/update must be atomic to prevent
    #    races where N concurrent threads all miss-dedup and create N copies.
    with store.lock:
        existing = _find_dedup_candidate(store, vector)
        if existing is not None:
            existing_id = _sanitize_id(existing["id"])
            existing_tier = existing.get("tier", "")

            # Trust-aware seen_count: subconscious writers cannot boost the
            # seen_count of an already-verified/probationary row, since that
            # would let hallucinations accelerate promotion of unrelated trusted
            # memories. Audit the dedup hit either way.
            should_bump = not (
                writer in CANDIDATE_ONLY_WRITERS
                and existing_tier in ("verified", "probationary")
            )

            if should_bump:
                new_seen = int(existing.get("seen_count", 1)) + 1
                store.table.update(
                    where=f"id = '{existing_id}'",
                    values={"seen_count": new_seen, "last_seen_at": _now_iso()},
                )
                _audit(
                    store, memory_id=existing_id, op="seen_bump", who=writer,
                    why=f"dedup hit on {source_ref}",
                    before=str(new_seen - 1), after=str(new_seen),
                )
            else:
                _audit(
                    store, memory_id=existing_id, op="seen_bump_blocked",
                    who=writer,
                    why=f"trust-block: {writer} cannot bump verified row from {source_ref}",
                    before=str(existing.get("seen_count", 1)),
                    after=str(existing.get("seen_count", 1)),
                )
            return existing_id

        # New record path
        rec = MemoryRecord(
            content=content,
            memory_type=memory_type,
            category=category,
            source=source,
            importance=effective_confidence,
            tags=tags or [],
        )
        rec.metadata["_v2_overrides"] = {
            "tier": tier,
            "provenance": provenance,
            "source_ref": source_ref,
            "writer": writer,
            "confidence": effective_confidence,
        }
        row = rec.to_lance_dict(vector, include_v2=True)
        store.table.add([row])

    _audit(
        store, memory_id=rec.id, op="create", who=writer,
        why=f"new memory at tier={tier} from {source_ref}"
            + (f" (scrubbed {secrets_redacted} secrets)" if secrets_redacted else "")
            + (f" [{tier_demotion_reason}]" if tier_demotion_reason else ""),
        after=json.dumps({
            "tier": tier,
            "provenance": provenance,
            "writer": writer,
            "secrets_redacted": secrets_redacted,
            **({"tier_demotion_reason": tier_demotion_reason} if tier_demotion_reason else {}),
        }),
    )

    return rec.id
