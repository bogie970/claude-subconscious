"""Promotion job — runs daily via /schedule.

Sonnet adjudicates candidate memories. As of 2026-05-13 this is a
NEIGHBOR-AWARE 5-verdict consolidator (not the old 3-verdict isolated
gatekeeper which had a ~95% reject rate). Each candidate is presented
to Sonnet alongside its top-5 vector neighbors so the model can pick
the right operation, not just promote-or-reject.

Verdicts (all non-destructive — no destructive verdict ships in this
phase. See `docs/promoter_v2_design.md` for the deferred MERGE/SUPERSEDE
plan that requires provenance lattice + revert tooling first.):

  ADD       — top-sim < 0.55, novel claim → promote candidate → probationary
  REINFORCE — top-sim > 0.85, semantic duplicate → tombstone candidate,
              bump neighbor's seen_count + last_seen_at
  LINK      — top-sim 0.55-0.85, related but not duplicate → promote
              candidate → probationary, append edge to neighbor.links
  DEFER     — uncertain, leave at candidate. Capped at 3 defers, then
              auto-REJECTed to prevent feedback-loop bumps from making
              poor candidates eligible forever.
  REJECT    — hallucination, low quality, or llm_inferred contradicting
              user_stated verified memory → tombstone candidate

Eligibility unchanged: seen_count >= 2 OR confidence >= 0.8.
Probationary memories survive 7 days without contradiction → verified
(handled by run_weekly() — Phase F.2).
"""

from __future__ import annotations

import json
import sys
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from memory.store import MemoryStore


# ---- Result dataclass ----

@dataclass
class PromotionResult:
    processed: int = 0
    added: int = 0          # ADD verdicts → probationary
    reinforced: int = 0     # REINFORCE verdicts (neighbor bumped, candidate tombstoned)
    linked: int = 0         # LINK verdicts → probationary + edge added
    deferred: int = 0       # DEFER verdicts
    rejected: int = 0       # REJECT verdicts
    auto_rejected: int = 0  # candidates that hit DEFER cap
    errors: int = 0
    run_id: str = field(default_factory=lambda: str(uuid.uuid4()))

    # Backwards-compat aliases for existing dashboards/logs
    @property
    def promoted(self) -> int:
        return self.added + self.linked
    @property
    def held(self) -> int:
        return self.deferred


# ---- Eligibility ----

SEEN_COUNT_THRESHOLD = 2
CONFIDENCE_THRESHOLD = 0.8
DEFER_CAP = 3  # after this many defers, auto-REJECT


def _eligible_for_review(row: dict) -> bool:
    if row.get("tier") != "candidate":
        return False
    seen = int(row.get("seen_count") or 1)
    conf = float(row.get("confidence") or 0.0)
    return seen >= SEEN_COUNT_THRESHOLD or conf >= CONFIDENCE_THRESHOLD


# ---- Neighbor lookup ----

NEIGHBOR_K = 5
NEIGHBOR_MIN_SIM = 0.3  # below this, neighbors aren't worth showing


def _row_by_id(store: MemoryStore, record_id: str) -> dict | None:
    """Fetch a raw LanceDB row by id (contains v2 fields that MemoryRecord drops)."""
    try:
        df = store.table.search().where(f"id = '{record_id}'").limit(1).to_pandas()
    except Exception:
        return None
    if df.empty:
        return None
    return df.iloc[0].to_dict()


def _fetch_neighbors(store: MemoryStore, candidate_row: dict) -> list[dict]:
    """Top-K vector neighbors of the candidate, excluding self and tombstoned.

    Returns list of dicts with: id, content (truncated), tier, provenance, similarity.
    """
    content = candidate_row.get("content", "")
    if not content.strip():
        return []
    try:
        results = store.search(content, k=NEIGHBOR_K + 5)  # over-fetch, then filter
    except Exception:
        return []

    neighbors = []
    for scored in results:
        rec = scored.record
        if rec.id == candidate_row["id"]:
            continue
        # MemoryRecord doesn't carry v2 fields; fetch the raw row for tier/provenance
        full = _row_by_id(store, rec.id)
        if full is None:
            continue
        tier = full.get("tier", "")
        if tier == "tombstoned":
            continue
        sim = float(scored.relevance)
        if sim < NEIGHBOR_MIN_SIM:
            continue
        neighbors.append({
            "id": rec.id,
            "content": rec.content[:400],
            "tier": tier,
            "provenance": full.get("provenance", "unknown"),
            "similarity": round(sim, 3),
        })
        if len(neighbors) >= NEIGHBOR_K:
            break
    return neighbors


# ---- Sonnet prompt ----

ADJUDICATION_PROMPT = """You are a memory consolidator. A candidate memory was extracted from a transcript and needs your decision.

You will see the CANDIDATE plus its TOP NEIGHBORS in the existing memory store. Each neighbor's tier is shown — `verified` (durable, trusted), `probationary` (settled but not yet promoted), or `candidate` (unsettled, not yet evaluated). Pick ONE verdict.

CANDIDATE (id={cand_id}, confidence={confidence}, seen_count={seen_count}):
"{cand_content}"

NEIGHBORS (top-{n_neighbors}, sorted by similarity):
{neighbors_block}

REJECT FIRST — recognize garbage before consolidating it:

The candidate pool contains a LOT of non-memory: raw tool output, ephemeral telemetry, placeholder fragments. Reject these aggressively even when neighbors are similar — duplicates of garbage are still garbage. Specifically REJECT if the candidate is:
- Raw command output (`docker ps`, `nvidia-smi`, log lines, stack traces pasted verbatim)
- Tool/system placeholders or empty role tokens (`attachment:`, `system: system:`, file-history-snapshot artifacts)
- Single-point telemetry snapshots (GPU temp at this instant, disk usage right now, "Updated task #N status")
- Source-code pastes / file content (not a fact ABOUT code, but the code itself)
- Single-event incident logs that shouldn't persist (one boot timestamp, one TLS handshake error)
- Garbled transcription artifacts, screenshot metadata, browser session ids
- Truncated/partial sentences that don't carry a complete claim

Apply REJECT to the candidate (it gets tombstoned). DO NOT pick REINFORCE/LINK on a garbage candidate just because a neighbor is also garbage.

ALSO REJECT: a llm_inferred candidate that contradicts a `user_stated` verified neighbor.

VERDICTS for non-garbage candidates:

- ADD: novel claim, no semantically equivalent neighbor. Use when top sim < ~0.55, or all neighbors are on different topics.
- REINFORCE: semantic duplicate of a SETTLED neighbor (tier=verified or probationary). Same fact, different wording. Specify target_id. **Do NOT REINFORCE into a candidate-tier neighbor** — that just bounces signal between candidates without converging. If the best match is candidate-tier, use LINK instead.
- LINK: candidate is related to a neighbor but not redundant — adds context, references the same entity, sibling fact. Use when sim ~0.55-0.85, OR when the best match is candidate-tier (preserves the association for future consolidation when one of them is promoted).
- DEFER: insufficient signal. The candidate is plausibly meaningful but you can't tell from these neighbors. Will re-evaluate next pass (capped at 3 defers).

Tier preference for REINFORCE target_id when multiple neighbors are similar:
verified > probationary. Never pick a candidate-tier neighbor as a REINFORCE target.

Rules:
- REINFORCE and LINK MUST specify target_id (one of the neighbor ids above).
- Do NOT pick MERGE or SUPERSEDE — not implemented; will fall through to DEFER.

Return ONLY JSON, no markdown fences:
{{"verdict": "ADD|REINFORCE|LINK|DEFER|REJECT", "target_id": "<neighbor_id_or_null>", "rationale": "one short sentence"}}
"""


def _format_neighbors(neighbors: list[dict]) -> str:
    if not neighbors:
        return "(no neighbors above threshold — candidate is likely novel)"
    lines = []
    for i, n in enumerate(neighbors, 1):
        lines.append(
            f"  {i}. id={n['id']} sim={n['similarity']} tier={n['tier']} "
            f"provenance={n['provenance']}\n     \"{n['content']}\""
        )
    return "\n".join(lines)


# ---- Audit helper ----

def _audit(store: MemoryStore, *, memory_id: str, op: str, who: str, why: str,
           before: str = "", after: str = "", run_id: str = "") -> None:
    """Append to memory_audit table."""
    from memory.write_gate import _audit as _write_gate_audit
    full_why = f"[{run_id[:8]}] {why}" if run_id else why
    _write_gate_audit(store, memory_id=memory_id, op=op, who=who, why=full_why,
                       before=before, after=after)


# ---- Verdict handlers ----

def _handle_add(store: MemoryStore, row: dict, rationale: str, run_id: str,
                now_iso: str) -> None:
    """Promote candidate to probationary."""
    memory_id = row["id"]
    with store.lock:
        store.table.update(
            where=f"id = '{memory_id}' AND tier = 'candidate'",
            values={"tier": "probationary", "promoted_at": now_iso},
        )
    _audit(store, memory_id=memory_id, op="promote",
           who="sonnet_promoter",
           why=f"ADD: {rationale}",
           before="candidate", after="probationary", run_id=run_id)


def _handle_reinforce(store: MemoryStore, row: dict, target_id: str,
                      rationale: str, run_id: str, now_iso: str) -> bool:
    """Bump neighbor's seen_count + last_seen_at; tombstone candidate.

    Returns True on success, False if target_id was bad OR if target is
    still a candidate itself. REINFORCE only makes sense when there's
    a SETTLED anchor to reinforce. Bouncing seen_count between two
    candidates just makes both more eligible, no convergence (A1-B audit).
    Caller falls through to DEFER when this returns False.
    """
    memory_id = row["id"]
    target_row = _row_by_id(store, target_id)
    if target_row is None:
        return False

    # Tier gate: only verified or probationary can serve as a REINFORCE anchor.
    target_tier = (target_row.get("tier") or "").lower()
    if target_tier not in ("verified", "probationary"):
        log_msg = (f"REINFORCE rejected: target {target_id[:8]} is tier={target_tier!r}, "
                   f"not a settled anchor. Falling through to DEFER.")
        _audit(store, memory_id=memory_id, op="reinforce_rejected",
               who="sonnet_promoter",
               why=log_msg,
               before="candidate", after="candidate", run_id=run_id)
        return False

    # Bump target's seen_count and last_seen_at
    new_seen = int(target_row.get("seen_count") or 1) + 1
    with store.lock:
        store.table.update(
            where=f"id = '{target_id}'",
            values={"seen_count": new_seen, "last_seen_at": now_iso},
        )
    _audit(store, memory_id=target_id, op="reinforce",
           who="sonnet_promoter",
           why=f"REINFORCE from candidate {memory_id[:8]}: {rationale}",
           before=str(new_seen - 1), after=str(new_seen), run_id=run_id)

    # Tombstone the candidate (signal absorbed into target)
    with store.lock:
        store.table.update(
            where=f"id = '{memory_id}' AND tier = 'candidate'",
            values={"tier": "tombstoned"},
        )
    _audit(store, memory_id=memory_id, op="tombstone",
           who="sonnet_promoter",
           why=f"REINFORCE: signal absorbed into {target_id[:8]}",
           before="candidate", after="tombstoned", run_id=run_id)
    return True


def _handle_link(store: MemoryStore, row: dict, target_id: str,
                 rationale: str, run_id: str, now_iso: str) -> bool:
    """Promote candidate to probationary, append edge to neighbor.links.

    Returns True on success, False if target_id was bad.
    """
    memory_id = row["id"]
    target_row = _row_by_id(store, target_id)
    if target_row is None:
        return False

    # Promote candidate
    with store.lock:
        store.table.update(
            where=f"id = '{memory_id}' AND tier = 'candidate'",
            values={"tier": "probationary", "promoted_at": now_iso},
        )

    # Append bidirectional link
    existing_links_str = target_row.get("links", "") or ""
    existing_links = [l for l in existing_links_str.split(",") if l]
    if memory_id not in existing_links:
        existing_links.append(memory_id)
        store.update_links(target_id, existing_links)

    cand_links_str = row.get("links", "") or ""
    cand_links = [l for l in cand_links_str.split(",") if l]
    if target_id not in cand_links:
        cand_links.append(target_id)
        store.update_links(memory_id, cand_links)

    _audit(store, memory_id=memory_id, op="link_promote",
           who="sonnet_promoter",
           why=f"LINK to {target_id[:8]}: {rationale}",
           before="candidate", after="probationary", run_id=run_id)
    return True


def _handle_defer(store: MemoryStore, row: dict, rationale: str, run_id: str) -> bool:
    """Leave at candidate. Returns True if deferred, False if hit cap → auto-reject.

    Uses the existing access_count field to track defer count (separate from
    seen_count which is for re-encounter signal).
    """
    memory_id = row["id"]
    defer_count = int(row.get("access_count") or 0)
    if defer_count >= DEFER_CAP:
        # Auto-reject: actually tombstone the row, not just audit-log.
        # The previous version only wrote an audit entry with before=after=candidate
        # so the row stayed eligible forever and the auto-reject branch fired every
        # daily run (infinite audit spam, no convergence). Bug found in A1-B audit.
        with store.lock:
            store.table.update(
                where=f"id = '{memory_id}' AND tier = 'candidate'",
                values={"tier": "tombstoned"},
            )
        _audit(store, memory_id=memory_id, op="reject",
               who="sonnet_promoter",
               why=f"auto-REJECT after {defer_count} defers (cap={DEFER_CAP}): {rationale}",
               before="candidate", after="tombstoned", run_id=run_id)
        return False
    # Bump defer counter (using access_count field as our defer counter)
    with store.lock:
        store.table.update(
            where=f"id = '{memory_id}'",
            values={"access_count": defer_count + 1},
        )
    _audit(store, memory_id=memory_id, op="defer",
           who="sonnet_promoter",
           why=f"DEFER ({defer_count + 1}/{DEFER_CAP}): {rationale}",
           before=str(defer_count), after=str(defer_count + 1), run_id=run_id)
    return True


def _handle_reject(store: MemoryStore, row: dict, rationale: str, run_id: str) -> None:
    """Tombstone candidate."""
    memory_id = row["id"]
    with store.lock:
        store.table.update(
            where=f"id = '{memory_id}' AND tier = 'candidate'",
            values={"tier": "tombstoned"},
        )
    _audit(store, memory_id=memory_id, op="reject",
           who="sonnet_promoter",
           why=f"REJECT: {rationale}",
           before="candidate", after="tombstoned", run_id=run_id)


# ---- Main entry point ----

def run_daily(
    store: MemoryStore,
    chat_fn=None,
    model: str = "sonnet",
) -> PromotionResult:
    """Run the daily neighbor-aware consolidation pass.

    For each eligible candidate:
      1. Vector-search top-5 neighbors
      2. Ask Sonnet for verdict in {ADD, REINFORCE, LINK, DEFER, REJECT}
      3. Apply verdict via the appropriate handler

    Returns PromotionResult with detailed counts per verdict + run_id.
    """
    if chat_fn is None:
        from memory.l1_manager import _default_chat_fn
        chat_fn = _default_chat_fn

    result = PromotionResult()
    run_id = result.run_id

    with store.lock:
        rows = store.scan_v2_lean()
        candidates = [r for r in rows if _eligible_for_review(r)]

    now_iso = datetime.now(timezone.utc).isoformat()

    for row in candidates:
        result.processed += 1
        memory_id = row["id"]

        try:
            neighbors = _fetch_neighbors(store, row)
        except Exception:
            neighbors = []

        prompt = ADJUDICATION_PROMPT.format(
            cand_id=memory_id,
            confidence=row.get("confidence", 0.0),
            seen_count=row.get("seen_count", 1),
            cand_content=(row.get("content", "") or "")[:1500],
            n_neighbors=len(neighbors),
            neighbors_block=_format_neighbors(neighbors),
        )

        try:
            response = chat_fn(
                messages=[{"role": "user", "content": prompt}],
                model=model,
                timeout=30,
            )
        except Exception:
            result.errors += 1
            continue

        if response is None:
            result.errors += 1
            continue

        text = response.get("content", "").strip()
        if text.startswith("```"):
            text = text.split("```", 2)[1]
            if text.startswith("json"):
                text = text[4:]
            text = text.strip()

        try:
            parsed = json.loads(text)
        except (json.JSONDecodeError, TypeError):
            result.errors += 1
            continue

        verdict = (parsed.get("verdict") or "DEFER").upper()
        target_id = parsed.get("target_id")
        if target_id in ("null", "None", ""):
            target_id = None
        rationale = (parsed.get("rationale") or "")[:200]

        # Dispatch. Each handler acquires store.lock for a brief write. If the
        # lock is contended (e.g. the subconscious Stop-hook writer holds it),
        # a FileLock Timeout must NOT abort the whole pass and discard hours of
        # completed LLM work — count it as a per-candidate error and move on.
        # (Root cause of the 2026-05-17..22 sonnet_daily ERR streak: one
        # contended write at the tail of a 6h run crashed the entire job.)
        try:
            if verdict == "ADD":
                _handle_add(store, row, rationale, run_id, now_iso)
                result.added += 1
            elif verdict == "REINFORCE":
                if target_id and _handle_reinforce(store, row, target_id, rationale, run_id, now_iso):
                    result.reinforced += 1
                else:
                    # Sonnet wanted REINFORCE but no valid target — fall through to DEFER
                    if _handle_defer(store, row, f"REINFORCE missing target: {rationale}", run_id):
                        result.deferred += 1
                    else:
                        result.auto_rejected += 1
            elif verdict == "LINK":
                if target_id and _handle_link(store, row, target_id, rationale, run_id, now_iso):
                    result.linked += 1
                else:
                    if _handle_defer(store, row, f"LINK missing target: {rationale}", run_id):
                        result.deferred += 1
                    else:
                        result.auto_rejected += 1
            elif verdict == "DEFER":
                if _handle_defer(store, row, rationale, run_id):
                    result.deferred += 1
                else:
                    result.auto_rejected += 1
            elif verdict == "REJECT":
                _handle_reject(store, row, rationale, run_id)
                result.rejected += 1
            else:
                # Unknown verdict (including MERGE/SUPERSEDE which aren't implemented)
                # Defer to be safe; surface in error count if it keeps happening.
                if _handle_defer(store, row, f"unknown verdict '{verdict}': {rationale}", run_id):
                    result.deferred += 1
                else:
                    result.auto_rejected += 1
        except Exception:
            # Lock timeout or transient write failure on THIS candidate.
            # Skip it; it stays at candidate tier and is retried next pass.
            result.errors += 1
            continue

    return result
