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
import logging
import sys
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from memory.store import MemoryStore

_log = logging.getLogger("memory.promotion")

try:
    from filelock import Timeout as _LockTimeout  # store lock contention signal
except Exception:  # pragma: no cover - filelock always present alongside store
    _LockTimeout = None


# ---- MF-2 guard: distilled rows must NEVER auto-graduate to verified ----
# Jacob's D2 decision (b): steward_distill rows stay VISIBLE-BUT-FLAGGED FOREVER
# at probationary — they are NEVER auto-promoted to verified. They only ever harden
# via genuine re-encounter (a user/non-distill writer re-states the same claim) or a
# manual trust action; the distill path itself can never push them past probationary.
#
# There is NO probationary->verified auto-promote path built TODAY (the documented
# `run_weekly` 7-day survival sweep does not exist — only cleanup.run_weekly_cleanup,
# which is GC). This guard exists so that if anyone LATER builds that sweep, distilled
# rows are explicitly excluded instead of silently hardening to verified with no
# re-check. Any future auto-promote-to-verified loop MUST call this before promoting.
_DISTILL_WRITER = "steward_distill"
# #69 (2026-06-15): writers that must NEVER auto-graduate to verified. Extends the
# distill bar to consolidation (gate-capped at probationary) and the legacy "manual"
# schema-default that the #69 backfill demoted — so a FUTURE probationary->verified
# survival sweep can't silently re-poison the verified tier. llm_inferred provenance
# is barred too (inferred != verified-worthy without genuine re-encounter).
_NEVER_AUTO_VERIFIED_WRITERS = {_DISTILL_WRITER, "consolidation_pass", "manual"}
_NO_AUTO_PROMOTE_TAGS = {"auto_unproven", "steward_distilled"}


def is_auto_promote_barred(row: dict) -> bool:
    """True if this memory row must NEVER auto-graduate to verified (MF-2 / MF4).

    Bars any row written by ``steward_distill`` OR tagged ``auto_unproven`` /
    ``steward_distilled``. Fail-CLOSED on malformed input (returns True) so a
    parsing slip can never let a distilled row sneak into a verified promotion.
    Tags are stored comma-joined in LanceDB; this accepts list or str.

    MF4 — NO BACKDOOR TO VERIFIED. Graduation does NOT take a parameter that
    relaxes this bar. Graduation moves a row shadow->probationary-VISIBLE at most
    (clears ``steward_shadow``, KEEPS ``auto_unproven``), so a graduated row STILL
    returns True here forever. The ONLY paths to ``verified`` are (a) genuine
    re-encounter — a NON-distill writer re-states the claim — and (b) Jacob's
    explicit trust. There is no probationary->verified survival sweep in the
    codebase today; if one is ever built, it MUST consult this guard
    (``assert_no_auto_promote_to_verified``) before hardening ANY row.
    """
    try:
        if (row.get("writer") or "") in _NEVER_AUTO_VERIFIED_WRITERS:
            return True
        if (row.get("provenance") or "") == "llm_inferred":
            return True
        raw_tags = row.get("tags") or []
        if isinstance(raw_tags, str):
            tag_set = {t.strip() for t in raw_tags.split(",") if t.strip()}
        else:
            tag_set = {str(t).strip() for t in raw_tags}
        return bool(tag_set & _NO_AUTO_PROMOTE_TAGS)
    except Exception:
        return True


class AutoPromoteBarViolation(RuntimeError):
    """Raised if a future survival-sweep tries to auto-promote a barred row."""


def assert_no_auto_promote_to_verified(row: dict) -> None:
    """Executable MF4 guard — NOT just a comment.

    Any future probationary->verified auto-promote loop MUST call this for each
    row before setting ``tier='verified'``. It raises ``AutoPromoteBarViolation``
    for any row ``is_auto_promote_barred`` rejects (distilled / auto_unproven /
    graduation-cleared shadow rows), making the no-backdoor invariant a hard,
    testable assertion instead of documentation. Graduation itself never sets
    ``verified`` and never calls this — it's a tripwire for code that doesn't
    exist yet.
    """
    if is_auto_promote_barred(row):
        rid = (row.get("id") or "?")
        raise AutoPromoteBarViolation(
            f"MF4: row {rid!r} is auto-promote-barred (distilled/auto_unproven) and "
            f"must NEVER be hardened to verified by an automatic sweep. The only "
            f"paths to verified are genuine re-encounter or Jacob's explicit trust."
        )


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
    # Split taxonomy so /maintenance-status can tell "provider down" from
    # "bad JSON" (#190). cli_errors = chat_fn returned None/empty (provider
    # outage, e.g. 02:00 auth/quota). parse_errors = json.loads failed (the
    # model replied but the output wasn't valid JSON). errors stays as the
    # grand total for backwards-compat with existing dashboards/logs.
    cli_errors: int = 0
    parse_errors: int = 0
    eligible: int = 0       # total eligible candidates this pass (pre-cap)
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


# ---- Per-pass drain cap + single-lock acquire bounds ----
#
# LOCK STARVATION FIX (2026-06-12, see docs/audits/MEMORY_PROMOTION_LOCK_
# STARVATION_2026-06-12.md). Previously each candidate's handler acquired
# store.lock 2-4x; across a 5,330-candidate backlog that was ~15k serialized
# 60s-timeout acquisitions — a positive-feedback starvation loop. The pass now
# acquires store.lock ONCE around the whole loop. filelock.FileLock is
# PROCESS-REENTRANT (internal counter), so the handlers' inner `with store.lock`
# become instant no-op counter bumps while the outer lock is held — they are
# LEFT UNCHANGED (lower risk than rewriting every handler).
import os as _os
import time as _time


def _promotion_max_per_pass() -> int:
    """Cap on candidates processed per run. Drains the backlog over many nights
    instead of one giant lock-hold. Default 400; env-overridable. <=0 -> uncapped."""
    try:
        return int(_os.environ.get("HERMES_PROMOTION_MAX_PER_PASS", "400"))
    except Exception:
        return 400


# Bounded retry/backoff on the SINGLE outer lock acquire, mirroring _atomic_write
# (lcw/steward/runner.py:469-477). Generous because promotion is offline/nightly.
_LOCK_ACQUIRE_ATTEMPTS = 3          # #54 (2026-06-15): 3 tries (was 4)...
_LOCK_ACQUIRE_BACKOFF_S = 5.0       # ...with EXPONENTIAL 5s,15s,45s sleeps between
#   (base * 3**attempt; was linear 5/10/15). Generous: distill folds can hold the
#   outer lock for minutes, and promotion is offline/nightly so a longer wait is
#   cheap. Combined with the store.py default timeout bump to 300s (#54), a normal
#   fold no longer starves the 2AM promotion cron. Per-attempt FileLock timeout
#   (default 300s, or HERMES_STORE_LOCK_TIMEOUT_S) still bounds each acquire.


# ---- 1A: Rule-based cheap-reject pre-filter (no LLM) ----
#
# Design 1A (PROMOTION_THROUGHPUT_AND_LOCK_LIFECYCLE_2026-06-13.md). Tombstone
# UNAMBIGUOUS garbage BEFORE the per-candidate Sonnet call, eliminating its LLM
# cost. Spec = the ADJUDICATION_PROMPT "REJECT FIRST" rules, but applied ONLY to
# cases that are non-memory by surface pattern alone. CONSERVATIVE BY DESIGN:
# when in doubt we DO NOT reject — the candidate falls through to the LLM. We never
# regex on semantics (that is the LLM's job); only on tool-output/telemetry/code
# shape. Gated by HERMES_PROMOTION_CHEAP_REJECT (default ON), reversible by setting
# it to 0/false. Runs UNDER the outer lock already held by run_daily — it calls the
# existing _handle_reject tombstone path and acquires NO new outer lock.
import re as _re

# DELIBERATELY NO bare length floor. The design lists "<25 chars = fragment" but
# length alone is the WEAKEST signal — a terse-but-real fact ("Atlas uses Clore.ai",
# "thing 1") is short yet valid memory. Per the CONSERVATIVE mandate (when in doubt,
# fall through to the LLM), we reject ONLY shapes that are unambiguously non-memory
# (tool-output/telemetry/code/empty), never short prose. Empty/whitespace is caught
# explicitly below; everything else short goes to the LLM.

# Line-START tokens that mark raw tool output / telemetry / log dumps. Matched
# case-sensitively against the stripped content's first ~40 chars. Deliberately
# narrow: each is a header/prefix a HUMAN FACT would essentially never start with.
_CHEAP_REJECT_PREFIXES = (
    "CONTAINER ID",      # docker ps
    "REPOSITORY",        # docker images
    "Traceback (most recent call last)",
    "nvidia-smi",
    "+--",                # ascii table border (nvidia-smi / docker)
    "+==",
    "| ",                 # ascii table row
    "Warning:",
    "WARNING:",
    "Error:",             # bare log-line error prefix (not "Error rate is..." — see guard)
    "ERROR:",
    "[ERROR]",
    "[WARN]",
    "[INFO]",
    "[DEBUG]",
    "PS C:\\",            # powershell prompt paste
    "$ ",                 # shell prompt paste
    "Updated task #",     # telemetry: task-status snapshot
    "attachment:",        # tool/system placeholder
    "system: system:",    # empty role token
)

# Source-code paste markers (the code itself, not a fact ABOUT code).
_CHEAP_REJECT_CODE_PREFIXES = (
    "def ",
    "import ",
    "from __future__",
    "#!/",
    "class ",
    "@app.",
    "</",                 # html/xml fragment
)

# All-telemetry line: only digits, colons, slashes, dashes, dots, spaces, %, and
# unit letters — a timestamp / temp / disk-usage snapshot with no prose.
_TELEMETRY_RE = _re.compile(r"^[\d\s:/\.\-%°CFGMKBTtimsh]+$")
# Browser/session UUID with little else around it.
_UUID_RE = _re.compile(
    r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b"
)


def _cheap_reject_match(content: str) -> str | None:
    """Pure, side-effect-free classifier. Return a short reason string if the
    content is UNAMBIGUOUS garbage (caller will tombstone), else None (fall
    through to the LLM). CONSERVATIVE: only the clearest non-memory shapes match.

    Never raises — a malformed input returns None (fall through, never reject).
    """
    try:
        if content is None:
            return None
        stripped = content.strip()
        if not stripped:
            return "empty/whitespace"

        head = stripped[:48]
        for pref in _CHEAP_REJECT_PREFIXES:
            if head.startswith(pref):
                return f"tool-output prefix {pref!r}"
        for pref in _CHEAP_REJECT_CODE_PREFIXES:
            if head.startswith(pref):
                return f"code-paste prefix {pref!r}"

        first_line = stripped.splitlines()[0].strip()
        if _UUID_RE.match(first_line):
            return "browser/session uuid"
        # Telemetry: the WHOLE content (single short snapshot) is punctuation+digits.
        # Guarded to single-line so a multi-paragraph fact is never swept here.
        if "\n" not in stripped and _TELEMETRY_RE.match(stripped):
            return "telemetry snapshot (digits/units only)"
        return None
    except Exception:
        return None


def _cheap_reject_enabled() -> bool:
    """1A gate. Default ON; set HERMES_PROMOTION_CHEAP_REJECT=0/false/off to disable."""
    raw = _os.environ.get("HERMES_PROMOTION_CHEAP_REJECT", "1").strip().lower()
    return raw not in ("0", "false", "off", "no", "")


# ---- Part 2: Lock-holder DETECTION guard (detect + warn; kill is opt-in) ----
#
# Design Part 2, SAFER variant. At promotion startup, BEFORE the outer acquire,
# inspect the memory_store.lock file age and (if psutil is available) any process
# holding it longer than LOCK_MAX_HOLD_H. DEFAULT = DETECT + LOG LOUDLY ONLY — it
# never kills. Auto-kill is gated behind HERMES_PROMOTION_KILL_STALE_LOCK=1
# (default OFF). FAILS OPEN on EVERY error path (psutil missing, permission denied,
# any exception): it logs and returns, never crashing or aborting the run.
_LOCK_MAX_HOLD_DEFAULT_H = 4.0
_LOCK_RECENT_GRACE_H = 1.0  # never flag a lock younger than this (live session)


def _lock_max_hold_h() -> float:
    try:
        v = float(_os.environ.get("HERMES_PROMOTION_LOCK_MAX_HOLD_H", str(_LOCK_MAX_HOLD_DEFAULT_H)))
        return v if v > 0 else _LOCK_MAX_HOLD_DEFAULT_H
    except Exception:
        return _LOCK_MAX_HOLD_DEFAULT_H


def _kill_stale_lock_enabled() -> bool:
    """Auto-kill gate. OFF by default; only '1'/'true'/'yes'/'on' enable it."""
    raw = _os.environ.get("HERMES_PROMOTION_KILL_STALE_LOCK", "0").strip().lower()
    return raw in ("1", "true", "yes", "on")


def _detect_stale_lock_holder(lock_path: str) -> None:
    """Detect a process holding memory_store.lock for > LOCK_MAX_HOLD_H and LOG it.

    DEFAULT: warn-only, NEVER kill. Auto-kill only if HERMES_PROMOTION_KILL_STALE_LOCK
    is set. FAILS OPEN on all errors — psutil missing, permission denied, or any
    exception is logged and swallowed; this function never raises and never aborts
    the promotion run. It is purely opportunistic visibility.
    """
    try:
        if not lock_path:
            return
        if not _os.path.exists(lock_path):
            return  # no lock file -> nothing held -> nothing to detect
        max_hold_s = _lock_max_hold_h() * 3600.0
        grace_s = _LOCK_RECENT_GRACE_H * 3600.0
        try:
            age_s = _time.time() - _os.path.getmtime(lock_path)
        except OSError as e:
            _log.warning("PROMOTION lock-guard: could not stat lock %s (%s) — "
                         "fail-open, proceeding.", lock_path, e)
            return
        if age_s < grace_s:
            return  # recently touched -> a live session, not a stale daemon
        if age_s < max_hold_s:
            return  # held a while but under the threshold -> not stale yet

        age_h = age_s / 3600.0
        _log.warning(
            "PROMOTION lock-guard: memory_store.lock at %s is %.1fh old "
            "(>%.1fh threshold) — possible STALE holder starving promotion. "
            "Detection-only by default (set HERMES_PROMOTION_KILL_STALE_LOCK=1 "
            "to auto-terminate).", lock_path, age_h, _lock_max_hold_h())

        # Best-effort PID identification (psutil optional). Never required.
        try:
            import psutil  # type: ignore
        except Exception:
            _log.warning("PROMOTION lock-guard: psutil unavailable — cannot identify "
                         "the holding PID; detection-only, proceeding.")
            return

        suspects = []
        try:
            for proc in psutil.process_iter(["pid", "name", "create_time", "cmdline"]):
                try:
                    name = (proc.info.get("name") or "")
                    if "python" not in name.lower():
                        continue
                    cmdline = " ".join(proc.info.get("cmdline") or [])
                    if not any(m in cmdline for m in (
                            "memory_server.py", "hook_session_end.py",
                            "l1_manager_cli.py", "memory_daemon")):
                        continue
                    proc_age_s = _time.time() - float(proc.info.get("create_time") or 0)
                    if proc_age_s < max_hold_s:
                        continue
                    suspects.append((proc.info.get("pid"), name, proc_age_s / 3600.0))
                except Exception:
                    continue  # per-proc fault -> skip that proc, keep scanning
        except Exception as e:
            _log.warning("PROMOTION lock-guard: process scan failed (%s) — "
                         "fail-open, proceeding.", e)
            return

        for pid, name, p_age_h in suspects:
            _log.warning("PROMOTION lock-guard: suspect stale holder pid=%s name=%s "
                         "age=%.1fh cmdline matched memory worker.", pid, name, p_age_h)

        if not _kill_stale_lock_enabled():
            if suspects:
                _log.warning("PROMOTION lock-guard: %d suspect(s) identified but "
                             "auto-kill is OFF (default) — NOT terminating. Set "
                             "HERMES_PROMOTION_KILL_STALE_LOCK=1 to enable.",
                             len(suspects))
            return

        # Opt-in kill path. Still fail-open per-proc.
        for pid, name, p_age_h in suspects:
            try:
                p = psutil.Process(pid)
                _log.warning("PROMOTION lock-guard: KILL enabled — terminating stale "
                             "holder pid=%s name=%s age=%.1fh.", pid, name, p_age_h)
                p.terminate()
                try:
                    p.wait(timeout=5)
                except Exception:
                    p.kill()
            except Exception as e:
                _log.warning("PROMOTION lock-guard: failed to terminate pid=%s (%s) "
                             "— fail-open, proceeding.", pid, e)
    except Exception as e:
        # Absolute backstop: the guard must NEVER crash the run.
        _log.warning("PROMOTION lock-guard: unexpected error (%s) — fail-open, "
                     "proceeding with normal lock acquire.", e)
        return


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

    # Part 2: detect (and, only if explicitly enabled, kill) a stale lock holder
    # BEFORE we attempt any acquire. Detection-only by default; fail-open always.
    try:
        _detect_stale_lock_holder(getattr(store.lock, "lock_file", "") or "")
    except Exception:
        pass  # belt-and-suspenders: guard already fails open internally

    with store.lock:
        rows = store.scan_v2_lean()
        candidates = [r for r in rows if _eligible_for_review(r)]

    # Drain ORDERING: oldest-first so each night processes the next slice of the
    # backlog (a ROTATING subset), not the same 400 forever. created_at ascending
    # means rows promoted/tombstoned this pass drop out of the eligible set, so
    # tomorrow's oldest-400 are genuinely different rows. Rows with no created_at
    # sort first (treated as oldest). Stable: ties keep scan order.
    candidates.sort(key=lambda r: (r.get("created_at") or ""))

    # Per-pass cap: process at most N. Does NOT touch verdict logic — only how
    # many candidates this run looks at. The 7-day probationary soak is untouched.
    cap = _promotion_max_per_pass()
    total_eligible = len(candidates)
    if cap > 0 and total_eligible > cap:
        candidates = candidates[:cap]
        _log.info("PROMOTION per-pass cap: %d eligible, processing oldest %d this pass",
                  total_eligible, cap)
    result.eligible = total_eligible

    now_iso = datetime.now(timezone.utc).isoformat()

    # SINGLE outer lock for the WHOLE pass (see LOCK STARVATION FIX above).
    # Bounded retry/backoff on this ONE acquire; on total failure, FAIL OPEN:
    # log loudly, skip the pass, retry next night — never crash the nightly job.
    _outer = store.lock
    _acquired = False
    for _attempt in range(_LOCK_ACQUIRE_ATTEMPTS):
        try:
            _outer.acquire()
            _acquired = True
            break
        except Exception as _e:
            is_timeout = _LockTimeout is not None and isinstance(_e, _LockTimeout)
            if not is_timeout:
                raise  # genuine fault (not contention) — surface it
            if _attempt < _LOCK_ACQUIRE_ATTEMPTS - 1:
                # Exponential backoff: 5s, 15s, 45s (#54 fix; was linear 5/10/15s).
                _sleep = _LOCK_ACQUIRE_BACKOFF_S * (3 ** _attempt)
                _log.warning(
                    "PROMOTION outer lock contended (attempt %d/%d), backing off %.0fs",
                    _attempt + 1, _LOCK_ACQUIRE_ATTEMPTS, _sleep)
                _time.sleep(_sleep)
    if not _acquired:
        _log.warning(
            "PROMOTION fail-open: could not acquire store lock after %d attempts — "
            "skipping this pass, %d eligible candidates retried next night.",
            _LOCK_ACQUIRE_ATTEMPTS, len(candidates))
        result.errors += 1
        return result

    try:
      _cheap_on = _cheap_reject_enabled()
      for row in candidates:
        result.processed += 1
        memory_id = row["id"]

        # 1A: rule-based pre-filter. Tombstone UNAMBIGUOUS garbage with ZERO LLM
        # cost, BEFORE the neighbor fetch + Sonnet call. Runs under the outer lock
        # already held (no new acquire); _handle_reject's inner `with store.lock`
        # is a reentrant no-op counter bump. Counts in result.rejected, identical
        # to an LLM REJECT verdict. Conservative: only the clearest non-memory
        # shapes match; everything else falls through to the LLM unchanged.
        if _cheap_on:
            _cr_reason = _cheap_reject_match(row.get("content", "") or "")
            if _cr_reason is not None:
                try:
                    _handle_reject(store, row, f"cheap-reject: {_cr_reason}", run_id)
                    result.rejected += 1
                except Exception as e:
                    # A write fault here is treated like any per-candidate write
                    # fault: skip, retried next pass, never abort the whole run.
                    _log.warning(
                        "PROMOTION cheap-reject write fault on %s (%s) — skipped, "
                        "retried next pass. %s: %s",
                        memory_id, _cr_reason, type(e).__name__, e)
                    result.errors += 1
                continue

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
            # chat_fn raised — provider/transport failure, not bad model output.
            result.errors += 1
            result.cli_errors += 1
            continue

        if response is None:
            # _default_chat_fn returns None when the caller returned empty
            # (rc!=0 from `claude --print`) — this is the #190 outage signature.
            result.errors += 1
            result.cli_errors += 1
            continue

        text = response.get("content", "").strip()
        if text.startswith("```"):
            text = text.split("```", 2)[1]
            if text.startswith("json"):
                text = text[4:]
            text = text.strip()

        # Empty text after a non-None response = provider returned an empty
        # string (also the #190 outage signature via _default_chat_fn paths
        # that yield {"content": ""}). Count as a CLI error, not a parse error.
        if not text:
            result.errors += 1
            result.cli_errors += 1
            continue

        try:
            parsed = json.loads(text)
        except (json.JSONDecodeError, TypeError):
            # Model replied but output wasn't valid JSON — bad model output,
            # not a provider outage.
            result.errors += 1
            result.parse_errors += 1
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
        except Exception as e:
            # Lock timeout or transient write failure on THIS candidate.
            # Skip it; it stays at candidate tier and is retried next pass.
            # Log LOUDLY so a contended store lock is visible, not silent.
            if _LockTimeout is not None and isinstance(e, _LockTimeout):
                _log.warning(
                    "PROMOTION fail-open: store lock contended on candidate %s "
                    "verdict=%s — skipped, retried next pass. %s",
                    row.get("id"), verdict, e,
                )
            else:
                _log.warning(
                    "PROMOTION fail-open: write fault on candidate %s verdict=%s "
                    "— skipped, retried next pass. %s: %s",
                    row.get("id"), verdict, type(e).__name__, e,
                )
            result.errors += 1
            continue
    finally:
        # Release the single outer lock held for the whole pass. The handlers'
        # inner reentrant acquires already balanced out; this drops the count to 0.
        try:
            _outer.release()
        except Exception:
            pass

    return result


# ===========================================================================
# MEMORY GRADUATION (Stage 1 = the ground-truth double-check; Stage 2 = the
# existing 5-verdict consolidator). Design: STEWARD_GRADUATION_DESIGN_2026-06-05
# + the 6 MUST-FIXES (steward.md §7). All paths fail-closed; flag-gated; dry-run.
# ===========================================================================

# The shadow lane these rows live in (set by distill_fold).
_SHADOW_TAG = "steward_shadow"
GRADUATION_DRYRUN_LOG = "steward_graduation_dryrun.jsonl"


@dataclass
class GraduationResult:
    processed: int = 0
    promoted: int = 0       # ACCURATE -> un-shadowed (AUTOPROMOTE on) then Stage-2
    queued: int = 0         # MF3: ACCURATE routed to /graduation-review (AUTOPROMOTE off)
    tombstoned: int = 0     # confident FABRICATED (MF6)
    flagged: int = 0        # PARTIAL / DISTORTED / UNRESOLVED surfaced to Jacob
    deferred: int = 0       # ERROR / parse fault — retried next pass
    errors: int = 0
    disabled: bool = False
    dry_run: bool = False
    autopromote: bool = False
    run_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    # In dry-run we collect the would-do plan instead of mutating.
    plan: list = field(default_factory=list)


def _graduation_flags():
    """Resolve (master_enabled, autopromote_enabled) from steward_config.
    Fail-closed: any import/resolve error -> both OFF."""
    try:
        import importlib
        sc = importlib.import_module("lcw.steward.steward_config")
        return bool(sc._graduation_enabled()), bool(sc._graduation_autopromote_enabled())
    except Exception:
        return False, False


def _scan_shadow_rows(store: MemoryStore, *, session_id: str | None,
                      limit: int | None) -> list[dict]:
    """Pull steward_shadow rows. Uses is_auto_promote_barred's own tag logic for
    the filter (the rows graduation operates on are exactly the barred shadow lane).
    """
    with store.lock:
        rows = store.scan_v2_lean()
    out = []
    for r in rows:
        if (r.get("tier") or "") == "tombstoned":
            continue
        tags = r.get("tags") or ""
        tag_str = tags if isinstance(tags, str) else ",".join(str(t) for t in tags)
        if _SHADOW_TAG not in tag_str:
            continue
        if not is_auto_promote_barred(r):
            continue  # only the distilled/auto_unproven lane is graduation's business
        if session_id and (r.get("session_id") or "") != session_id:
            continue
        out.append(r)
        if limit and len(out) >= limit:
            break
    return out


def _remove_shadow_tag(store: MemoryStore, row: dict, run_id: str,
                       verdict: str = "ACCURATE") -> None:
    """Un-shadow: drop ``steward_shadow`` only. KEEP auto_unproven + steward_distilled
    (MF4 — never a backdoor; row stays auto-promote-barred forever)."""
    memory_id = row["id"]
    tags = row.get("tags") or ""
    tag_list = ([t.strip() for t in tags.split(",")] if isinstance(tags, str)
                else [str(t).strip() for t in tags])
    new_tags = [t for t in tag_list if t and t != _SHADOW_TAG]
    basis = ("span-verified" if verdict == "ACCURATE"
             else "concrete-token grounded + entailment confirmed")
    with store.lock:
        store.table.update(where=f"id = '{memory_id}'",
                           values={"tags": ",".join(new_tags)})
    _audit(store, memory_id=memory_id, op="graduate_unshadow", who="graduation",
           why=f"{verdict} ({basis}): shadow->probationary-visible; auto_unproven KEPT",
           before=",".join(tag_list), after=",".join(new_tags), run_id=run_id)


def _tombstone_fabrication(store: MemoryStore, row: dict, why: str,
                           run_id: str) -> None:
    """Bitemporal reject (MF6) — reversible, touches ONLY this row, never neighbors.
    Mirrors the consolidator's REJECT path + sets valid_to (never hard-delete)."""
    from datetime import datetime, timezone
    memory_id = row["id"]
    now_iso = datetime.now(timezone.utc).isoformat()
    with store.lock:
        store.table.update(
            where=f"id = '{memory_id}'",
            values={"tier": "tombstoned", "valid_to": now_iso},
        )
    _audit(store, memory_id=memory_id, op="graduate_tombstone", who="graduation",
           why=why, before="probationary(shadow)", after="tombstoned", run_id=run_id)


def _audit_flag(store: MemoryStore, row: dict, verdict: str, why: str,
                run_id: str) -> None:
    """Record a graduate_flag audit so /graduation-review + the dashboard can read
    the awaiting-Jacob queue from memory_audit (no separate bookkeeping)."""
    _audit(store, memory_id=row["id"], op="graduate_flag", who="graduation",
           why=f"{verdict}: {why}", before="shadow", after="awaiting_jacob",
           run_id=run_id)


def _audit_queue(store: MemoryStore, row: dict, why: str, run_id: str,
                 verdict: str = "ACCURATE") -> None:
    """MF3: ACCURATE / VERIFIED_PARAPHRASE row queued for Jacob (AUTOPROMOTE off).
    Distinct op so the dashboard can show 'awaiting-you' separately from
    'flagged borderline'."""
    basis = ("span-verified" if verdict == "ACCURATE"
             else "concrete-token grounded + entailment confirmed")
    _audit(store, memory_id=row["id"], op="graduate_queue", who="graduation",
           why=f"{verdict} ({basis}) queued for review (autopromote OFF): {why}",
           before="shadow", after="awaiting_jacob", run_id=run_id)


def run_graduation(
    store: MemoryStore,
    chat_fn=None,
    model: str = "sonnet",
    *,
    dry_run: bool = False,
    limit: int | None = None,
    session_id: str | None = None,
    marker_dir=None,
) -> GraduationResult:
    """Stage-1 ground-truth gate over the steward_shadow distilled lane.

    For each shadow row:
      1. ``verify_against_source`` (HARD substring check, fail-closed) — Stage 1.
      2. Route:
         - ACCURATE + AUTOPROMOTE on  -> un-shadow, then flow VERBATIM into the
           existing 5-verdict consolidator handlers (Stage 2).
         - ACCURATE + AUTOPROMOTE off -> MF3: queue to /graduation-review, NO
           un-shadow (review-EVERY-promotion-first; Jacob's LOCKED ruling).
         - confident FABRICATED (MF6)  -> auto-tombstone (allowed even AUTOPROMOTE off).
         - PARTIAL / DISTORTED / UNRESOLVED -> flag to Jacob, never promote.
         - ERROR / parse fault -> defer (retried next pass).

    Gates: master ``STEWARD_GRADUATION`` (off -> disabled no-op). ``dry_run`` writes
    a would-do plan jsonl and makes ZERO store mutations.
    """
    result = GraduationResult(dry_run=dry_run)
    run_id = result.run_id

    enabled, autopromote = _graduation_flags()
    result.autopromote = autopromote
    if not enabled:
        result.disabled = True
        return result

    if chat_fn is None:
        try:
            from memory.l1_manager import _default_chat_fn
            chat_fn = _default_chat_fn
        except Exception:
            chat_fn = None

    try:
        from lcw.steward.graduate import verify_against_source  # hermes root on path
    except Exception:
        from steward.graduate import verify_against_source  # type: ignore

    rows = _scan_shadow_rows(store, session_id=session_id, limit=limit)

    for row in rows:
        result.processed += 1
        try:
            vr = verify_against_source(store, row, chat_fn=chat_fn,
                                       marker_dir=marker_dir)
        except Exception:
            result.errors += 1
            result.deferred += 1
            continue

        verdict = vr.verdict
        plan_entry = {
            "id": row.get("id"),
            "verdict": verdict,
            "span_verified": vr.span_verified,
            "confident_fabrication": vr.confident_fabrication,
            "overlap": round(vr.overlap, 4),
            # VERIFIED_PARAPHRASE audit (design §4.2): WHY a paraphrase was judged
            # faithful — surfaced in the dry-run plan + /graduation-review digest.
            "grounding_coverage": round(getattr(vr, "grounding_coverage", 0.0), 4),
            "numeric_grounded": getattr(vr, "numeric_grounded", False),
            "source_via": vr.source_via,
            "claim": (row.get("content") or "")[:200],
            "span": vr.supporting_span[:200],
            "reasoning": vr.reasoning[:200],
        }

        # ---- decide the action (MF3/MF5/MF6) --------------------------------
        # VERIFIED_PARAPHRASE routes IDENTICALLY to ACCURATE (design §4.2): a
        # faithful paraphrase that passed BOTH the CODE concrete-token grounding
        # AND the entailment confirmation is as trustworthy as a span-verified
        # quote FOR THE PURPOSE OF QUEUING — it still NEVER auto-hardens to
        # verified (MF4) and, with autopromote OFF (default), it QUEUES for Jacob.
        if verdict in ("ACCURATE", "VERIFIED_PARAPHRASE"):
            if autopromote:
                action = "would_promote" if dry_run else "promote"
            else:
                action = "would_queue" if dry_run else "queue"  # MF3
        elif verdict == "FABRICATED" and vr.confident_fabrication:
            action = "would_tombstone" if dry_run else "tombstone"  # MF6
        else:
            # PARTIAL / DISTORTED / UNRESOLVED -> flag; ERROR -> defer.
            if verdict == "ERROR":
                action = "defer"
            else:
                action = "would_flag" if dry_run else "flag"
        plan_entry["action"] = action

        if dry_run:
            result.plan.append(plan_entry)
            if action == "would_promote":
                result.promoted += 1
            elif action == "would_queue":
                result.queued += 1
            elif action == "would_tombstone":
                result.tombstoned += 1
            elif action == "would_flag":
                result.flagged += 1
            elif action == "defer":
                result.deferred += 1
            continue

        # ---- LIVE mutation (gated; per-row isolated) ------------------------
        try:
            if action == "promote":
                _remove_shadow_tag(store, row, run_id, verdict=verdict)
                # Stage 2: the row is now a visible probationary distilled row.
                # Flow it VERBATIM through the existing consolidator handlers by
                # presenting it as a candidate-shaped row to the neighbor logic.
                _run_stage2_consolidate(store, row, run_id)
                result.promoted += 1
            elif action == "queue":
                _audit_queue(store, row, vr.reasoning, run_id, verdict=verdict)  # MF3
                result.queued += 1
            elif action == "tombstone":
                _tombstone_fabrication(
                    store, row,
                    f"confident FABRICATED (span absent, overlap={vr.overlap:.2f}): "
                    f"{vr.reasoning}", run_id)
                result.tombstoned += 1
            elif action == "flag":
                _audit_flag(store, row, verdict, vr.reasoning, run_id)
                result.flagged += 1
            else:  # defer
                result.deferred += 1
        except Exception as e:
            # FAIL-OPEN: a contended store lock (the live backend holds it) or a
            # transient write fault on THIS row must NOT abort the pass. Skip the
            # row (it stays shadow/probationary and is retried next pass) and log
            # LOUDLY so contention is visible rather than silently swallowed.
            if _LockTimeout is not None and isinstance(e, _LockTimeout):
                _log.warning(
                    "GRADUATION fail-open: store lock contended (backend busy) on "
                    "row %s action=%s — skipped, will retry next pass. %s",
                    row.get("id"), action, e,
                )
            else:
                _log.warning(
                    "GRADUATION fail-open: write fault on row %s action=%s — "
                    "skipped, will retry next pass. %s: %s",
                    row.get("id"), action, type(e).__name__, e,
                )
            result.errors += 1
            continue

    if dry_run:
        _write_graduation_plan(result)
    return result


def _run_stage2_consolidate(store: MemoryStore, row: dict, run_id: str) -> None:
    """Stage 2: run the just-un-shadowed row through the EXISTING 5-verdict
    consolidator handlers VERBATIM. The row is probationary-visible now; we present
    it to the neighbor-aware verdict so REINFORCE/LINK/REJECT still apply (a faithful-
    but-garbage claim still gets caught by the REJECT-first net). Best-effort: a
    Stage-2 failure does not undo the (valid, audited) un-shadow.
    """
    from datetime import datetime, timezone
    try:
        from memory.l1_manager import _default_chat_fn
        chat_fn = _default_chat_fn
    except Exception:
        return
    now_iso = datetime.now(timezone.utc).isoformat()
    try:
        neighbors = _fetch_neighbors(store, row)
    except Exception:
        neighbors = []
    prompt = ADJUDICATION_PROMPT.format(
        cand_id=row["id"], confidence=row.get("confidence", 0.0),
        seen_count=row.get("seen_count", 1),
        cand_content=(row.get("content", "") or "")[:1500],
        n_neighbors=len(neighbors), neighbors_block=_format_neighbors(neighbors),
    )
    try:
        resp = chat_fn(messages=[{"role": "user", "content": prompt}],
                       model="sonnet", timeout=30)
        parsed = json.loads(_strip_code_fence(resp.get("content", "")))
    except Exception:
        return
    verdict = (parsed.get("verdict") or "DEFER").upper()
    target_id = parsed.get("target_id")
    if target_id in ("null", "None", ""):
        target_id = None
    rationale = (parsed.get("rationale") or "")[:200]
    try:
        if verdict == "REINFORCE" and target_id:
            _handle_reinforce(store, row, target_id, rationale, run_id, now_iso)
        elif verdict == "LINK" and target_id:
            _handle_link(store, row, target_id, rationale, run_id, now_iso)
        elif verdict == "REJECT":
            _handle_reject(store, row, rationale, run_id)
        # ADD/DEFER: the row is already probationary-visible; nothing further to do.
    except Exception:
        return


def _strip_code_fence(text: str) -> str:
    text = (text or "").strip()
    if text.startswith("```"):
        text = text.split("```", 2)[1]
        if text.startswith("json"):
            text = text[4:]
        text = text.strip()
    return text


def _write_graduation_plan(result: "GraduationResult") -> None:
    """Dry-run: append the would-do plan to a jsonl (mirrors distill_loop dry-run).
    ZERO store mutation already guaranteed by the dry_run branch above."""
    try:
        try:
            from lcw.steward.distill_loop import _runtime_dir  # type: ignore
        except Exception:
            from steward.distill_loop import _runtime_dir  # type: ignore
        rt = _runtime_dir(None)
    except Exception:
        rt = Path.home() / ".hermes" / "runtime"
    try:
        rt.mkdir(parents=True, exist_ok=True)
        path = rt / GRADUATION_DRYRUN_LOG
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps({
                "kind": "graduation_dryrun",
                "run_id": result.run_id,
                "ts": datetime.now(timezone.utc).isoformat(),
                "autopromote": result.autopromote,
                "counts": {
                    "processed": result.processed, "would_promote": result.promoted,
                    "would_queue": result.queued, "would_tombstone": result.tombstoned,
                    "would_flag": result.flagged, "deferred": result.deferred,
                },
                "plan": result.plan,
            }) + "\n")
    except Exception:
        pass


# ---------------------------------------------------------------------------
# /graduation-review surface (MVP) — read the awaiting-Jacob queue + counts from
# memory_audit (the same source /maintenance-status reads; no new bookkeeping).
# ---------------------------------------------------------------------------
def graduation_review_queue(store: MemoryStore, *, limit: int = 5) -> list[dict]:
    """List the rows awaiting Jacob (graduate_queue / graduate_flag audit ops whose
    row is still un-tombstoned + still shadow/visible). Returns newest-first, capped.
    """
    try:
        recs = store.audit_scan()
    except Exception:
        return []
    recs = [a for a in recs if a.get("op") in ("graduate_queue", "graduate_flag")]
    recs.sort(key=lambda a: a.get("when", ""), reverse=True)
    out: list[dict] = []
    seen: set[str] = set()
    for a in recs:
        mid = a.get("memory_id")
        if not mid or mid in seen:
            continue
        seen.add(mid)
        row = _row_by_id(store, mid)
        if row is None or (row.get("tier") or "") == "tombstoned":
            continue
        out.append({
            "id": mid,
            "op": a.get("op"),
            "why": a.get("why", ""),
            "claim": (row.get("content") or "")[:200],
            "tier": row.get("tier"),
        })
        if len(out) >= limit:
            break
    return out


def graduation_dashboard_counts(store: MemoryStore) -> dict:
    """Dashboard panel counts from memory_audit graduate_* ops (the weekly metric).
    'proposed N / promoted / probation(queued+flagged) / awaiting-you / rejected'."""
    counts = {"proposed": 0, "promoted": 0, "queued": 0, "flagged": 0,
              "tombstoned": 0}
    try:
        recs = store.audit_scan()
    except Exception:
        return counts
    op_map = {"graduate_unshadow": "promoted", "graduate_queue": "queued",
              "graduate_flag": "flagged", "graduate_tombstone": "tombstoned"}
    for a in recs:
        key = op_map.get(a.get("op"))
        if key:
            counts[key] += 1
    counts["proposed"] = (counts["promoted"] + counts["queued"]
                          + counts["flagged"] + counts["tombstoned"])
    counts["awaiting_you"] = counts["queued"] + counts["flagged"]
    return counts
