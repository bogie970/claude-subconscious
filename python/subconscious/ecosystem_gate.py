"""
Ecosystem-cwd gate for the subconscious memory extractor.

PROBLEM: the hermes-memory plugin is enabled GLOBALLY in ~/.claude/settings.json,
so its Stop-hook subconscious worker runs in EVERY Claude Code session on the
machine and writes candidate memories into the ONE shared vector store
(hermes/data/memory_store), all tagged namespace='hermes'. An unrelated chat
(e.g. a "studio" desktop chat opened in some other folder) therefore pollutes
Hermes memory.

FIX: only allow memory writes for sessions whose cwd is under the Hermes
ecosystem root (C:/Users/jbogi/claude-nodes/, covering hermes, atlas, daedalus
— all legit ecosystem agents), in any path form (Windows, WSL, slash variants).

FAIL-SAFE: if cwd is missing/empty/ambiguous, DEFAULT TO WRITE (and warn).
Rationale — cost asymmetry: silently killing legit Hermes extraction (lost
memory across every session) is far worse than a little pollution. A missing
cwd is an input-shape anomaly, not evidence of a foreign session, so we do not
treat it as grounds to drop real Hermes work. The warning makes it visible if
it ever happens systematically.

OVERRIDE: set HERMES_MEMORY_ALLOW_ANY_CWD=1 to bypass the gate entirely
(always write), e.g. if the ecosystem root ever moves.
"""

from __future__ import annotations

import os

# The single ecosystem root. All legit agents (hermes, atlas, daedalus) live
# under this directory. Lowercased, forward-slashed for matching.
_ECOSYSTEM_ROOTS = (
    "c:/users/jbogi/claude-nodes",       # native Windows
    "/mnt/c/users/jbogi/claude-nodes",   # WSL mount of the same path
)


def _normalize(cwd: str) -> str:
    """Lowercase + forward-slash a path so Windows/WSL/slash forms compare equal.

    Handles: backslashes, mixed slashes, trailing slashes, and case.
    """
    s = cwd.strip().replace("\\", "/").lower()
    while "//" in s:
        s = s.replace("//", "/")
    s = s.rstrip("/")
    return s


def is_ecosystem_cwd(cwd: str | None) -> bool:
    """True if this session's cwd is a Hermes-ecosystem session (-> allow write).

    Fail-safe contract:
      - ecosystem cwd            -> True  (write)
      - non-ecosystem cwd        -> False (skip)
      - missing/empty/None cwd   -> True  (write, fail-safe; caller should warn)
      - HERMES_MEMORY_ALLOW_ANY_CWD=1 -> True (override)
    """
    if os.environ.get("HERMES_MEMORY_ALLOW_ANY_CWD", "") == "1":
        return True

    # Fail-safe: missing/empty cwd -> WRITE. "." is the runner's own default
    # sentinel for "no cwd in payload", so treat it as missing too.
    if cwd is None or cwd.strip() in ("", "."):
        return True

    norm = _normalize(cwd)
    for root in _ECOSYSTEM_ROOTS:
        if norm == root or norm.startswith(root + "/"):
            return True
    return False


def cwd_is_missing(cwd: str | None) -> bool:
    """True if cwd is absent/empty/sentinel — caller logs a fail-safe warning."""
    return cwd is None or cwd.strip() in ("", ".")
