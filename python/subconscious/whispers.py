"""Whisper feature — RETIRED 2026-06-04.

The subconscious "whisper" (a one-shot one-sentence message injected into
Claude Code's next prompt) has been permanently removed at the user's request.
The steward now occupies a real context-window slot instead.

This module is kept as a no-op shim so any stray import does not crash. It no
longer writes or reads any whisper queue. Do NOT re-introduce whispers here.
The SEPARATE tool-advisor whisper system (aisys/tools/advisor.py,
pending_whispers/) is unrelated and unaffected.
"""

from __future__ import annotations

import logging
from pathlib import Path

log = logging.getLogger(__name__)

WHISPERS_FILENAME = "whispers.json"


def write_whisper(letta_dir: Path, text: str) -> None:
    """No-op. Whisper feature retired — see module docstring."""
    log.debug("write_whisper called but feature is retired; ignoring (%d chars)", len(text or ""))


def read_and_consume(letta_dir: Path) -> list[dict]:
    """No-op. Whisper feature retired — returns nothing."""
    return []
