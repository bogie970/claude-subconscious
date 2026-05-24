"""Unified memory maintenance entrypoint — CUT 4b (S4 consolidation).

Replaces direct invocation of promotion_cli / cleanup with a single
entrypoint that accepts --stage daily|weekly.  The two Task Scheduler
entries (HermesSonnetDaily at 02:00, HermesOpusWeekly Sundays at 03:00)
keep their SEPARATE schedules and cadences — this file just eliminates
the duplicate env-setup boilerplate in the .cmd wrappers.

Usage (matches existing scheduled wrappers):
  python -m memory.maintenance --stage daily   # was: promotion_cli --model sonnet
  python -m memory.maintenance --stage weekly  # was: cleanup

NOTE (CUT 2 pre-condition): this file lives in the claude-subconscious
cache copy because the .cmd wrappers set PYTHONPATH to that location.
When CUT 2 (pip install -e) lands, this module will move to the canonical
aisys/memory/ source and the cache copy will be removed.
"""

from __future__ import annotations

import argparse
import sys


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Hermes memory maintenance dispatcher",
        epilog="Stages: daily = Sonnet promotion pass; weekly = Opus cleanup pass",
    )
    ap.add_argument(
        "--stage",
        choices=["daily", "weekly"],
        required=True,
        help="Which maintenance stage to run",
    )
    # Forward any extra args to the underlying CLI (e.g. --dry-run, --model)
    args, remaining = ap.parse_known_args()

    if args.stage == "daily":
        # Delegate to promotion_cli.main() — preserves all its arg handling
        from memory.promotion_cli import main as promotion_main  # type: ignore[import]
        # Rewrite argv so promotion_cli sees its own flags
        sys.argv = ["memory.promotion_cli", "--model", "sonnet"] + remaining
        return promotion_main()
    else:
        # args.stage == "weekly"
        from memory.cleanup import main as cleanup_main  # type: ignore[import]
        sys.argv = ["memory.cleanup"] + remaining
        return cleanup_main()


if __name__ == "__main__":
    sys.exit(main() or 0)
