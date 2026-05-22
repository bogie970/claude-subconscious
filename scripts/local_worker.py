"""Thin bridge: spawned by TS hooks, imports and runs the Python worker from hermes.

The TS hook sets HERMES_ROOT env var before spawning this script.
This script adds HERMES_ROOT to sys.path and delegates to the actual runner.

Concurrency cap (2026-05-17): on startup, count live local_worker.py
processes. If > MAX_CONCURRENT_WORKERS, exit silently. Prevents the
runaway-spawn pattern that produced 20+ stuck workers when the memory
store lock was contended. Stop hook fires every turn — missing one is
cheap; piling up is dangerous.

Orphan sweep (2026-05-22): BEFORE the concurrency-cap check, run a
bounded sweep over the temp state dir to surface payloads that prior
workers orphaned (cap-exit, crash, kill -9, lease-TTL-expiry). The
sweep ticks a per-payload failure counter; after _MAX_SWEEP_FAILURES,
moves the payload to dead-letter. This is the foundation of the
lifecycle contract — every payload reaches either successful processing
or dead-letter, never sits in temp forever.

Usage: python local_worker.py <payload.json>
"""

import json
import os
import sys
import time

MAX_CONCURRENT_WORKERS = 3

# --- Orphan sweep (2026-05-22) ------------------------------------------------

# Payload is orphan-eligible if older than this AND has no active lease.
_ORPHAN_MIN_AGE_SECONDS = 600          # 10 minutes
# Sweep budget per worker startup — don't slow down dispatch.
_SWEEP_BUDGET_SECONDS = 0.2            # 200 ms
# Max payloads to dead-letter or re-tick per sweep pass.
_SWEEP_MAX_PAYLOADS = 5
# After this many sweep ticks without successful processing, dead-letter.
_MAX_SWEEP_FAILURES = 5
# Lease file TTL — claimed payloads stay claimed up to this long.
_LEASE_TTL_SECONDS = 300               # 5 minutes


def _temp_state_dir() -> str:
    """Same TEMP_STATE_DIR the TS dispatcher writes to. Matches
    send_messages_to_letta.ts:14 (path.join(os.tmpdir(), 'letta-claude-sync-...'))."""
    base = os.environ.get("TEMP") or os.environ.get("TMP") or "/tmp"
    user = os.environ.get("USERNAME") or os.environ.get("USER") or "default"
    return os.path.join(base, f"letta-claude-sync-{user}")


def _payload_sweep_failures_path(payload_path: str) -> str:
    """Per-payload failure counter sidecar file."""
    return f"{payload_path}.sweep_failures"


def _read_sweep_failures(payload_path: str) -> int:
    try:
        with open(_payload_sweep_failures_path(payload_path), "r", encoding="utf-8") as f:
            return int(f.read().strip())
    except (OSError, ValueError):
        return 0


def _write_sweep_failures(payload_path: str, n: int) -> None:
    try:
        with open(_payload_sweep_failures_path(payload_path), "w", encoding="utf-8") as f:
            f.write(str(n))
    except OSError:
        pass


def _lease_path(payload_path: str) -> str:
    return f"{payload_path}.lease"


def _has_active_lease(payload_path: str) -> bool:
    """Active = lease file exists, started_at < TTL ago, AND pid is still alive."""
    lp = _lease_path(payload_path)
    if not os.path.exists(lp):
        return False
    try:
        with open(lp, "r", encoding="utf-8") as f:
            lease = json.load(f)
    except (OSError, ValueError, json.JSONDecodeError):
        return False
    started = lease.get("started_at_epoch", 0)
    ttl = lease.get("ttl_seconds", _LEASE_TTL_SECONDS)
    if time.time() - started > ttl:
        return False  # expired
    pid = lease.get("pid", 0)
    return _pid_alive(pid)


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    if os.name == "nt":
        try:
            import ctypes
            PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
            handle = ctypes.windll.kernel32.OpenProcess(
                PROCESS_QUERY_LIMITED_INFORMATION, False, pid
            )
            if handle == 0:
                return False
            ctypes.windll.kernel32.CloseHandle(handle)
            return True
        except Exception:
            return False
    else:
        try:
            os.kill(pid, 0)
            return True
        except (OSError, ProcessLookupError):
            return False


def _move_to_dead_letter(payload_path: str) -> None:
    """Move payload + sidecars to dead_letter/."""
    dl_dir = os.path.join(_temp_state_dir(), "dead_letter")
    try:
        os.makedirs(dl_dir, exist_ok=True)
    except OSError:
        return
    base = os.path.basename(payload_path)
    try:
        os.replace(payload_path, os.path.join(dl_dir, base))
    except OSError:
        return
    # Clean up sidecars (failure counter, lease) — don't pollute dead_letter
    for sidecar in (_payload_sweep_failures_path(payload_path), _lease_path(payload_path)):
        try:
            os.unlink(sidecar)
        except OSError:
            pass


def _sweep_orphans() -> None:
    """Best-effort orphan scan. Bounded by _SWEEP_BUDGET_SECONDS.

    For each old payload with no active lease:
    - Increment its per-payload failure counter
    - If counter >= _MAX_SWEEP_FAILURES, move to dead_letter
    - Otherwise leave in place — next worker run (or this one if cap allows)
      will pick it up

    Fails open: any error here is logged-silently and the worker proceeds.
    """
    start = time.monotonic()
    try:
        temp_dir = _temp_state_dir()
        if not os.path.isdir(temp_dir):
            return
        now = time.time()
        processed = 0
        for entry in os.listdir(temp_dir):
            if (time.monotonic() - start) > _SWEEP_BUDGET_SECONDS:
                break
            if processed >= _SWEEP_MAX_PAYLOADS:
                break
            if not entry.startswith("local-payload-") or not entry.endswith(".json"):
                continue
            path = os.path.join(temp_dir, entry)
            try:
                age = now - os.path.getmtime(path)
            except OSError:
                continue
            if age < _ORPHAN_MIN_AGE_SECONDS:
                continue
            if _has_active_lease(path):
                continue
            # Orphan candidate — tick failure counter
            n = _read_sweep_failures(path) + 1
            if n >= _MAX_SWEEP_FAILURES:
                _move_to_dead_letter(path)
            else:
                _write_sweep_failures(path, n)
            processed += 1
    except Exception:
        # Sweep must never crash the worker. Fail open.
        pass


# Sweep BEFORE the concurrency cap so even cap-firing workers do janitorial work
_sweep_orphans()


def _count_live_local_workers() -> int:
    """Number of currently-running local_worker.py processes (incl. self).
    Returns 1 (self) on any error — fails open, doesn't block normal operation.
    """
    try:
        import psutil  # type: ignore
        n = 0
        for p in psutil.process_iter(["pid", "cmdline"]):
            try:
                cmdline = p.info.get("cmdline") or []
                if any("local_worker.py" in (a or "") for a in cmdline):
                    n += 1
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        return max(n, 1)
    except ImportError:
        pass

    # psutil unavailable — try platform fallback
    if os.name == "nt":
        try:
            import subprocess
            out = subprocess.run(
                ["wmic", "process", "where",
                 "CommandLine like '%local_worker.py%'", "get", "ProcessId"],
                capture_output=True, text=True, timeout=5,
            )
            return max(out.stdout.count("\n") - 1, 1)
        except Exception:
            return 1
    else:
        try:
            import subprocess
            out = subprocess.run(
                ["pgrep", "-fc", "local_worker.py"],
                capture_output=True, text=True, timeout=3,
            )
            return max(int(out.stdout.strip() or 1), 1)
        except Exception:
            return 1


# Concurrency cap check — must run before heavy imports
_live = _count_live_local_workers()
if _live > MAX_CONCURRENT_WORKERS:
    # Already enough workers in flight. Next Stop hook will get through.
    # Orphan sweep above already ran — this exit doesn't leave the temp dir worse off.
    sys.exit(0)


hermes_root = os.environ.get("HERMES_ROOT")
if not hermes_root:
    print("ERROR: HERMES_ROOT environment variable not set", file=sys.stderr)
    sys.exit(1)

# Insert hermes/aisys FIRST so `memory.*` resolves to the canonical hermes copy.
# Otherwise PYTHONPATH (which points to claude-subconscious/python/) wins and
# we end up importing a stale duplicate of the memory package.
sys.path.insert(0, os.path.join(hermes_root, "aisys"))
sys.path.insert(1, hermes_root)

from aisys.subconscious.runner import main  # noqa: E402

main(sys.argv[1] if len(sys.argv) > 1 else None)
