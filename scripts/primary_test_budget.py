"""Cumulative limits for the prospective primary test capture."""

from contextlib import contextmanager
from pathlib import Path
import shutil
import signal
import time

from smart_swarm.episode import atomic, load

TEST_LIMITS = {"wall_seconds": 28_800, "disk_bytes": 1024**3, "rollout_timeout_seconds": 100,
               "attempts_per_source": 5}
ROLLOUT_DISK_RESERVE = 116 * 1024**2
NANOS = 1_000_000_000
LEDGER = "primary-test-capture-budget.json"


class CaptureBudgetExhausted(RuntimeError):
    """A documented acquisition cap was reached."""


def guard_space(root, rolls):
    used = sum(path.stat().st_size for path in root.rglob("*") if path.is_file())
    reserve = rolls * ROLLOUT_DISK_RESERVE
    if used + reserve > TEST_LIMITS["disk_bytes"] or reserve > shutil.disk_usage(root).free:
        raise CaptureBudgetExhausted("Primary test capture cannot reserve the next rollout's disk allowance")


def boot_identity():
    return Path("/proc/sys/kernel/random/boot_id").read_text().strip()


def restore_ledger(path, cap_ns, release, boot):
    row = load(path) if path.is_file() else {"used_wall_ns": 0, "limits": TEST_LIMITS, "primary_test_release_sha256": release}
    if row["limits"] != TEST_LIMITS or row["primary_test_release_sha256"] != release:
        raise ValueError("The test capture budget belongs to a different release or limit")
    interrupted = row.pop("active_capture_wall_ns", None)
    elapsed_start = row.pop("active_capture_monotonic_ns", None)
    original_boot = row.pop("active_capture_boot_id", None)
    if interrupted is not None:
        if original_boot == boot and elapsed_start is not None:
            elapsed = max(0, time.monotonic_ns() - elapsed_start)
        else:
            elapsed = max(0, time.time_ns() - interrupted)
        row["used_wall_ns"] = min(cap_ns, row["used_wall_ns"] + elapsed)
    return row


def alarm(signum, frame):
    raise CaptureBudgetExhausted("Primary test capture reached its cumulative wall-time cap")


@contextmanager
def capture_budget(root, release):
    path = root / LEDGER
    cap_ns = TEST_LIMITS["wall_seconds"] * NANOS
    boot = boot_identity()
    row = restore_ledger(path, cap_ns, release, boot)
    started = time.monotonic_ns()
    remaining = cap_ns - row["used_wall_ns"]
    installed = False
    failure = None
    try:
        if remaining <= 0:
            raise CaptureBudgetExhausted("Primary test capture reached its cumulative wall-time cap")
        previous = signal.signal(signal.SIGALRM, alarm)
        installed = True
        signal.setitimer(signal.ITIMER_REAL, remaining / NANOS)
        atomic(path, {**row, "status": "capturing", "active_capture_wall_ns": time.time_ns(),
                      "active_capture_monotonic_ns": started, "active_capture_boot_id": boot})
        yield
    except CaptureBudgetExhausted as error:
        failure = str(error)
        raise
    finally:
        if installed:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, previous)
        row["used_wall_ns"] = min(cap_ns, row["used_wall_ns"] + time.monotonic_ns() - started)
        atomic(path, {**row, "status": "cap_reached" if failure else "between_sources", "reason": failure})
