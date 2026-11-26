#!/usr/bin/env python3
"""Acquire reserved primary simulation tests only after validation release."""

import argparse
from pathlib import Path
import uuid

from capture_memory_validation import DEFAULT_OUTPUT, ROOT, acquire, load_slot, pending_slots
from freeze_primary_profiles import require_profiles
from primary_test_budget import TEST_LIMITS, CaptureBudgetExhausted, capture_budget, guard_space
from primary_test_release import bound_test_release
from smart_swarm.cli import settings
from smart_swarm.episode import atomic, load
from smart_swarm.research_jobs import admitted

TEST_OUTPUT = ROOT.parent.parent / "work/primary-sim-test"


def mark_capped_attempt(slot, interval, reason):
    family = TEST_OUTPUT / slot["source_group"]
    for path in family.glob("*/capture.json"):
        row = load(path)
        if row["status"] == "running" and row["reservation"].get("budget_interval_id") == interval:
            atomic(path, {**row, "status": "interrupted", "reason": reason})


def capture_slot(config, slot, grant, heartbeat, profile_path, profile):
    if len(slot["seed_candidates"]) != TEST_LIMITS["attempts_per_source"]:
        raise ValueError("The reserved source no longer has the frozen attempt allowance")
    release = bound_test_release(TEST_OUTPUT, profile_path, profile, publish=True)
    interval = uuid.uuid4().hex
    reserved = {**grant, "primary_test_release_sha256": release, "budget_interval_id": interval}

    def safe_heartbeat():
        heartbeat()
        guard_space(TEST_OUTPUT, len(slot["capture_variants"]))

    try:
        with capture_budget(TEST_OUTPUT, release):
            safe_heartbeat()
            receipt = acquire(config, TEST_OUTPUT, slot, reserved, safe_heartbeat)
    except CaptureBudgetExhausted as error:
        mark_capped_attempt(slot, interval, str(error))
        raise
    if receipt["reservation"].get("primary_test_release_sha256") != release:
        raise ValueError("The saved test recording belongs to a different frozen release")
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profiles", type=Path, required=True)
    parser.add_argument("--family", choices=tuple(f"F{index}" for index in range(1, 7)))
    parser.add_argument("--index", type=int)
    parser.add_argument("--next", type=int, metavar="COUNT")
    args = parser.parse_args()
    explicit = args.family is not None and args.index is not None and args.next is None
    batch = args.next is not None and args.family is None and args.index is None
    if not (explicit or batch):
        parser.error("Choose --family with --index, or --next COUNT")
    profile = require_profiles(args.profiles)
    if TEST_OUTPUT.resolve().is_relative_to(DEFAULT_OUTPUT.resolve()):
        raise ValueError("Primary tests require a separate recording root")
    slots = pending_slots(TEST_OUTPUT, args.next, "test") if batch else [load_slot(args.family, args.index, "test")]
    config = settings()
    for slot in slots:
        with admitted(config, slot["source_group"]) as (grant, heartbeat):
            receipt = capture_slot(config, slot, grant, heartbeat, args.profiles, profile)
        print(receipt["source_group"], receipt["status"], receipt["seed"], flush=True)
    if not slots:
        print("All reserved primary test groups are filled")


if __name__ == "__main__":
    main()
