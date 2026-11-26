"""Qualify and reserve a fresh held-out cohort after verified camera clipping."""

import argparse
from pathlib import Path
import shutil
import time

from audit_witness_transfer_cohort import bound_design
from capture_witness_transfer import (CASE_LIMIT_S, CASE_QUOTA_BYTES, DISK_CAP_BYTES, FREE_RESERVE_BYTES, LIMIT_S,
                                      capture, one_group, prior_attempts, validate_case)
from smart_swarm.cli import ROOT, settings
from smart_swarm.episode import atomic, load, sha
from smart_swarm.rendering import configure_renderer
from smart_swarm.research_jobs import admitted

SCHEMA = "smart-swarm-fresh-witness-transfer-reframe-v2"


def archived_gate(parent, policy):
    plan, digest = bound_design(parent)
    old = load(parent / "progress.json")
    frozen = load(policy)
    failed = [row for row in old["attempts"] if row["recipe"]["split"] == "test" and row["status"] == "rejected"]
    visited = [row for row in old["attempts"] if row["recipe"]["split"] == "test"]
    if (len(visited) != 2 or not failed or frozen["capture_design_sha256"] != digest
            or frozen["held_out_likelihoods_read"] is not False):
        raise ValueError("The reframe requires the original fit and a closed capture-only left-side failure")
    for row in failed:
        path = parent / row["recipe"]["source_group"] / "receipt.json"
        case, _ = load(path)["cases"]
        view = case["scoped_view_audit"]["tank-1"]
        if case["terminal"] != "left":
            raise ValueError("The preserved failure is outside the documented left framing defect")
        if view["before_relocation"]["both_marker_frames"] or not view["after_relocation"]["both_marker_frames"]:
            raise ValueError("The preserved failure is outside the documented left framing defect")
    return plan, old, frozen


def reframe(source, seed):
    aim = [.46, source["station_aim_m"][1], source["station_aim_m"][2]]
    return {**source, "seed": seed, "source_group": f"witness-transfer-reframe-{seed}", "station_aim_m": aim}


def qualification(config, parent, policy, output):
    plan, original, _ = archived_gate(parent, policy)
    qualification_path = output / "qualification.json"
    first = next(row for row in plan["sources"] if row["split"] == "test")
    recipe = {**reframe(first, 730016), "split": "development"}
    if qualification_path.exists():
        raise ValueError("The reframe qualification was already attempted")
    output.mkdir(parents=True, exist_ok=True)
    limit = min(CASE_LIMIT_S, original["available_capture_seconds"])
    report = {"status": "running", "recipe": recipe, "new_physical_attempts": 0}
    started = time.monotonic()
    atomic(qualification_path, report)
    try:
        captured = capture(config, output / "development-qualification", recipe, "tank-1", limit)
        report["case"] = captured
        validate_case(recipe, captured)
        report["status"] = "verified"
    except (OSError, RuntimeError, ValueError) as error:
        report.update(status="rejected", reason=str(error))
        raise
    finally:
        report["wall_seconds"] = time.monotonic() - started
        atomic(qualification_path, report)
    return qualification_path


def reframe_design(parent, policy, output):
    plan, original, _ = archived_gate(parent, policy)
    qualification_path = output / "qualification.json"
    proof = load(qualification_path)
    if proof["status"] != "verified":
        raise ValueError("The corrected framing has not been qualified")
    replacement = {731080: 731160, 731081: 731161, 731090: 731170}
    sources = [reframe(row, replacement[row["seed"]]) if row["seed"] in replacement else row
               for row in plan["sources"] if row["split"] == "test"]
    charge = LIMIT_S - original["available_capture_seconds"] + proof["wall_seconds"]
    result = {"schema": SCHEMA, "sources": sources, "original_capture_design_sha256": sha(parent / "design.json"),
              "original_capture_progress_sha256": sha(parent / "progress.json"), "frozen_policy_sha256": sha(policy),
              "qualification_sha256": sha(qualification_path), "charged_before_revised_capture_s": charge,
              "original_total_wall_seconds": LIMIT_S, "implementation_sha256": sha(ROOT / "scripts/reframe_witness_transfer.py")}
    path = output / "design.json"
    if path.is_file() and load(path) != result:
        raise ValueError("The separately reserved evaluation changed after allocation")
    if not path.is_file():
        atomic(path, result)
    return result


def run(config, parent, policy, output, cases, heartbeat):
    design = reframe_design(parent, policy, output)
    attempts = prior_attempts(output, design)
    visited = {row["recipe"]["source_group"] for row in attempts}
    pending = [row for row in design["sources"] if row["source_group"] not in visited]
    for source in pending[:cases]:
        remaining = LIMIT_S - design["charged_before_revised_capture_s"] - sum(row["wall_seconds"] for row in attempts)
        estimated_bytes = len(source["owners"]) * CASE_QUOTA_BYTES
        used_bytes = sum(item.stat().st_size for root in (parent, output) for item in root.rglob("*") if item.is_file())
        if remaining > 1 and used_bytes + estimated_bytes <= DISK_CAP_BYTES and shutil.disk_usage(output).free >= FREE_RESERVE_BYTES + estimated_bytes:
            attempts.append(one_group(config, output / source["source_group"], source, remaining, heartbeat))
    remaining = max(0., LIMIT_S - design["charged_before_revised_capture_s"] - sum(row["wall_seconds"] for row in attempts))
    report = {"design_sha256": sha(output / "design.json"), "attempted_sources": len(attempts),
              "verified_sources": sum(row["status"] == "verified" for row in attempts), "remaining_wall_seconds": remaining,
              "attempts": [{k: v for k, v in row.items() if k != "cases"} for row in attempts]}
    path = output / "progress.json"
    atomic(path, report)
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-groups", type=int, choices=range(1, 7), default=1)
    parser.add_argument("--qualify", action="store_true")
    args = parser.parse_args()
    config = settings()
    configure_renderer(config)
    with admitted(config, "smart-swarm-heldout-reframe", archive_only=False) as (_, heartbeat):
        result = (qualification(config, args.original, args.policy, args.output) if args.qualify
                  else run(config, args.original, args.policy, args.output, args.max_groups, heartbeat))
    print(result)


if __name__ == "__main__":
    main()
