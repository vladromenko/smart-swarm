"""Verify the two post-evaluation diagnostics in the revised paper package."""

import math
from pathlib import Path
import tarfile

from archive_primary_release import read_member
from audit_witness_calibration import SOURCES, audit
from smart_swarm.episode import load, sha

ACT = "act-grasp-schedule-development-v2.json"
WITNESS = "witness-transfer-calibration-limits-v1.json"
ACT_INPUT = "act-clock-closure-window.json"
STUDY = SOURCES["study"][0]
MATERIAL = (ACT, WITNESS, ACT_INPUT, STUDY)
HASHES = {ACT: "2ca4efc4e7cff4c4864d240a3875d4772c7186dbf1c1e08213bf392fb8e1223a",
          WITNESS: "10b10ddd09351c7368405bb8d1cf3c65afbe1bcf8b202a2d89f5bafdeb8eba77",
          ACT_INPUT: "7562a4112b80aa6f14a9b238346238faeb9ac4f3e5a48ca4fb777e47228e2339",
          STUDY: SOURCES["study"][1]}
EVIDENCE = Path("evidence/learning-pilot-v3")


def gather_witness_inputs(evidence, working, archives):
    material = working / "witness-calibration-inputs"
    material.mkdir()
    with tarfile.open(archives["witness-transfer"]) as archive:
        for key in ("policy", "predictions", "score"):
            name, _ = SOURCES[key]
            (material / name).write_bytes(read_member(archive, f"results/{name}"))
    name, _ = SOURCES["profile"]
    with tarfile.open(archives["primary-frozen-calibration"]) as archive:
        (material / name).write_bytes(read_member(archive, f"calibration/{name}"))
    (material / STUDY).write_bytes((evidence / STUDY).read_bytes())
    return material


def act_branch_metrics(branch, window):
    trace = branch["grasp_observations"]
    if [row["after_issue"] for row in trace] != list(window) or not branch["sample_times_exact"]:
        raise ValueError("An ACT intervention lost an aligned grasp observation")
    first = next((row["after_issue"] for row in trace if row["bilateral"]), None)
    placed = (branch["stable_containment"] and branch["held_lift_10cm"] and branch["first_failed_stage"] is None)
    if (first != branch["first_bilateral_contact_after_issue"] or (first in window) != branch["first_bilateral_contact_during_patch"]
            or placed != branch["contact_valid_placement"] or branch["precontact_state_max_abs_residual"] > 1e-4):
        raise ValueError("An ACT intervention outcome disagrees with its saved replay samples")
    return first, placed


def verify_act_diagnostic(evidence):
    result, archived = load(evidence / ACT), load(evidence / ACT_INPUT)
    if (result["archive_sha256"] != HASHES[ACT_INPUT] or result["fit_sha256"] != archived["fit_receipt_sha256"]
            or result["window_half_open"] != archived["offline_issued_interval"] or result["model_inference"]
            or result["fresh_resets_read"] or result["optimizer_updates"] or len(result["cases"]) != 7
            or not math.isclose(result["cumulative_charged_wall_s"], result["elapsed_s"] + result["prior_charge_s"])
            or result["cumulative_charged_wall_s"] > result["wall_limit_s"]):
        raise ValueError("The ACT replay differs from its original archive scope or allowance")
    for row, source in zip(result["cases"], archived["sources"], strict=True):
        if (row["seed"], row["split"], row["teacher_sha256"]) != (source["seed"], source["split"], source["source_sha256"]):
            raise ValueError("An ACT intervention cites a different teacher source")
    metrics = {}
    window = range(*result["window_half_open"])
    for name in result["branches"]:
        branches = [row["branches"][name] for row in result["cases"]]
        first, placed = zip(*(act_branch_metrics(branch, window) for branch in branches), strict=True)
        derived = {"contact_during_patch": sum(item in window for item in first), "contact_valid_placements": sum(placed),
                   "first_bilateral_after_issue": list(first), "first_failed_stage": [row["first_failed_stage"] for row in branches]}
        if derived != result["summary"][name]:
            raise ValueError("The packaged ACT summary differs from its source samples")
        metrics[name] = derived
    original = metrics["teacher"]["first_bilateral_after_issue"]
    first = metrics["archived_act_gripper"]["first_bilateral_after_issue"]
    delay = [mixed - teacher for mixed, teacher in zip(first, original, strict=True)]
    return {"placements_by_branch": {name: row["contact_valid_placements"] for name, row in metrics.items()},
            "act_contact_delay_issue_range": [min(delay), max(delay)], "development_sources": len(result["cases"])}


def verify_diagnostics(root, working, archives):
    evidence = root / EVIDENCE
    for name, digest in HASHES.items():
        if sha(evidence / name) != digest:
            raise ValueError(f"The original paper diagnostic changed at {name}")
    scores = audit(root, gather_witness_inputs(evidence, working, archives))
    if scores != load(evidence / WITNESS):
        raise ValueError("The viewpoint calibration limit no longer reproduces from the archived messages")
    return {"witness_calibration": {"independent_groups": scores["independent_groups"], "variants": scores["variants"],
                                    "prediction_and_bounds_reproduced": True},
            "act_gripper_schedule": verify_act_diagnostic(evidence)}
