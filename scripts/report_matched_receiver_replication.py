#!/usr/bin/env python3
"""Verify the matched-receiver development recordings and summarize frozen forecasts."""

import argparse
from hashlib import sha256
from pathlib import Path

from rehearse_matched_histories import station_observation
from score_primary_validation import CURRENT, SELECTIVE, TEMPORAL
from smart_swarm.cli import ROOT
from smart_swarm.episode import TRUTH, atomic, json_bytes, load, rows, sha, validate
from smart_swarm.evidence_inputs import runtime_signature

LEAVES = ("insertion-witness", "failed-placement-witness")
KINDS = ("events_only", "events_plus_reveal")
SAME = {"so101-1": False, "tank-1": True, "rosmaster-1": True}
FIXED = ("query", "query_ns", "decision_ns", "camera_grants", "runtime_cache", "seed", "unseen_subscriptions")


def read_execution(root):
    design_path = root / "design.json"
    design, execution = load(design_path), load(root / "execution.json")
    profiles = ROOT / "evidence/learning-pilot-v3/primary-profiles-frozen.json"
    allocation = ROOT / "evidence/learning-pilot-v3/primary-sim-source-allocation.json"
    if (design["schema"] != "smart-swarm-matched-receiver-development-v2" or design["actor_count"] != 3
            or execution["design_sha256"] != sha(design_path) or design["profile_sha256"] != sha(profiles)
            or design["disjoint_primary_seed_allocation_sha256"] != sha(allocation)
            or design["capture_and_forecast_worker_sha256"] != sha(ROOT / "scripts/rehearse_matched_histories.py")):
        raise ValueError("The matched-receiver allocation or frozen profile changed")
    if [row["seed"] for row in execution["attempts"]] != design["seeds"]:
        raise ValueError("The execution ledger does not match the ordered allocation")
    return design, execution


def verified_signature(paths):
    first, second = [runtime_signature(path) for path in paths]
    same = {key: first["private_history"][key] == second["private_history"][key]
            for key in first["private_history"]}
    if same != SAME or any(first[key] != second[key] for key in FIXED):
        raise ValueError("The archived private observations, grants, or query timing escaped the matched control")
    return {"same_private_histories": same, "query_ns": first["query_ns"], "decision_ns": first["decision_ns"],
            "receiver_history_sha256": sha256(json_bytes(first["private_history"]["rosmaster-1"])).hexdigest(),
            "tank_history_sha256": sha256(json_bytes(first["private_history"]["tank-1"])).hexdigest()}


def verified_record(path, declared, seed, family):
    manifest_path = path / "manifest.json"
    manifest = load(manifest_path)
    validate(path, manifest)
    gate = manifest["physical_story_gate"]
    reveal, = rows(path / TRUTH / "reveal.jsonl")
    outcome = reveal["ground_truth"]
    if (manifest["pair_seed"] != seed or manifest["simulation_recipe_seed"] != seed or manifest["family"] != family
            or manifest["run_mode"] != "mujoco" or not manifest["expanded_station_view"]
            or gate != declared["physical_checks"] or outcome != declared["outcome"]
            or not reveal["relations_preserved_until_reveal"]):
        raise ValueError("The archived recording differs from its capture or reveal receipt")
    contained = gate["first_insertion"] and gate["carry_contact_samples"] > 0 and gate["final_relation"] == "carrier"
    missed = not gate["first_insertion"] and gate["carry_contact_samples"] == 0 and gate["final_relation"] == "surface"
    if not ((family == "F1" and contained and outcome in ("left", "right"))
            or (family == "F2" and missed and outcome == "surface")):
        raise ValueError("The archived pair does not support a contact-valid placement and a controlled failure")
    witness = station_observation(path)
    position = witness["last_position"]
    witness["last_position"] = list(position) if position is not None else None
    if witness != declared["station_observation"]:
        raise ValueError("The captured station visibility changed")
    return {"family": family, "outcome": outcome, "capture_manifest_sha256": sha(manifest_path),
            "station_observation": witness, "physical_checks": gate, "broadcast_rule": declared["rule_decision"]}


def head_result(predictions, outcomes):
    first, second = predictions
    if any(first[method] != second[method] for method in (CURRENT, TEMPORAL)):
        raise ValueError("A receiver-private policy did not preserve its frozen forecast")
    selective = [row[SELECTIVE] for row in predictions]
    wire = [row["wire"]["replies"] for row in selective]
    containment = [[message["probabilities"] for message in replies
                    if message["sender"] == "so101-1" and message["factor"] == "containment"] for replies in wire]
    tank = [[message for message in replies if message["sender"] == "tank-1"] for replies in wire]
    decisions = [row["decision"] for row in selective]
    return {"receiver_current_decision": first[CURRENT]["decision"], "receiver_temporal_decision": first[TEMPORAL]["decision"],
            "selective_decisions": decisions, "selective_posterior": [row["belief"] for row in selective],
            "correct_answer_both_conditions": decisions == outcomes, "so101_containment_replies": containment,
            "tank_reply_unchanged": tank[0] == tank[1], "serialized_bytes": [row["serialized_bytes"] for row in selective]}


def scored_pair(root, attempt, design):
    seed = attempt["seed"]
    directory = root / str(seed)
    path = directory / "development-pair.json"
    forecast = directory / "predictions.json"
    receipt = load(path)
    if (receipt["status"] != "scored" or receipt["seed"] != seed or receipt["independent_source_groups"] != 1
            or receipt["profile_sha256"] != design["profile_sha256"]
            or receipt["predictions_sha256"] != sha(forecast) or receipt["shared_private_scopes"] != SAME
            or receipt["reservation"]["device"] != "cpu" or receipt["reservation"]["gpu_render"]):
        raise ValueError("The archived group does not bind the frozen predictions and CPU admission")
    predictions = load(forecast)
    paths = [directory / leaf for leaf in LEAVES]
    matched = verified_signature(paths)
    records = [verified_record(raw, record, seed, family) for raw, record, family
               in zip(paths, receipt["records"], ("F1", "F2"), strict=True)]
    outcomes = [record["outcome"] for record in records]
    return {**attempt, **matched, "records": records, "prediction_sha256": sha(forecast),
            "recording_receipt_sha256": sha(path), "heads": {kind: head_result(predictions[kind], outcomes) for kind in KINDS}}


def report(root):
    design, execution = read_execution(root)
    rows = []
    for attempt in execution["attempts"]:
        seed = attempt["seed"]
        if "log_sha256" in attempt and sha(root / f"{seed}.log") != attempt["log_sha256"]:
            raise ValueError("A native runner log changed")
        if attempt.get("returncode") == 0:
            rows.append(scored_pair(root, attempt, design))
        else:
            path = root / str(seed) / "development-pair.json"
            rows.append({**attempt, "retained_capture_receipt_sha256": sha(path) if path.is_file() else None})
    qualified = [row for row in rows if "heads" in row]
    return {"scope": design["scope"], "design_sha256": execution["design_sha256"], "simulated": True,
            "independent_matched_source_groups": len(qualified), "scored_variants": 2 * len(qualified),
            "both_correct_with_broadcast_rule": sum(all(case["outcome"] == case["broadcast_rule"] for case in row["records"])
                                                   for row in qualified),
            "both_correct_by_head": {kind: sum(row["heads"][kind]["correct_answer_both_conditions"] for row in qualified)
                                     for kind in KINDS}, "sources": rows, "optimizer_updates": 0,
            "primary_cohort_included": False, "charged_wall_s": execution["charged_wall_s"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = report(args.root.resolve())
    if args.output.is_file() and load(args.output) != result:
        parser.error("Choose a new report path to preserve the existing result")
    if not args.output.exists():
        atomic(args.output, result)
    print(args.output)


if __name__ == "__main__":
    main()
