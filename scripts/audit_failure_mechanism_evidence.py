#!/usr/bin/env python3
"""Replay which frozen replies support surface answers across two native MuJoCo failures."""

import argparse
from hashlib import sha256
from pathlib import Path

from audit_witness_calibration import (ADAPTIVE, FIXED, HEADS, SOURCES, bound_inputs, categorical_replay,
                                       matched_transcript, score_diagnostics)
from smart_swarm.episode import TRUTH, atomic, json_bytes, load, rows, sha
from smart_swarm.evidence_conditioned import conditioned
from smart_swarm.particle_memory import prior_weights, result

WINDOW = "90"
TASKS = ("missed_grasp", "early_release")


def posterior_decision(joint, inspect_cost):
    belief = result(joint)["belief"]
    winner = max(belief, key=belief.get)
    return {"belief": belief, "decision": winner if 1. - belief[winner] <= inspect_cost else "request_inspection"}


def replay_replies(methods, prior, tables, inspect_cost, cap):
    wire, adverts, _, spent = matched_transcript(methods, WINDOW, cap)
    updates, states, reports = categorical_replay(wire, adverts, prior, tables)
    arm, tank = wire["replies"]
    initial = prior_weights(prior)
    _, assignment_only = conditioned(initial, adverts[tank["sender"]], tables, reports["association"]["reported_label"])
    context = {name: posterior_decision(joint, inspect_cost) for name, joint in (
        ("no_delivered_replies", initial), ("only_arm_containment", states["containment"]),
        ("only_tank_assignment", assignment_only), ("both_replies", states["association"]))}
    fixed = methods[FIXED]
    if updates != fixed["updates"] or context["both_replies"] != {key: fixed[key] for key in ("belief", "decision")}:
        raise ValueError("Dropping neither reply must reproduce the archived decision")
    saved = {name: {"decision": methods[name]["decision"], "belief": methods[name]["belief"],
                    "reply_factors": [reply["factor"] for reply in methods[name]["wire"]["replies"]]} for name in ADAPTIVE}
    return {"conditions": context, "arm_surface_probability": float(arm["probabilities"]["surface"]),
            "arm_report": reports["containment"], "tank_report": reports["association"],
            "transmitted_bytes_unchanged": spent, "saved_adaptive": saved}


def prelabel_replay(policy, predictions, priors, cost):
    fields = ("path", "source_group", "manifest_sha256")
    return [{**{key: row[key] for key in fields},
             "replies": {head: replay_replies(row["methods"][head][WINDOW], priors[head],
                                             policy["heads"][head]["empirical_channels"], cost, policy["byte_cap"])
                         for head in HEADS}}
            for row in predictions["predictions"]]


def verified_physics(trace, task, gate):
    times = [int(point["at"]["monotonic_ns"]) for point in trace]
    phases = {point["teacher_phase"] for point in trace}
    if times != sorted(set(times)) or not {"lift", "release", "post_relocation"} <= phases:
        raise ValueError("A failure trace lacks ordered physical events")
    end, = (point for point in trace if point["teacher_phase"] == "post_relocation")
    carry = [point for point in trace if point["teacher_phase"] in ("lift", "transfer", "lower")
             and len(point["gripper_contact_bodies"]) >= 2]
    release = [point for point in trace if point["teacher_phase"] == "release"]
    bilateral = sum(len(point["gripper_contact_bodies"]) >= 2 for point in trace)
    height = max(point["cube_height_above_table_m"] for point in trace)
    supported = len(carry)
    if (gate["carry_contact_samples"] != supported or gate["first_insertion"] or gate["final_relation"] != "surface"
            or any(point["inside_carrier"] for point in trace) or end["cube_relation"] != "surface"
            or not end["stationary"] or not end["released"]):
        raise ValueError("Contact samples do not establish failed containment and a settled surface outcome")
    if task == "missed_grasp" and (bilateral or height > .03):
        raise ValueError("The missed-grasp source shows a supported or elevated target")
    if task == "early_release" and (supported < 5 or height < .10 or trace.index(carry[-1]) >= trace.index(release[0])
                                   or any(len(point["gripper_contact_bodies"]) >= 2 for point in release)):
        raise ValueError("The early-release source lacks supported carry followed by release")
    return {"supported_lift_carry_samples": supported, "bilateral_samples_all": bilateral,
            "maximum_cube_height_m": height, "verified_terminal": end["cube_relation"]}


def join_physics(capture, scored, forecast, source):
    receipt = load(capture / scored["source_group"] / "receipt.json")
    if receipt["status"] != "verified" or receipt["recipe"] != source or source["task"] != scored["task"]:
        raise ValueError("The failure source does not match the frozen capture recipe")
    path = Path(scored["path"])
    native, = (case for case in receipt["cases"] if Path(case["path"]) == path)
    trace = path / TRUTH / "physics.jsonl"
    if (not path.resolve().is_relative_to((capture / scored["source_group"]).resolve())
            or native["manifest_sha256"] != scored["manifest_sha256"] or sha(path / "manifest.json") != native["manifest_sha256"]
            or sha(trace) != native["contact_trace_sha256"] or native["terminal"] != scored["truth"]
            or native["owner"] != scored["evaluator_visibility_condition"]):
        raise ValueError("The physics receipt does not match the original scored variant")
    return {**forecast, "task": source["task"], "owner_condition": native["owner"],
            "contact_trace_sha256": native["contact_trace_sha256"],
            "physics": verified_physics(rows(trace), source["task"], native["physical_gate"])}


def summarize_failures(cases):
    summary = {}
    for task in TASKS:
        variants = [row for row in cases if row["task"] == task]
        groups = {row["source_group"] for row in variants}
        if len(groups) != 1 or len(variants) != 2 or len({row["owner_condition"] for row in variants}) != 2:
            raise ValueError("Expected one held-out source and two scoped variants per failure mechanism")
        summary[task] = {"independent_groups": len(groups), "camera_variants": len(variants),
                         "supported_lift_carry_samples": [v["physics"]["supported_lift_carry_samples"] for v in variants],
                         "surface_answers_by_head_and_reply_condition": {
                             head: {name: sum(row["replies"][head]["conditions"][name]["decision"] == "surface" for row in variants)
                                    for name in variants[0]["replies"][head]["conditions"]} for head in HEADS}}
    return summary


def audit(capture):
    policy, predictions, priors, cost, score_path = bound_inputs()
    forecasts = prelabel_replay(policy, predictions, priors, cost)
    prelabel = sha256(json_bytes(forecasts)).hexdigest()
    if sha(score_path) != SOURCES["score"][1] or sha(capture / "design.json") != predictions["binding"]["test_design_sha256"]:
        raise ValueError("The original evaluator or held-out capture design changed")
    evaluator = load(score_path)
    score_diagnostics(forecasts, predictions, evaluator)
    sources = {row["source_group"]: row for row in load(capture / "design.json")["sources"]}
    cases = [join_physics(capture, scored, forecast, sources[scored["source_group"]])
             for forecast, scored in zip(forecasts, evaluator["case_results"], strict=True)
             if scored["truth"] == "surface" and scored["task"] in TASKS]
    return {"scope": "Exploratory reply-drop replay on the two previously scored surface sources",
            "reply_window_ms": int(WINDOW), "diagnostic_prelabel_sha256": prelabel,
            "predictions_sha256": SOURCES["predictions"][1], "score_sha256": SOURCES["score"][1],
            "capture_design_sha256": predictions["binding"]["test_design_sha256"],
            "summary": summarize_failures(cases), "cases": cases,
            "new_model_inference": 0, "optimizer_updates": 0, "new_recordings": 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Choose a new immutable audit result path")
    atomic(args.output, audit(args.capture_root.resolve()))
    print(args.output)


if __name__ == "__main__":
    main()
