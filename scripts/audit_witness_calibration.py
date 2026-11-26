#!/usr/bin/env python3
"""Locate the inspection ceiling in the frozen viewpoint-transfer exchange."""

import argparse
from collections import Counter
from hashlib import sha256
import math
from pathlib import Path
from statistics import mean

from smart_swarm.cli import ROOT
from smart_swarm.episode import atomic, json_bytes, load, sha
from smart_swarm.evidence_budget_replay import wire_size
from smart_swarm.evidence_conditioned import conditioned
from smart_swarm.memory import PERMUTATIONS, SLOTS, STATES, TRACKS
from smart_swarm.particle_memory import prior_weights, result, total_variation

PILOT = ROOT / "evidence/learning-pilot-v3"
SOURCES = {
    "policy": ("witness-transfer-policy-release.json", "c0d048b51af68a5c0f553a5822ce1bde719663cc5c3c14afa3f10799bfae5522"),
    "predictions": ("witness-transfer-reframed-evaluation-sealed-predictions.json", "bc7ac4ceaf162b4a5d568f28f67c2b3000a1aa808ac7e3f63c16070982d4c704"),
    "score": ("witness-transfer-reframed-evaluation.json", "205df427f080af133c3e0abc6470623ef99b0f80758fd8605ea60fd17d75f481"),
    "profile": ("primary-profiles-frozen.json", "47c6a0ceb36eb5aecff76d198bbd4f6e12bbb09533e4ea5effdaac36181fb74c"),
    "study": ("story-learning.json", "092dbe3ec06a7a645536ffa83efe99f396bc5518d03653e43e0e59669a97acce"),
}
HEADS = ("events_only", "events_plus_reveal")
WINDOWS = ("90", "120")
FIXED = "calibrated_fixed_selection"
ORIGINAL = "frozen_factor_coverage"
ADAPTIVE = ("empirical_expected_risk", "learned_channel_expected_risk")
TOLERANCE = 1e-12


def bound_inputs(project=ROOT, material=PILOT):
    paths = {key: material / pair[0] for key, pair in SOURCES.items()}
    for key, path in paths.items():
        if key != "score" and sha(path) != SOURCES[key][1]:
            raise ValueError(f"Original witness-transfer {key} receipt changed")
    policy, predictions, profile, study = (load(paths[key]) for key in ("policy", "predictions", "profile", "study"))
    config = project / "configs/primary-memory-study.json"
    if (policy["profile_sha256"] != SOURCES["profile"][1] or profile["config_sha256"] != sha(config)
            or predictions["binding"]["policy_sha256"] != SOURCES["policy"][1]
            or policy["communication_windows_ms"] != [int(value) for value in WINDOWS]
            or len(predictions["predictions"]) != 13 or policy["held_out_likelihoods_read"] is not False):
        raise ValueError("The original exchange allocation and frozen receipt disagree")
    for code, digest in policy["code_sha256"].items():
        if sha(project / code) != digest:
            raise ValueError("The original witness fusion implementation changed")
    priors = {}
    for model in study["models"]:
        kind = model["kind"]
        if model["checkpoint_sha256"] != profile["fits"][kind]["checkpoint_sha256"]:
            raise ValueError("The archived containment prior came from a different model")
        priors[kind] = model["state_prior"]
    return policy, predictions, priors, load(config)["inspect_cost"], paths["score"]


def matched_transcript(methods, window, cap):
    fixed, original = methods[FIXED], methods[ORIGINAL]
    wire = fixed["wire"]
    spent = sum(wire_size(values) for values in wire.values())
    if (wire != original["wire"] or len(wire["requests"]) != 1 or len(wire["replies"]) != 2
            or spent != fixed["serialized_bytes"] or spent != original["serialized_bytes"] or spent > cap):
        raise ValueError("The calibrated and original methods did not receive identical budgeted replies")
    request, = wire["requests"]
    recipients = request["recipients"]
    query = int(request["issued_ns"])
    ads = {ad["sender"]: ad for ad in wire["advertisements"]}
    for reply in wire["replies"]:
        ad = ads[reply["sender"]]
        arrival = int(reply["delivered_ns"])
        if (int(reply["capture_end_ns"]) != int(ad["capture_end_ns"]) or reply["factor"] != ad["factor"]
                or int(reply["capture_end_ns"]) > query or int(reply["sent_ns"]) != query
                or arrival != query + ad["delay_ns"] or arrival > query + int(window) * 1_000_000):
            raise ValueError("An archived reply does not meet its declared chronology")
    if [reply["sender"] for reply in wire["replies"]] != recipients:
        raise ValueError("The archived reply order and request recipients differ")
    return wire, ads, query, spent


def categorical_replay(wire, ads, prior, tables):
    joint = prior_weights(prior)
    updates, states, reports = [], {}, {}
    for reply in wire["replies"]:
        sender, factor = reply["sender"], reply["factor"]
        probabilities = {label: float(value) for label, value in reply["probabilities"].items()}
        observed = max(probabilities, key=probabilities.get)
        _, joint = conditioned(joint, ads[sender], tables, observed)
        states[factor] = joint
        reports[factor] = {"sender": sender, "reported_label": observed, "reported_probability": probabilities[observed],
                           "visual_support": ads[sender]["visual_support"], "delivered_ns": int(reply["delivered_ns"])}
        updates.append({"sender": sender, "delivered_ns": int(reply["delivered_ns"]), "belief": result(joint)["belief"]})
    if list(states) != ["containment", "association"]:
        raise ValueError("The analysis requires a containment reply followed by an assignment reply")
    return updates, states, reports


def factor_bounds(states):
    before = dict(zip(STATES, states["containment"], strict=True))
    after = dict(zip(STATES, states["association"], strict=True))
    track = {name: float(sum(prob for (cube, _), prob in before.items() if cube == name)) for name in TRACKS}
    after_track = {name: float(sum(prob for (cube, _), prob in after.items() if cube == name)) for name in TRACKS}
    if any(not math.isclose(value, after_track[name], abs_tol=TOLERANCE) for name, value in track.items()):
        raise ValueError("Association changed the calibrated containment marginals")
    assignment = {name: float(sum(prob for (_, key), prob in after.items() if key == name)) for name in PERMUTATIONS}
    terminals = result(states["association"])["belief"]
    return {"track_posterior": track, "assignment_posterior": assignment,
            "max_side_with_perfect_assignment": max(track.values()),
            "max_side_with_perfect_containment": max(assignment.values()),
            "max_side_from_observed_replies": max(terminals[side] for side in SLOTS),
            "assignment_update_joint_total_variation": total_variation(states["containment"], states["association"])}


def replay_case(methods, window, prior, tables, inspect_cost, cap):
    wire, ads, query, spent = matched_transcript(methods, window, cap)
    updates, states, reports = categorical_replay(wire, ads, prior, tables)
    final = result(states["association"])
    winner = max(final["belief"], key=final["belief"].get)
    decision = winner if 1. - final["belief"][winner] <= inspect_cost else "request_inspection"
    fixed = methods[FIXED]
    if fixed["updates"] != updates or fixed["decision"] != decision or any(fixed[key] != value for key, value in final.items()):
        raise ValueError("Replayed categorical fusion differs from the original frozen prediction")
    return {**factor_bounds(states), "categorical_decision": decision, "categorical_belief": final["belief"],
            "original_decision": methods[ORIGINAL]["decision"], "original_belief": methods[ORIGINAL]["belief"],
            "modeled_latest_reply_ms": (max(item["delivered_ns"] for item in reports.values()) - query) / 1e6,
            "reply_reports": reports, "fixed_exchange_bytes": spent,
            "adaptive_reply_factors": {name: [reply["factor"] for reply in methods[name]["wire"]["replies"]]
                                       for name in ADAPTIVE}}


def forecast_diagnostics(policy, predictions, priors, inspect_cost):
    rows = []
    identity = ("path", "source_group", "manifest_sha256")
    for row in predictions["predictions"]:
        windows = {kind: {window: replay_case(row["methods"][kind][window], window, priors[kind],
                         policy["heads"][kind]["empirical_channels"], inspect_cost, policy["byte_cap"]) for window in WINDOWS}
                   for kind in HEADS}
        rows.append({**{key: row[key] for key in identity}, "diagnostics": windows})
    return rows


def score_diagnostics(rows, predictions, score):
    if (score["predictions_sha256"] != SOURCES["predictions"][1] or score["policy_sha256"] != SOURCES["policy"][1]
            or score["test_capture_design_sha256"] != predictions["binding"]["test_design_sha256"]
            or score["independent_test_sources"] != 6 or len(score["case_results"]) != len(rows)):
        raise ValueError("The evaluator receipt is unrelated to the frozen predictions")
    labeled = []
    for row, sealed, case in zip(rows, predictions["predictions"], score["case_results"], strict=True):
        if any(case.get(key) != value for key, value in sealed.items()):
            raise ValueError("A saved evaluator row changed its pre-inspection forecast")
        labeled.append({**row, "terminal": case["truth"], "assignment": case["evaluator_assignment"]})
    return labeled


def grouped_correct(cases):
    groups = {}
    for row, correct in cases:
        groups.setdefault(row["source_group"], []).append(correct)
    return mean(mean(observed) for observed in groups.values())


def summarize_window(rows, kind, window, threshold):
    containers = [(row, row["diagnostics"][kind][window]) for row in rows if row["terminal"] in SLOTS]
    data = [item for _, item in containers]
    local = [(row, item["reply_reports"]["association"]["reported_label"] == row["assignment"]) for row, item in containers]
    supported = [(row, correct) for (row, correct), item in zip(local, data, strict=True)
                 if item["reply_reports"]["association"]["visual_support"]]
    limits = {name: {"maximum": max(item[name] for item in data),
                     "variants_reaching_threshold": sum(item[name] >= threshold for item in data)} for name in
              ("max_side_with_perfect_assignment", "max_side_with_perfect_containment", "max_side_from_observed_replies")}
    decisions = Counter("correct" if item["original_decision"] == row["terminal"] else
                        "inspection" if item["original_decision"] == "request_inspection" else "wrong" for row, item in containers)
    return {"container_variants": len(containers), "independent_container_groups": len({row["source_group"] for row, _ in containers}),
            "selected_assignment_replies_correct": sum(correct for _, correct in local),
            "source_weighted_assignment_accuracy": grouped_correct(local), "supported_selected_replies": len(supported),
            "supported_selected_replies_correct": sum(correct for _, correct in supported),
            "latest_reply_ms": max(item["modeled_latest_reply_ms"] for item in data), "limits": limits,
            "categorical_inspections": sum(item["categorical_decision"] == "request_inspection" for item in data),
            "association_changed_joint": sum(item["assignment_update_joint_total_variation"] > TOLERANCE for item in data),
            "original_container_decisions": dict(decisions),
            "adaptive_association_requests": {method: sum("association" in item["adaptive_reply_factors"][method] for item in data)
                                              for method in ADAPTIVE}}


def audit(project=ROOT, material=PILOT):
    policy, predictions, priors, inspect_cost, score = bound_inputs(project, material)
    rows = forecast_diagnostics(policy, predictions, priors, inspect_cost)
    digest = sha256(json_bytes(rows)).hexdigest()
    if sha(score) != SOURCES["score"][1]:
        raise ValueError("The original witness-transfer evaluator receipt changed")
    labeled = score_diagnostics(rows, predictions, load(score))
    threshold = 1. - inspect_cost
    return {"scope": "Exploratory analysis of the already scored witness transfer cohort",
            "inputs_sha256": {key: pair[1] for key, pair in SOURCES.items()}, "diagnostic_prelabel_sha256": digest,
            "independent_groups": len({row["source_group"] for row in rows}), "variants": len(rows),
            "decision_threshold": threshold, "new_training_updates": 0, "new_recordings": 0, "new_model_inference": 0,
            "matched_methods": [FIXED, ORIGINAL], "cases": labeled,
            "summary": {kind: {window: summarize_window(labeled, kind, window, threshold) for window in WINDOWS} for kind in HEADS}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Choose a new immutable audit result path")
    atomic(args.output, audit())
    print(args.output)


if __name__ == "__main__":
    main()
