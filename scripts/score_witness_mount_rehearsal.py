"""Evaluate frozen likelihoods when simulated relocation visibility changes owner."""

import argparse
from pathlib import Path

import torch

from audit_screened_association import STUDY, model_parameters
from audit_team_evidence import decode_features
from compare_compact_witness_controls import custom_selection, score_wire, transcript
from freeze_primary_profiles import KINDS, PILOT, require_profiles
from score_primary_test import runtime
from score_primary_validation import PRIMARY_CAP
from smart_swarm.cli import settings
from smart_swarm.episode import LABELS, TRUTH, atomic, load, rows, sha, validate
from smart_swarm.primary_inputs import primary_policy_view
from smart_swarm.research_jobs import admitted
from smart_swarm.team_evidence import select
from verify_mobile_frozen_head_reuse import clips, shared_contexts

PROFILE = PILOT / "primary-profiles-frozen.json"


def scored_case(path, contexts):
    with primary_policy_view(path) as scoped:
        window, features, markers = clips(scoped, contexts)
    query, deadline, ads = window
    updated = [dict(ad, color_support=bool(markers[ad["sender"]]["supported"])) if ad["factor"] == "association"
               else dict(ad, color_support=False) for ad in ads]
    shared = (query, deadline, updated)
    results = {}
    for kind, context in contexts.items():
        common = context["common"]
        gains = common["gains"]
        forced = select("factor_coverage", updated, query, deadline, PRIMARY_CAP, gains)
        eligible = sorted(updated, key=lambda ad: (-gains[ad["factor"]], not ad["color_support"],
                                                   -int(ad["capture_end_ns"]), ad["sender"]))
        visible = custom_selection(shared, eligible)
        _, head, temperature = context["network"]
        predictions = {sender: decode_features(feature, factor, head, temperature)
                       for sender, (factor, feature) in features.items()}
        arms = {method: score_wire(transcript(shared, chosen, predictions), shared, context["mounts"]["4"], common)
                for method, chosen in (("factor_coverage", forced), ("visual_quality", visible))}
        if any(not arm["within_cap"] for arm in arms.values()):
            raise ValueError("A compact forecast exceeded the matched envelope")
        results[kind] = {"arms": arms, "unfused_witness_probabilities": predictions}
    label, = rows(path / LABELS / "settled-event.jsonl")
    return {"input_manifest_sha256": sha(path / "manifest.json"), "correct_assignment_for_scoring": label["relocation_assignment"],
            "marker_support": markers, "results": results}


def audit(source, grant, heartbeat):
    design = load(source)
    if design["status"] != "verified" or any(not group["verified"] for group in design["groups"]):
        raise ValueError("Only the completed contact-valid mount pairs can be compared")
    torch.set_num_threads(3)
    profile = require_profiles(PROFILE)
    contexts = shared_contexts({kind: runtime(profile, kind) for kind in KINDS})
    outcomes = []
    for group in design["groups"]:
        for case in group["cases"]:
            heartbeat()
            episode = Path(case["path"])
            manifest = load(episode / "manifest.json")
            validate(episode, manifest)
            if (manifest["scene_group_id"] != f"witness-mount-development-{group['seed']}"
                    or sha(episode / TRUTH / "physics.jsonl") != case["contact_trace_sha256"]):
                raise ValueError("The scored scene no longer matches its physical verification receipt")
            result = scored_case(episode, contexts)
            supported = {name for name, row in result["marker_support"].items() if row["supported"]}
            if supported != {case["owner"]}:
                raise ValueError("The frozen clip did not retain the uniquely informative camera")
            outcomes.append({**result, "source_group_seed": group["seed"], "visible_camera_for_scoring": case["owner"],
                             "terminal_outcome_for_scoring": case["outcome"]})
    return {"scope": "Contact-validated MuJoCo development pairs with new camera mounts", "optimizer_updates": 0,
            "new_physical_attempts": 0, "profiles_sha256": sha(PROFILE), "source_receipt_sha256": sha(source),
            "reservation": {key: grant[key] for key in ("scope", "locks")}, "cases": outcomes}


def calibration_diagnostic(source, replay, heartbeat):
    design, prior = load(source), load(replay)
    if prior["source_receipt_sha256"] != sha(source) or prior["profiles_sha256"] != sha(PROFILE):
        raise ValueError("The replay does not belong to the verified mount pairs and frozen profile")
    profile, study = require_profiles(PROFILE), load(STUDY)
    episodes = [case for group in design["groups"] for case in group["cases"]]
    findings = []
    for kind in KINDS:
        heartbeat()
        common, mounts = model_parameters(profile, study, kind)
        fit = mounts["4"]
        compact = fit["association"]["compact"]
        previous = compact["tank-2"]["slope"]
        revised = {**compact, "tank-2": {**compact["tank-2"], "slope": 1.}}
        identity = {**fit, "association": {**fit["association"], "compact": revised}}
        for case, saved in zip(episodes, prior["cases"], strict=True):
            if case["owner"] == "tank-2":
                path = Path(case["path"]) / "manifest.json"
                current = saved["results"][kind]["arms"]["visual_quality"]
                manifest = load(path)
                query, stop = (int(manifest[key]["monotonic_ns"]) for key in ("query_boundary", "decision_boundary"))
                if saved["input_manifest_sha256"] != sha(path) or score_wire(current["wire"], (query, stop, []), fit, common) != current:
                    raise ValueError("The original calibrated result could not be reproduced")
                intervention = score_wire(current["wire"], (query, stop, []), identity, common)
                findings.append({"model": kind, "source_group_seed": saved["source_group_seed"], "original_tank2_slope": previous,
                                 "assignment_for_scoring": saved["correct_assignment_for_scoring"],
                                 "outcome_for_scoring": case["outcome"], "frozen": current, "unscaled_tank2": intervention})
    return {"scope": "post-outcome development diagnostic using identical previously saved packets",
            "replay_sha256": sha(replay), "fitted_parameters": 0, "changed_input": "Tank-2 association slope set to 1",
            "cases": findings}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rehearsal", type=Path, required=True)
    parser.add_argument("--calibration-replay", type=Path, help="Diagnose attenuation using the already sealed compact packets")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with admitted(settings(), "smart-swarm-frozen-witness-mount-replay", archive_only=True) as (grant, heartbeat):
        result = (calibration_diagnostic(args.rehearsal, args.calibration_replay, heartbeat) if args.calibration_replay
                  else audit(args.rehearsal, grant, heartbeat))
    atomic(args.output, result)
    print(args.output)


if __name__ == "__main__":
    main()
