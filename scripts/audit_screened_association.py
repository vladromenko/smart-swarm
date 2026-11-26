#!/usr/bin/env python3
"""Attribute screened F4 choices using sealed validation packets and frozen fusion."""

import argparse
from collections import Counter
from pathlib import Path

from audit_evidence_replay import verified_model
from compare_primary_supervision import (
    KINDS, TEAMS, authority, evaluator_record, prediction_path, read_scores, score_paths, verify_scores,
)
from freeze_primary_profiles import PILOT
from score_primary_validation import IDENTITY, SELECTIVE, choose, likelihoods_from_reports, on_time_compact
from smart_swarm.episode import atomic, load, sha
from smart_swarm.evidence_comparison import learned_prior_map
from smart_swarm.memory import PERMUTATIONS, SLOTS, STATES, TRACKS
from smart_swarm.particle_memory import exact_filter, total_variation

PROFILE = PILOT / "primary-profiles-frozen.json"
STUDY = PILOT / "story-learning.json"
TOLERANCE = 1e-12
INSPECT = "request_inspection"
INTERVENTIONS = {"full": (), "without_association": ("association",),
                 "without_containment": ("containment",), "prior_only": ("association", "containment")}


def model_parameters(profile, study, kind):
    model = next(row for row in study["models"] if row["kind"] == kind)
    fit = profile["fits"][kind]
    if (model["checkpoint_sha256"] != fit["checkpoint_sha256"] or model["checkpoint"] != fit["checkpoint"]
            or study["frozen_backbone_sha256"] != profile["encoder_sha256"]):
        raise ValueError(f"{kind} checkpoint or encoder lineage changed")
    frozen, _ = verified_model(model)
    common = {"state_prior": frozen["state_prior"], "reference": learned_prior_map(frozen["priors"])}
    return common, fit["mount_and_receiver_profiles"]


def view(state):
    probabilities = state["belief"]
    joint = dict(zip(STATES, state["joint"], strict=True))
    contained = sum(probabilities[slot] for slot in SLOTS)
    return {
        "decision": state["decision"], "terminal_probability": probabilities,
        "assignment_probability": {assignment: sum(value for (_, key), value in joint.items() if key == assignment)
                                   for assignment in PERMUTATIONS},
        "track_probability": {track: sum(value for (key, _), value in joint.items() if key == track)
                              for track in TRACKS},
        "containment_probability": contained, "right_probability_given_containment": probabilities["right"] / contained,
    }


def predict(common, factors, threshold):
    return choose(exact_filter(common["state_prior"], factors), threshold)


def replay_variant(row, manifest, common, fit):
    sealed = row[SELECTIVE]
    deadline = int(manifest["decision_boundary"]["monotonic_ns"])
    packets = on_time_compact(sealed["wire"]["replies"], deadline)
    factors = likelihoods_from_reports(packets, fit["association"]["compact"], common)
    threshold = fit["decision_thresholds"][SELECTIVE]
    states = {name: predict(common, [(kind, values) for kind, values in factors if kind not in excluded], threshold)
              for name, excluded in INTERVENTIONS.items()}
    full = states["full"]
    error = total_variation(full["joint"], sealed["joint"])
    if error > TOLERANCE or full["decision"] != sealed["decision"]:
        raise ValueError(f"Sealed selective posterior failed reconstruction for {row['source']}")
    prior_control = row.get("bounded_ablations", {}).get("withheld_association")
    if prior_control is not None:
        removed = states["without_association"]
        if total_variation(removed["joint"], prior_control["joint"]) > TOLERANCE or removed["decision"] != prior_control["decision"]:
            raise ValueError(f"Removing association changed the existing N=3 control for {row['source']}")
    support = row["observed_marker_support_by_witness"]
    marker = row[IDENTITY]
    if marker["serialized_bytes"] != sealed["serialized_bytes"]:
        raise ValueError(f"The marker diagnostic changed the communication charge for {row['source']}")
    marker_delta = total_variation(marker["joint"], states["without_association"]["joint"])
    return {
        "terminal_truth_for_scoring": row["truth"], "variant": row["variant"],
        "source_manifest_sha256": row["source_manifest_sha256"], "threshold": threshold,
        "marker_threshold": fit["decision_thresholds"][IDENTITY], "serialized_bytes": sealed["serialized_bytes"],
        "predictions": {name: view(state) for name, state in states.items()} | {"observed_marker_check": view(marker)},
        "retained_reports": [{key: packet[key] for key in ("factor", "sender", "probabilities")} for packet in packets],
        "marker_support_by_witness": support, "marker_to_without_association_joint_tv": marker_delta,
        "pooled_likelihoods": dict(factors), "reconstruction_joint_tv": error,
        "existing_n3_withheld_association_reproduced": prior_control is not None,
    }


def swap_factor(first, second, common, factor):
    own = first["pooled_likelihoods"]
    alternative = second["pooled_likelihoods"]
    if set(own) != {"association", "containment"} or set(own) != set(alternative):
        raise ValueError("A screened F4 factor-pair comparison requires both compact reports")
    mixed = [(key, alternative[key] if key == factor else value) for key, value in own.items()]
    return view(predict(common, mixed, first["threshold"]))


def screened_context(row, manifest):
    gate = manifest["physical_story_gate"]
    masked = all(value is None for value in row["event_labels"].values())
    if (not masked or not gate["first_insertion"] or gate["carry_contact_samples"] < 1
            or gate["contents_after_witness"] != "carrier" or gate["final_relation"] != "carrier"):
        raise ValueError(f"Screened F4 outcome or intermediate label mask changed for {row['source']}")
    return {"event_labels_are_masked": masked, "evaluator_physics_for_scoring": gate}


def decisions(groups):
    result = {}
    for arm in (*INTERVENTIONS, "observed_marker_check"):
        counts = Counter({"correct_answers": 0, "wrong_answers": 0, "inspections": 0})
        for rows in groups:
            for row in rows:
                decision = row["predictions"][arm]["decision"]
                outcome = "inspections" if decision == INSPECT else (
                    "correct_answers" if decision == row["terminal_truth_for_scoring"] else "wrong_answers")
                counts[outcome] += 1
        result[arm] = dict(counts)
    return {"independent_sources": len(groups), "variants": sum(map(len, groups)), "outcomes": result}


def summarize(replayed):
    result = {}
    for size in TEAMS:
        groups = [rows for (_, team), rows in replayed.items() if team == size]
        bins = {family: [rows for rows in groups if rows[0]["family"] == family]
                for family in ("F1", "F2", "F3", "F4", "F5", "F6")}
        if any(len(rows) != 8 for rows in bins.values()) or any(len(rows) != 2 for rows in bins["F6"]):
            raise ValueError("The audit requires the complete 48-source calibration cohort")
        for pair in bins["F6"]:
            if (pair[0]["predictions"] != pair[1]["predictions"]
                    or pair[0]["terminal_truth_for_scoring"] == pair[1]["terminal_truth_for_scoring"]):
                raise ValueError("A frozen factor replay distinguishes the paired F6 hidden alternatives")
        result[str(size)] = {name: decisions(rows) for name, rows in bins.items()}
        result[str(size)]["resolvable_container_control"] = decisions(bins["F1"] + bins["F5"])
    return result


def screened_cases(replayed, common):
    result = []
    for (group, size), rows in replayed[KINDS[0]].items():
        if rows[0]["family"] == "F4":
            cases = {kind: replayed[kind][group, size][0] for kind in KINDS}
            truth = cases[KINDS[0]]["terminal_truth_for_scoring"]
            if truth not in SLOTS or any(row["terminal_truth_for_scoring"] != truth for row in cases.values()):
                raise ValueError("Screened F4 cases require matched terminal container truth")
            for kind, row in cases.items():
                other = cases[next(candidate for candidate in KINDS if candidate != kind)]
                row["cross_head_factor_swaps"] = {factor: swap_factor(row, other, common[kind], factor)
                                                  for factor in ("containment", "association")}
            result.append({"source_group": group, "team_size": size, "by_checkpoint": cases})
    return result


def audit():
    files = score_paths("validation")
    identity, traces = authority(PROFILE, "validation")
    receipts = {kind: read_scores(path, kind, identity) for kind, path in files.items()}
    groups = {key: [evaluator_record(trace, identity["manifests"]) for trace in rows] for key, rows in traces.items()}
    checked = {kind: verify_scores(outcomes, predictions, groups) for kind, (outcomes, predictions) in receipts.items()}
    profile, study = load(PROFILE), load(STUDY)
    parameters = {kind: model_parameters(profile, study, kind) for kind in KINDS}
    manifests = {source: load(Path(source) / "manifest.json") for source in identity["manifests"]}
    replayed = {kind: replay_groups(checked[kind], manifests, *parameters[kind]) for kind in KINDS}
    return {
        "scope": "Exploratory frozen validation packet replay", "independent_source_count": len(identity["source_groups"]),
        "recorded_variants_across_teams": len(manifests), "frozen_profiles_sha256": identity["profiles_sha256"],
        "program_sha256": sha(Path(__file__)), "study_sha256": sha(STUDY),
        "validation_score_sha256": {kind: sha(path) for kind, path in files.items()},
        "validation_prediction_sha256": {kind: sha(prediction_path(path)) for kind, path in files.items()},
        "maximum_full_reconstruction_joint_tv": max(row["reconstruction_joint_tv"]
                                                    for cases in replayed.values() for rows in cases.values() for row in rows),
        "existing_n3_withheld_association_reproduced": sum(row["existing_n3_withheld_association_reproduced"]
                                                         for cases in replayed.values() for rows in cases.values() for row in rows),
        "counterfactual_threshold": "Registered per team and head, including swaps of one pooled likelihood",
        "evaluation": {kind: summarize(cases) for kind, cases in replayed.items()},
        "screened_cases": screened_cases(replayed, {kind: pair[0] for kind, pair in parameters.items()}),
    }


def replay_groups(groups, manifests, common, mounts):
    result = {}
    for key, group in groups.items():
        fit = mounts[str(key[1])]
        result[key] = []
        for row in group["outcomes"]:
            manifest = manifests[row["source"]]
            replayed = {"family": row["family"], **replay_variant(row, manifest, common, fit)}
            if row["family"] == "F4":
                replayed.update(screened_context(row, manifest))
            result[key].append(replayed)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit()
    atomic(args.output, result)
    print(f"Reproduced both frozen predictions on {result['independent_source_count']} validation groups")
    print(args.output)


if __name__ == "__main__":
    main()
