#!/usr/bin/env python3
"""Use validation-fitted profiles for the primary test without fitting on its readings."""

import argparse
from hashlib import sha256
from pathlib import Path
import time

import torch

from audit_central_history import load_backbone
from audit_evidence_replay import verified_model
from audit_memory_validation import accepted_sources
from audit_primary_ablations import evidence_controls, report as ablation_report
from audit_primary_pairs import COMPARATORS, compare
from audit_team_ledgers import factors, ledger_observations
from capture_memory_validation import CONFIG, DEFAULT_OUTPUT, allocated_slots, validate_group
from capture_primary_test import TEST_OUTPUT
from freeze_primary_profiles import KINDS, PILOT, require_profiles
from primary_test_release import bound_test_release
from score_primary_validation import FULL, FULL_WITNESSES, IDENTITY, SELECTIVE, choose, compare_caps, method_predictions, scene_observations, summarize
from smart_swarm.cli import settings
from smart_swarm.episode import LABELS, TRUTH, atomic, json_bytes, load, rows, sha
from smart_swarm.evidence_comparison import learned_prior_map
from smart_swarm.evidence_inputs import runtime_signature
from smart_swarm.memory import STATES
from smart_swarm.particle_memory import exact_filter, particle_filter
from smart_swarm.primary_inputs import primary_policy_view
from smart_swarm.research_jobs import admitted
from smart_swarm.temporal_evidence import EvidenceHead


def input_receipts(profile, rehearsal, release=None):
    root = DEFAULT_OUTPUT if rehearsal else TEST_OUTPUT
    phase = "validation" if rehearsal else "test"
    allocation = {slot["source_group"]: slot for slot in allocated_slots(phase)}
    required = set(profile["validation_cohort"]["source_groups"]) if rehearsal else set(allocation)
    receipts = list(accepted_sources(root, allocation, required, verify=False))
    if {entry["source_group"] for entry in receipts} != required:
        raise ValueError(f"{phase.capitalize()} allocation incomplete ({len(receipts)}/{len(required)})")
    if not rehearsal and (release is None or any(row["reservation"].get("primary_test_release_sha256") != release
                                                for row in receipts)):
        raise ValueError("A primary test recording lacks the original frozen-profile binding")
    manifests = {}
    for case in receipts:
        for team in case["teams"]:
            for trace in team["traces"]:
                path = Path(trace["path"])
                if not path.resolve().is_relative_to(root.resolve()) or str(path) in manifests:
                    raise ValueError("An episode path escaped its reserved root or appears more than once")
                manifests[str(path)] = sha(path / "manifest.json")
    if rehearsal and manifests != profile["validation_cohort"]["manifests"]:
        raise ValueError("Rehearsal recordings differ from the frozen validation inputs")
    return receipts, manifests, allocation


def runtime(profile, kind):
    study = load(PILOT / "story-learning.json")
    config = load(CONFIG)
    model = next(row for row in study["models"] if row["kind"] == kind)
    frozen, gains = verified_model(model)
    fit = profile["fits"][kind]
    if (fit["checkpoint_sha256"] != model["checkpoint_sha256"]
            or profile["encoder_sha256"] != study["frozen_backbone_sha256"]):
        raise ValueError("Frozen network identity differs from the original study")
    particle_path = Path(study["calibrated_particle_report"])
    if sha(particle_path) != fit["particle_reference_sha256"]:
        raise ValueError("Frozen particle reference changed")
    head = EvidenceHead().eval()
    head.load_state_dict(frozen["weights"], strict=True)
    common = {"state_prior": frozen["state_prior"], "terminal_prior": exact_filter(frozen["state_prior"], [])["belief"],
              "threshold": frozen["threshold"], "reference": learned_prior_map(frozen["priors"]), "gains": gains,
              "caps": config["communication_byte_caps"], "methods": config["primary_methods"],
              "inspect_cost": config["inspect_cost"], "wrong_cost": config["wrong_answer_cost"]}
    return {"network": (load_backbone(study), head, frozen["temperature"]), "common": common,
            "horizon": config["communication_reply_window_ms"], "mounts": fit["mount_and_receiver_profiles"],
            "particle": load(particle_path), "inspect_markers": fit["marker_support_diagnostic"]}


def apply_saved_thresholds(outcomes, thresholds, methods):
    scored = (*methods, IDENTITY) if IDENTITY in outcomes else methods
    if set(thresholds) != set(scored):
        raise ValueError("Frozen thresholds differ from the declared method list")
    for method in scored:
        outcomes[method] = choose(outcomes[method], thresholds[method])
    outcomes[FULL_WITNESSES] = choose(outcomes[FULL_WITNESSES], thresholds[FULL])
    for method, caps in outcomes["budget_diagnostics"].items():
        outcomes["budget_diagnostics"][method] = {cap: choose(row, thresholds[method]) for cap, row in caps.items()}
    return outcomes


def inference(root, size, session, ablate):
    common = session["common"]
    fit = session["mounts"][str(size)]
    obs = scene_observations(root, session["network"], session["horizon"],
                             inspect_markers=session["inspect_markers"], reverse_clip=ablate)
    rule = ledger_observations(root)
    likelihoods = factors(rule["reports"], session["particle"], fit["rule_report_likelihoods_by_extra_mount"])
    particle = particle_filter(common["state_prior"], likelihoods, fit["particle_count"], seed=session["particle"]["seed"])
    charge = {"serialized_bytes": rule["serialized_bytes"]}
    ledgers = {"deterministic": {**rule["deterministic"], **charge}, "particle": {**particle, **charge}}
    predictions = method_predictions(obs, ledgers, fit["association"], fit["receiver"], common)
    thresholds = fit["decision_thresholds"]
    outcomes = apply_saved_thresholds(predictions, thresholds, common["methods"])
    if ablate:
        outcomes["bounded_ablations"] = evidence_controls(root, obs, outcomes[SELECTIVE], fit["association"]["compact"],
                                                         common, thresholds[SELECTIVE], count=fit["particle_count"],
                                                         seed=session["particle"]["seed"])
    return outcomes


def prediction_inputs(receipts, allocation, rehearsal):
    inputs = []
    for source in receipts:
        ablation_group = rehearsal or allocation[source["source_group"]]["frozen_n3_ablation_subset"]
        for team in source["teams"]:
            size = team["actor_count"]
            inputs.append([(Path(trace["path"]), size, ablation_group and size == 3) for trace in team["traces"]])
    return inputs


def predict_sources(inputs, session, heartbeat):
    predictions = {}
    for team in inputs:
        heartbeat()
        for path, size, ablate in team:
            with primary_policy_view(path) as scoped:
                predictions[str(path)] = inference(scoped, size, session, ablate)
    return predictions


def evaluator_labels(trace):
    source = Path(trace["path"])
    reveal, = rows(source / TRUTH / "reveal.jsonl")
    observed = reveal["ground_truth"]
    if observed != trace["terminal_outcome"]:
        raise ValueError("Capture receipt disagrees with the isolated evaluator label")
    labels, = rows(source / LABELS / "settled-event.jsonl")
    if labels["terminal"] != reveal:
        raise ValueError("Event-annotation provenance disagrees with the terminal evaluator")
    return {"truth": observed, "event_labels": {key: labels[key] for key in ("settled_placement", "relocation_assignment")}}


def paired_f6(cases, methods):
    first, second = cases
    left, right = [runtime_signature(Path(item["source"])) for item in cases]
    if left != right or first["truth"] == second["truth"]:
        raise ValueError("The test pair lacks equal complete inputs and different outcomes")
    names = (*methods, FULL_WITNESSES, "budget_diagnostics")
    if IDENTITY in first:
        names = (*names, IDENTITY, "observed_marker_support_by_witness")
    if "bounded_ablations" in first:
        names = (*names, "bounded_ablations")
    if any(first[name] != second[name] for name in names):
        raise ValueError("A frozen prediction changed across identical F6 inputs")


def evaluate(receipts, predictions, manifests, methods, allocation):
    groups = []
    for source in receipts:
        name = source["source_group"]
        validate_group(allocation[name], source["seed"], source["teams"])
        for team in source["teams"]:
            cases = []
            for trace in team["traces"]:
                path = trace["path"]
                if sha(Path(path) / "manifest.json") != manifests[path]:
                    raise ValueError("An input manifest changed between prediction and evaluation")
                cases.append({"source": path, "source_manifest_sha256": manifests[path], "family": trace["family"],
                              "variant": trace["variant"], **evaluator_labels(trace), **predictions[path]})
            if cases[0]["family"] == "F6":
                paired_f6(cases, methods)
            groups.append({"group": source["source_group"], "team_size": team["actor_count"], "outcomes": cases})
    return groups


def joint_argmax(state):
    if "posterior" in state:
        best = max(state["posterior"], key=lambda item: item["p"])
        return best["location"], best["assignment"]
    if "joint" in state:
        return STATES[max(range(len(STATES)), key=state["joint"].__getitem__)]
    return None


def joint_scores(groups, method):
    labels, observations = set(), []
    for group in groups:
        known = [row for row in group["outcomes"] if all(value is not None for value in row["event_labels"].values())]
        if known:
            labels.add(group["group"])
        matches = []
        for row in known:
            state = joint_argmax(row[method])
            if state is not None:
                truth = row["event_labels"]
                matches.append(state == (truth["settled_placement"], truth["relocation_assignment"]))
        if matches:
            observations.append(matches)
    return {"labelled_source_groups": len(labels), "predicted_source_groups": len(observations),
            "group_mean_joint_argmax_accuracy": sum(sum(row) / len(row) for row in observations) / len(observations)
            if observations else None}


def metrics(grouped, common, marker_diagnostic):
    by_team = summarize(grouped, common)
    names = (*common["methods"], IDENTITY) if marker_diagnostic else common["methods"]
    joint = {size: {name: joint_scores([row for row in grouped if row["team_size"] == size], name) for name in names}
             for size in by_team}
    optional = {"marker_support_diagnostic": summarize(grouped, {**common, "methods": [IDENTITY]})} if marker_diagnostic else {}
    return {"metrics_by_team_size": by_team, "joint_event_state_accuracy_by_team": joint,
            "budget_diagnostics_by_cap": compare_caps(grouped, common),
            "paired_container_uncertainty": compare({"outcomes": grouped}, COMPARATORS), **optional}


def check_prediction_receipt(receipt, expected, sources):
    if any(receipt.get(key) != value for key, value in expected.items()):
        raise ValueError("Saved predictions refer to a different profile or input cohort")
    predictions = receipt["predictions"]
    if sha256(json_bytes(predictions)).hexdigest() != receipt["prediction_payload_sha256"]:
        raise ValueError("Saved prediction checksum differs")
    if set(predictions) != set(sources) or any(set(row) & {"truth", "event_labels"} for row in predictions.values()):
        raise ValueError("Saved predictions lack the exact inputs or contain evaluator fields")
    return predictions


def save_or_resume_predictions(path, identity, inputs, session, heartbeat):
    reused = path.is_file()
    if reused:
        receipt = load(path)
    else:
        started = time.monotonic()
        predictions = predict_sources(inputs, session, heartbeat)
        receipt = {**identity, "predictions": predictions, "prediction_elapsed_s": time.monotonic() - started,
                   "prediction_payload_sha256": sha256(json_bytes(predictions)).hexdigest()}
        check_prediction_receipt(receipt, identity, identity["manifests"])
        atomic(path, receipt)
    return check_prediction_receipt(receipt, identity, identity["manifests"]), reused, receipt["prediction_elapsed_s"]


def run(args, profile, prediction_path, grant, heartbeat):
    torch.set_num_threads(3)
    rehearsal = args.rehearse_on_validation
    release = None if rehearsal else bound_test_release(TEST_OUTPUT, args.profiles, profile)
    receipts, manifests, allocation = input_receipts(profile, rehearsal, release)
    session = runtime(profile, args.kind)
    scope = "validation replay using fixed profiles" if rehearsal else "independent primary test"
    identity = {"scope": scope, "checkpoint_kind": args.kind, "profiles_sha256": sha(args.profiles),
                "primary_test_release_sha256": release,
                "source_groups": sorted({row["source_group"] for row in receipts}), "manifests": manifests}
    inputs = prediction_inputs(receipts, allocation, rehearsal)
    predictions, reused, inference_elapsed = save_or_resume_predictions(prediction_path, identity, inputs, session, heartbeat)
    prediction_sha = sha(prediction_path)
    if not rehearsal and release != bound_test_release(TEST_OUTPUT, args.profiles, profile):
        raise ValueError("The primary test release changed before scoring")
    common = session["common"]
    grouped = evaluate(receipts, predictions, manifests, common["methods"], allocation)
    ablations = ablation_report(grouped, common, session["mounts"]["3"]["decision_thresholds"][SELECTIVE])
    if not rehearsal and len(ablations["source_groups"]) != 50:
        raise ValueError("The fixed N=3 ablation subset is incomplete")
    return {**identity, "predictions_sha256": prediction_sha, "reservation": grant, "reused_sealed_predictions": reused,
            "replay_elapsed_s_through_prediction_seal": inference_elapsed, "scorer_cpu_threads": torch.get_num_threads(),
            "fit_on_inputs": False, "training_labels_read_by_predictor": False,
            "runtime_scope": "Policy-only frames and transcript, with opaque identities and minimal manifest.",
            **metrics(grouped, common, session["inspect_markers"]),
            "bounded_n3_ablations": ablations, "outcomes": grouped}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profiles", type=Path, required=True)
    parser.add_argument("--kind", choices=KINDS, required=True)
    parser.add_argument("--rehearse-on-validation", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    profile = require_profiles(args.profiles, rehearsal=args.rehearse_on_validation)
    predictions = args.output.with_name(f"{args.output.stem}-predictions.json")
    with admitted(settings(), "smart-swarm-frozen-primary-replay") as (grant, heartbeat):
        if args.output.exists():
            raise ValueError("Choose a new output name to preserve an existing evaluation")
        report = run(args, profile, predictions, grant, heartbeat)
        atomic(args.output, report)
    print(args.output)


if __name__ == "__main__":
    main()
