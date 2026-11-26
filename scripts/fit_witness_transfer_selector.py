"""Freeze a bounded acquisition fit before any learned evaluation on reserved views."""

import argparse
from collections import Counter
from pathlib import Path
import time

import torch

from audit_team_evidence import decode_features
from audit_witness_transfer_cohort import audit, bound_design
from capture_memory_validation import CONFIG
from compare_compact_witness_controls import score_wire, transcript
from freeze_primary_profiles import KINDS, PILOT, require_profiles
from score_primary_test import runtime
from score_primary_validation import PRIMARY_CAP
from smart_swarm.cli import ROOT, settings
from smart_swarm.episode import LABELS, atomic, load, rows, sha
from smart_swarm.evidence_comparison import grouped_mean, nll
from smart_swarm.evidence_conditioned import fit_channels, risk_exchange
from smart_swarm.learned_channel import fit_learned
from smart_swarm.primary_inputs import primary_policy_view
from smart_swarm.research_jobs import admitted
from smart_swarm.team_evidence import select
from verify_mobile_frozen_head_reuse import clips, shared_contexts

PROFILE = PILOT / "primary-profiles-frozen.json"
ARMS = ("frozen_factor_coverage", "calibrated_fixed_selection", "empirical_expected_risk", "learned_channel_expected_risk")
WINDOWS_MS = (90, 120)
SCHEMA = "smart-swarm-frozen-witness-acquisition-v1"
EVENT_LABEL = {"containment": "settled_placement", "association": "relocation_assignment"}


def code_binding():
    paths = ("scripts/fit_witness_transfer_selector.py", "scripts/audit_witness_transfer_cohort.py",
             "src/smart_swarm/evidence_conditioned.py", "src/smart_swarm/learned_channel.py",
             "scripts/verify_mobile_frozen_head_reuse.py", "scripts/compare_compact_witness_controls.py")
    return {path: sha(ROOT / path) for path in paths}


def observation(path, contexts):
    with primary_policy_view(path) as scoped:
        window, features, markers = clips(scoped, contexts)
    query, deadline, ads = window
    updated = [{**ad, "visual_support": ad["factor"] == "containment" or markers[ad["sender"]]["supported"]}
               for ad in ads]
    probabilities = {}
    for kind, context in contexts.items():
        _, head, temperature = context["network"]
        probabilities[kind] = {sender: decode_features(feature, factor, head, temperature)
                               for sender, (factor, feature) in features.items()}
    return (query, deadline, updated), probabilities


def calibration_rows(intake, contexts, heartbeat):
    examples = {kind: [] for kind in KINDS}
    for group in intake["sources"]:
        for case in group["variants"]:
            heartbeat()
            path = Path(case["path"])
            window, posteriors = observation(path, contexts)
            labels, = rows(path / LABELS / "settled-event.jsonl")
            for kind, reports in posteriors.items():
                for ad in window[2]:
                    probabilities = reports[ad["sender"]]
                    examples[kind].append({"source_group": group["source_group"], "advertisement": ad,
                        "event_label": labels[EVENT_LABEL[ad["factor"]]], "reported_class": max(probabilities, key=probabilities.get)})
    return examples


def sessions():
    torch.set_num_threads(3)
    profile = require_profiles(PROFILE)
    return shared_contexts({kind: runtime(profile, kind) for kind in KINDS})


def freeze(root, destination, grant, heartbeat):
    intake = audit(root, "validation")
    if intake["status"] != "complete":
        raise ValueError(f"Validation capture incomplete at {intake['verified_independent_groups']} of six independent groups")
    start = time.monotonic()
    examples = calibration_rows(intake, sessions(), heartbeat)
    fitted = {}
    for kind, material in examples.items():
        heartbeat()
        empirical = fit_channels(material)
        before = time.monotonic()
        learned, report = fit_learned(material, empirical)
        fitted[kind] = {"empirical_channels": empirical, "learned_planning_channels": learned, "fit": report,
                        "eligible_likelihoods": len(material), "optimizer_wall_seconds": time.monotonic() - before}
    result = {"schema": SCHEMA, "status": "frozen", "capture_design_sha256": intake["design_sha256"],
              "profile_sha256": sha(PROFILE), "code_sha256": code_binding(), "validation_intake": intake,
              "fit_scope": "validation only", "held_out_likelihoods_read": False, "cuda_used": False,
              "communication_windows_ms": list(WINDOWS_MS), "byte_cap": PRIMARY_CAP,
              "reservation": {key: grant[key] for key in ("device", "scope", "locks")},
              "total_wall_seconds": time.monotonic() - start, "heads": fitted}
    atomic(destination, result)
    return destination


def evaluation_inputs(root, release):
    plan, digest = bound_design(root)
    if (release["schema"] != SCHEMA or release["status"] != "frozen" or release["capture_design_sha256"] != digest
            or release["profile_sha256"] != sha(PROFILE) or release["code_sha256"] != code_binding()
            or release["held_out_likelihoods_read"] is not False):
        raise ValueError("The original fitted release or capture plan changed")
    if release["validation_intake"] != audit(root, "validation"):
        raise ValueError("One or more sources used in the fit changed")
    sources = [group for group in plan["sources"] if group["split"] == "test"]
    inputs = []
    for source in sources:
        group = root / source["source_group"]
        journal = group / "receipt.json"
        if not journal.is_file():
            raise ValueError(f"The reserved test cohort is incomplete at {source['source_group']}")
        receipt = load(journal)
        if receipt["status"] != "verified" or receipt["recipe"] != source:
            raise ValueError(f"The reserved test group failed capture admission: {source['source_group']}")
        for variant in receipt["cases"]:
            episode = Path(variant["path"]).resolve()
            manifest = episode / "manifest.json"
            if not episode.is_relative_to(group.resolve()) or sha(manifest) != variant["manifest_sha256"]:
                raise ValueError("The reserved evaluation episode changed after capture")
            inputs.append({"path": str(episode), "source_group": source["source_group"],
                          "manifest_sha256": variant["manifest_sha256"]})
    return inputs


def forecasts(path, contexts, release):
    window, by_model = observation(path, contexts)
    output = {}
    for kind, reports in by_model.items():
        context = contexts[kind]
        common, fit = context["common"], context["mounts"]["4"]
        empirical = release["heads"][kind]["empirical_channels"]
        learned = release["heads"][kind]["learned_planning_channels"]

        def responder(sender, _factor):
            return reports[sender]

        output[kind] = {}
        query, _, ads = window
        for milliseconds in WINDOWS_MS:
            boundary = query + milliseconds * 1_000_000
            scenario = query, boundary, ads
            fixed = select("factor_coverage", ads, query, boundary, PRIMARY_CAP, common["gains"])
            output[kind][str(milliseconds)] = {
                "frozen_factor_coverage": score_wire(transcript(scenario, fixed, reports), scenario, fit, common),
                "calibrated_fixed_selection": risk_exchange(scenario, PRIMARY_CAP, responder, empirical, common["state_prior"],
                                                            fixed_choices=fixed),
                "empirical_expected_risk": risk_exchange(scenario, PRIMARY_CAP, responder, empirical, common["state_prior"]),
                "learned_channel_expected_risk": risk_exchange(scenario, PRIMARY_CAP, responder, learned, common["state_prior"],
                                                                evidence_tables=empirical)}
    return output


def score_summary(results):
    study = load(CONFIG)

    def measures(kind, window, arm):
        rows = [{"scene_group_id": row["source_group"], "terminal": row["truth"], **row["methods"][kind][window][arm]}
                for row in results]
        counts = Counter("correct" if row["decision"] == row["terminal"] else (
            "inspection" if row["decision"] == "request_inspection" else "wrong") for row in rows)
        cost = lambda row: study["inspect_cost"] if row["decision"] == "request_inspection" else (
            study["wrong_answer_cost"] * (row["decision"] != row["terminal"]))
        return {"counts": dict(counts), "cost": grouped_mean(rows, cost), "nll": nll(rows),
                "brier": grouped_mean(rows, lambda row: sum(
                    (value - (name == row["terminal"])) ** 2 for name, value in row["belief"].items())),
                "average_bytes": grouped_mean(rows, lambda row: row["serialized_bytes"]),
                "mean_modeled_latency_ms": grouped_mean(rows, lambda row: row.get("serial_latency_ms", row.get("modeled_latency_ms"))),
                "average_replies": grouped_mean(rows, lambda row: len(row["wire"]["replies"])),
                "second_request_histories": sum(len(row["wire"]["requests"]) > 1 for row in rows)}

    return {kind: {str(time): {arm: measures(kind, str(time), arm) for arm in ARMS} for time in WINDOWS_MS} for kind in KINDS}


def evaluate(root, policy, destination, heartbeat):
    release = load(policy)
    sources = evaluation_inputs(root, release)
    directory = destination.parent
    separate = directory / (destination.stem + "-sealed-predictions.json")
    binding = {"policy_sha256": sha(policy), "test_input_manifests": sources}
    if separate.is_file():
        saved = load(separate)
        if saved["binding"] != binding:
            raise ValueError("The independent predictions belong to a different fitted release or set of views")
    else:
        contexts = sessions()
        predictions = []
        for row in sources:
            heartbeat()
            predictions.append({**row, "methods": forecasts(Path(row["path"]), contexts, release)})
        saved = {"binding": binding, "predictions": predictions}
        atomic(separate, saved)
    verified = audit(root, "test")
    labels = {case["path"]: group["terminal"] for group in verified["sources"] for case in group["variants"]}
    if set(labels) != {row["path"] for row in saved["predictions"]} or len(labels) != len(sources):
        raise ValueError("Isolated observed outcomes do not cover the sealed predictions")
    cases = [{**row, "truth": labels[row["path"]]} for row in saved["predictions"]]
    report = {"status": "evaluated", "scope": "reserved MuJoCo viewpoints after validation-only fitting",
              "policy_sha256": sha(policy), "predictions_sha256": sha(separate), "test_intake": verified,
              "matched_methods": list(ARMS), "case_results": cases, "source_weighted_metrics": score_summary(cases)}
    atomic(destination, report)
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frozen-policy", type=Path, help="Use the frozen release for the independent evaluation")
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Use a new receipt path to preserve the earlier calibration or evaluation")
    with admitted(settings(), "smart-swarm-witness-acquisition", archive_only=True) as (grant, heartbeat):
        location = (evaluate(args.root, args.frozen_policy, args.output, heartbeat) if args.frozen_policy
                    else freeze(args.root, args.output, grant, heartbeat))
    print(location)


if __name__ == "__main__":
    main()
