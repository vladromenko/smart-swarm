#!/usr/bin/env python3
"""Join the full F6 input audit to both sealed pre-inspection score receipts."""

import argparse
from collections import Counter
from hashlib import sha256
from pathlib import Path

from audit_primary_scope import ALLOWED_MANIFEST, paired
from compare_primary_supervision import (
    TEAMS, authority, evaluator_record, prediction_path, read_scores, score_paths, verify_scores,
)
from freeze_primary_profiles import KINDS, PILOT
from score_primary_validation import CONFIG
from smart_swarm.contract import json_bytes
from smart_swarm.episode import atomic, load, sha
from smart_swarm.evidence_inputs import runtime_signature

RESOLVABLE = ("F1", "F2", "F3", "F5")
INSPECT = "request_inspection"


def paired_payloads(traces, predictions, methods):
    sources = [row["path"] for row in traces]
    result = {}
    for kind in KINDS:
        first, second = [predictions[kind][source] for source in sources]
        if first != second or not set(methods) <= first.keys():
            raise ValueError(f"{kind} pre-inspection payload differs across equal F6 inputs")
        result[kind] = {"payload_sha256": sha256(json_bytes(first)).hexdigest(),
                        "decision_by_method": {key: first[key]["decision"] for key in methods}}
    return result


def verify_pair(traces, size, predictions, methods):
    inputs = paired({"traces": traces, "actor_count": size})
    outcomes = set(inputs["outcomes"])
    if "surface" not in outcomes or not outcomes <= {"left", "right", "surface"}:
        raise ValueError("The F6 evaluator alternatives differ from the contained versus surface protocol")
    checks = paired_payloads(traces, predictions, methods)
    return {**inputs, "pre_inspection_payloads": checks}


def resolvable_controls(indexed, size, methods):
    groups = [row for (_, team), row in indexed.items() if team == size and row["outcomes"][0]["family"] in RESOLVABLE]
    if not groups or any(len(row["outcomes"]) != 1 for row in groups):
        raise ValueError("The resolving control requires one observed outcome per independent source")
    histories = [row["outcomes"][0] for row in groups]
    return {"independent_groups": len(histories), "always_inspect_correct": 0,
            "correct_by_method": {name: sum(row[name]["decision"] == row["truth"] for row in histories) for name in methods}}


def team_checks(groups, scores, size, methods):
    results = {}
    for kind in KINDS:
        choices = [item["pre_inspection_payloads"][kind]["decision_by_method"] for item in groups]
        counts = {name: dict(Counter(row[name] for row in choices)) for name in methods}
        results[kind] = {"f6_pair_decisions_by_method": counts,
                         "f6_inspected_variants_by_method": {name: 2 * counts[name].get(INSPECT, 0) for name in methods},
                         "resolvable_controls": resolvable_controls(scores[kind], size, methods)}
    return results


def coupled_particle_seed(profile_path):
    profile = load(profile_path)
    particle = Path(load(PILOT / "story-learning.json")["calibrated_particle_report"])
    if any(fit["particle_reference_sha256"] != sha(particle) for fit in profile["fits"].values()):
        raise ValueError("The particle seed is outside the frozen prediction profile")
    return load(particle)["seed"]


def collect(profile_path, scope):
    paths = score_paths(scope)
    identity, traces = authority(profile_path, scope)
    receipts = {kind: read_scores(path, kind, identity) for kind, path in paths.items()}
    truth = {key: [evaluator_record(trace, identity["manifests"]) for trace in rows] for key, rows in traces.items()}
    scores = {kind: verify_scores(outcomes, predictions, truth) for kind, (outcomes, predictions) in receipts.items()}
    predictions = {kind: pair[1] for kind, pair in receipts.items()}
    config = load(CONFIG)
    methods = config["primary_methods"]
    expected = config["validation_groups_per_family"] if scope == "validation" else config["heldout_groups_per_team_by_family"]["F6"]
    matches = {key: rows for key, rows in traces.items() if rows[0]["family"] == "F6"}
    names = {name for name, _ in matches}
    if len(names) != expected or set(matches) != {(name, size) for name in names for size in TEAMS}:
        raise ValueError("Allocated F6 capture coverage differs from the full registered cohort")
    groups = {name: [verify_pair(matches[(name, size)], size, predictions, methods) for size in TEAMS] for name in sorted(names)}
    summary = {str(size): team_checks([groups[name][ix] for name in groups], scores, size, methods)
               for ix, size in enumerate(TEAMS)}
    fields = sorted(runtime_signature(Path(next(iter(matches.values()))[0]["path"])))
    return {"schema": "smart-swarm-frozen-f6-audit-v1", "scope": identity["scope"],
            "profiles_sha256": identity["profiles_sha256"], "primary_test_release_sha256": identity["primary_test_release_sha256"],
            "prediction_seals_sha256": {kind: sha(prediction_path(path)) for kind, path in paths.items()},
            "independent_source_groups": len(groups), "team_pairs": len(matches), "qualified_policy_methods": methods,
            "allowed_manifest_fields": sorted(ALLOWED_MANIFEST), "permitted_input_signature_fields": fields,
            "particle_seed": coupled_particle_seed(profile_path), "evaluator_and_training_sidecars_substituted": True,
            "full_saved_prediction_payload_identical_per_pair": True, "groups": groups, "team_checks": summary}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scope", choices=("validation", "test"), required=True)
    parser.add_argument("--profiles", type=Path, default=PILOT / "primary-profiles-frozen.json")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    try:
        receipt = collect(args.profiles, args.scope)
        if args.out.exists() and load(args.out) != receipt:
            raise ValueError("Choose a new immutable audit output")
        if not args.out.exists():
            atomic(args.out, receipt)
    except (OSError, ValueError, KeyError) as error:
        parser.exit(2, f"Frozen F6 audit withheld. {error}\n")
    print(args.out)


if __name__ == "__main__":
    main()
