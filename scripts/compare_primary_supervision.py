#!/usr/bin/env python3
"""Compare the two frozen supervision objectives on identical sealed source groups."""

import argparse
from pathlib import Path
from statistics import mean

import numpy as np

from audit_primary_pairs import conservative_interval, paired_rows
from capture_primary_test import TEST_OUTPUT, bound_test_release
from freeze_primary_profiles import KINDS, PILOT, require_profiles
from score_primary_test import check_prediction_receipt, evaluator_labels, input_receipts
from score_primary_validation import CONFIG, IDENTITY, SELECTIVE, by_family, team_metrics
from smart_swarm.episode import LABELS, TRUTH, atomic, load, sha

TEAMS = (3, 4, 8)
METHODS = (SELECTIVE, IDENTITY)
LOWER_IS_BETTER = ("group_mean_nll", "group_mean_brier", "group_mean_decision_cost")
METRIC_NAMES = (*LOWER_IS_BETTER, "group_mean_bytes", "group_answer_coverage", "group_confident_wrong_rate")
FIELDS = ("source", "source_manifest_sha256", "family", "variant", "truth", "event_labels")
BOOTSTRAP_DRAWS = 4000


def score_paths(scope):
    stem = "primary-test" if scope == "test" else "primary-frozen-rehearsal-forty-eight"
    files = {kind: PILOT / f"{stem}-{kind}.json" for kind in KINDS}
    missing = [path.name for path in files.values()
               if not path.is_file() or not prediction_path(path).is_file()]
    if missing:
        raise ValueError(f"Both complete scores and prediction seals are required. Pending {', '.join(missing)}")
    return files


def prediction_path(path):
    return path.with_name(path.stem + "-predictions.json")


def evaluator_record(trace, manifests):
    source = trace["path"]
    root = Path(source)
    sealed = load(root / "manifest.json")
    for reference in (f"{TRUTH}/reveal.jsonl", f"{LABELS}/settled-event.jsonl"):
        if sha(root / reference) != sealed["sidecar_sha256"][reference]:
            raise ValueError("An evaluator or label sidecar changed after the capture seal")
    return {"source": source, "source_manifest_sha256": manifests[source],
            "family": trace["family"], "variant": trace["variant"], **evaluator_labels(trace)}


def authority(profile_path, scope):
    validation = scope == "validation"
    profile = require_profiles(profile_path)
    release = None if validation else bound_test_release(TEST_OUTPUT, profile_path, profile)
    captures, manifests, _ = input_receipts(profile, validation, release)
    groups = {}
    for trial in captures:
        for team in trial["teams"]:
            key = trial["source_group"], team["actor_count"]
            if key in groups:
                raise ValueError("The scorer authority repeats a source and team")
            groups[key] = team["traces"]
    clusters = sorted({group for group, _ in groups})
    if set(groups) != {(group, size) for group in clusters for size in TEAMS}:
        raise ValueError("Each independent source must have all the declared team sizes")
    identity = {"scope": "validation replay using fixed profiles" if validation else "independent primary test",
                "profiles_sha256": sha(profile_path), "primary_test_release_sha256": release,
                "source_groups": clusters, "manifests": manifests}
    return identity, groups


def read_scores(path, kind, identity):
    key = {**identity, "checkpoint_kind": kind}
    report = load(path)
    if (any(report.get(name) != value for name, value in key.items()) or report["fit_on_inputs"] is not False
            or report["training_labels_read_by_predictor"] is not False):
        raise ValueError(f"The {kind} report differs from the original frozen cohort or policy scope")
    sealed = prediction_path(path)
    if report["predictions_sha256"] != sha(sealed):
        raise ValueError(f"The {kind} scored prediction seal changed")
    predictions = check_prediction_receipt(load(sealed), key, identity["manifests"])
    return report["outcomes"], predictions


def verify_scores(outcomes, predictions, groups):
    indexed = {}
    for group in outcomes:
        identifier = group["group"], group["team_size"]
        expected = groups[identifier]
        payload = group["outcomes"]
        if identifier in indexed or [{name: row[name] for name in FIELDS} for row in payload] != expected:
            raise ValueError("A score changes source identities, evaluator labels or variant order")
        for row in payload:
            if {k: v for k, v in row.items() if k not in FIELDS} != predictions[row["source"]]:
                raise ValueError("A scored result disagrees with the pre-evaluator prediction")
        indexed[identifier] = group
    if set(indexed) != set(groups):
        raise ValueError("A scored result omitted an allocated source or team")
    return indexed


def describe(groups, method, config):
    metrics = team_metrics(groups, method, inspect_cost=config["inspect_cost"], wrong_cost=config["wrong_answer_cost"])
    coverage, errors = [], []
    for group in groups:
        rows = group["outcomes"]
        answers = [row[method]["decision"] for row in rows]
        coverage.append(mean(decision != "request_inspection" for decision in answers))
        errors.append(mean(choice not in (row["truth"], "request_inspection")
                           for choice, row in zip(answers, rows, strict=True)))
    return {**metrics, "families": by_family(groups, method), "group_answer_coverage": mean(coverage),
            "group_confident_wrong_rate": mean(errors)}


def paired_interval(values):
    array = np.asarray(values, dtype=float)
    if not array.size or not np.isfinite(array).all():
        raise ValueError("A paired group-level estimate lacks finite observations")
    seed = np.random.default_rng(20261126)
    draws = seed.choice(array, (BOOTSTRAP_DRAWS, len(array)), replace=True).mean(axis=1)
    limits = np.quantile(draws, [.025, .975]).tolist()
    return {"difference": float(array.mean()), "paired_group_bootstrap_95pct": limits,
            "bootstrap_degenerate": limits[0] == limits[1]}


def container_effect(sets, method, config):
    event, reveal = [paired_rows(sets[kind], method) for kind in KINDS]
    if [(row["world_episode_id"], row["terminal"]) for row in event] != [
            (row["world_episode_id"], row["terminal"]) for row in reveal]:
        raise ValueError("The first-choice endpoint uses different eligible decisions")
    checks = conservative_interval(reveal, event)
    n = len(event)
    return {**checks, "independent_groups": n, "difference": (checks["wins"] - checks["losses"]) / n,
            "practical_absolute_difference": config["practical_difference"],
            "target_interval_width": config["paired_95pct_interval_target_width"],
            "precision_target_met": checks["interval_width"] <= config["paired_95pct_interval_target_width"]}


def team_comparison(sets, method, config):
    summaries = {kind: describe(groups, method, config) for kind, groups in sets.items()}
    clusters = {kind: [describe([group], method, config) for group in groups] for kind, groups in sets.items()}
    event, reveal = [clusters[kind] for kind in KINDS]
    paired = {key: paired_interval([b[key] - a[key] for a, b in zip(event, reveal, strict=True)])
              for key in METRIC_NAMES}
    f6 = [group for group in sets[KINDS[0]] if group["outcomes"][0]["family"] == "F6"]
    if any(len(group["outcomes"]) != 2 for group in f6):
        raise ValueError("F6 must contribute two hidden alternatives per independent source")
    return {"by_model": summaries, "paired_group_differences": paired, "paired_container_effect": container_effect(sets, method, config),
            "f6_independent_groups": len(f6)}


def compare(profile_path, scope):
    files = score_paths(scope)
    identity, traces = authority(profile_path, scope)
    receipts = {kind: read_scores(path, kind, identity) for kind, path in files.items()}
    groups = {key: [evaluator_record(trace, identity["manifests"]) for trace in rows] for key, rows in traces.items()}
    results = {kind: verify_scores(outcomes, predictions, groups) for kind, (outcomes, predictions) in receipts.items()}
    config = load(CONFIG)
    methods = {}
    for method in METHODS:
        methods[method] = {}
        for size in TEAMS:
            sets = {kind: [results[kind][(group, size)] for group in identity["source_groups"]] for kind in KINDS}
            methods[method][str(size)] = team_comparison(sets, method, config)
    return {"scope": identity["scope"], "profiles_sha256": identity["profiles_sha256"],
            "primary_test_release_sha256": identity["primary_test_release_sha256"],
            "score_sha256": {kind: sha(path) for kind, path in files.items()},
            "sources_per_team": len(identity["source_groups"]), "independent_sources_across_teams": len(identity["source_groups"]),
            "recorded_variants_across_teams": len(identity["manifests"]), "difference_direction": "events plus reveal minus events only",
            "analysis_unit": "same independent source within each N, averaging hidden F6 alternatives inside the source",
            "interpretation": "Descriptive paired validation replay" if scope == "validation" else "Independent paired test",
            "paired_bootstrap_draws": BOOTSTRAP_DRAWS, "lower_is_better": list(LOWER_IS_BETTER), "methods": methods}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profiles", type=Path, default=PILOT / "primary-profiles-frozen.json")
    parser.add_argument("--scope", choices=("validation", "test"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--figure", type=Path)
    args = parser.parse_args()
    try:
        result = compare(args.profiles, args.scope)
    except (OSError, ValueError, KeyError) as error:
        parser.exit(2, f"Paired supervision report withheld. {error}\n")
    if args.output.is_file() and load(args.output) != result:
        raise ValueError("Choose a new report path to preserve the existing result")
    if not args.output.is_file():
        atomic(args.output, result)
    if args.figure:
        from study_figures import paired_supervision

        paired_supervision(args.output, args.figure)
    print(args.output)


if __name__ == "__main__":
    main()
