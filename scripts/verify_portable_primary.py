#!/usr/bin/env python3
"""Recompute the two frozen primary score tables with the Python standard library."""

import argparse
from collections import defaultdict
from hashlib import sha256
import json
import math
from pathlib import Path
from statistics import mean

HEADS = ("events_only", "events_plus_reveal")
TEAMS = (3, 4, 8)
FAMILIES = ("F1", "F2", "F3", "F4", "F5", "F6")
TERMINALS = ("left", "right", "surface", "held", "outside")
INSPECT = "request_inspection"
SCOPE = ("validation", "test")
MEMBERS = {"verify_portable_primary.py", "README.md", "study.json", "forecasts.json", "outcomes.json", "reference.json"}


def load(root, name):
    return json.loads((root / name).read_bytes())


def verify_members(root):
    manifest = load(root, "SHA256.json")
    if set(manifest) != MEMBERS:
        raise ValueError("The score package has an unexpected integrity roster")
    for name, digest in manifest.items():
        if sha256((root / name).read_bytes()).hexdigest() != digest:
            raise ValueError(f"Package member changed: {name}")


def valid_prediction(state):
    belief = state["belief"]
    return (set(belief) == set(TERMINALS) and all(0. <= p <= 1. for p in belief.values())
            and math.isclose(sum(belief.values()), 1., abs_tol=1e-8)
            and state["decision"] in (*TERMINALS, INSPECT) and type(state["serialized_bytes"]) is int
            and state["serialized_bytes"] >= 0)


def verify_forecasts(forecasts, config):
    keys, groups = set(), defaultdict(list)
    methods = set(config["primary_methods"])
    for row in forecasts:
        key = tuple(row[name] for name in ("split", "group", "team", "variant"))
        if key in keys or row["split"] not in SCOPE or row["team"] not in TEAMS or row["family"] not in FAMILIES:
            raise ValueError("Invalid or repeated primary prediction identity")
        forecasts_by_head = row["heads"]
        if set(forecasts_by_head) != set(HEADS) or any(set(by_method) != methods or
                not all(valid_prediction(state) for state in by_method.values()) for by_method in forecasts_by_head.values()):
            raise ValueError("A primary posterior or method roster is invalid")
        keys.add(key)
        groups[key[:3]].append(row)
    for pair in groups.values():
        family = pair[0]["family"]
        if len(pair) != (2 if family == "F6" else 1) or any(row["family"] != family for row in pair):
            raise ValueError("A source group has the wrong number of variants")
        if len(pair) == 2 and pair[0]["heads"] != pair[1]["heads"]:
            raise ValueError("A hidden F6 pair changed its saved policy outputs")


def group_cases(forecasts, labels, config):
    fields = ("split", "group", "team", "variant")
    truth = {tuple(row[name] for name in fields): row for row in labels}
    keys = {tuple(row[name] for name in fields) for row in forecasts}
    if len(labels) != len(truth) or set(truth) != keys:
        raise ValueError("The evaluator cases differ from the forecast roster")
    groups = defaultdict(lambda: defaultdict(list))
    for row in forecasts:
        key = tuple(row[name] for name in fields)
        label = truth[key]
        if label["truth"] not in TERMINALS or label["manifest_sha256"] != row["manifest_sha256"]:
            raise ValueError("An evaluator value or source manifest reference is invalid")
        groups[row["split"], row["team"]][row["group"]].append({**row, "truth": label["truth"]})
    for (split, team), roster in groups.items():
        tracked = defaultdict(set)
        for name, cases in roster.items():
            tracked[cases[0]["family"]].add(name)
            if cases[0]["family"] == "F6" and len({row["truth"] for row in cases}) != 2:
                raise ValueError("An F6 pair lacks two distinct evaluator outcomes")
        expected = config["heldout_groups_per_team_by_family"] if split == "test" else dict.fromkeys(
            FAMILIES, config["validation_groups_per_family"])
        baseline = groups.get((split, TEAMS[0]), {})
        if ({family: len(names) for family, names in tracked.items()} != expected or roster.keys() != baseline.keys()
                or any(cases[0]["family"] != baseline[name][0]["family"] for name, cases in roster.items())):
            raise ValueError(f"Source roster changed for {split}, N={team}")
    expected_scopes = {(split, team) for split in SCOPE for team in TEAMS}
    if set(groups) != expected_scopes or groups["validation", 3].keys() & groups["test", 3].keys():
        raise ValueError("The package omits a team size or mixes validation and test sources")
    return groups


def scored_case(row, head, method, config):
    state, truth = row["heads"][head][method], row["truth"]
    decision, belief = state["decision"], state["belief"]
    correct, inspect = decision == truth, decision == INSPECT
    wrong = not (correct or inspect)
    resolvable = row["family"] in ("F1", "F2", "F3", "F5")
    return {"family": row["family"], "correct": correct, "inspect": inspect, "wrong": wrong,
            "resolvable": resolvable, "container": resolvable and truth in ("left", "right"),
            "nll": -math.log(max(belief[truth], 1e-12)),
            "brier": sum((p - (label == truth)) ** 2 for label, p in belief.items()),
            "cost": config["inspect_cost"] * inspect + config["wrong_answer_cost"] * wrong,
            "bytes": state["serialized_bytes"], "answered": not inspect}


def score_method(groups, head, method, config):
    cohorts = [[scored_case(row, head, method, config) for row in cases] for cases in groups.values()]
    items = [row for cases in cohorts for row in cases]
    families = {family: {"groups": sum(cases[0]["family"] == family for cases in cohorts),
                         "variants": sum(row["family"] == family for row in items),
                         "exact_answers": sum(row["family"] == family and row["correct"] for row in items),
                         "inspections": sum(row["family"] == family and row["inspect"] for row in items)}
                for family in FAMILIES}
    means = {"nll": "group_mean_nll", "brier": "group_mean_brier", "cost": "group_mean_decision_cost",
             "bytes": "group_mean_bytes", "answered": "group_answer_coverage", "wrong": "group_confident_wrong_rate"}
    return {**{name: mean(mean(row[key] for row in case) for case in cohorts) for key, name in means.items()},
            "families": families, "independent_groups": len(cohorts), "variant_count": len(items),
            "resolvable_count": sum(row["resolvable"] for row in items),
            "resolvable_correct": sum(row["resolvable"] and row["correct"] for row in items),
            "container_count": sum(row["container"] for row in items),
            "container_correct": sum(row["container"] and row["correct"] for row in items),
            "f6_variants": families["F6"]["variants"], "f6_inspections": families["F6"]["inspections"],
            "confident_wrong": sum(row["wrong"] for row in items), "peak_bytes": max(row["bytes"] for row in items)}


def compare_row(actual, expected):
    if actual.keys() != expected.keys():
        raise ValueError("Primary score fields differ from the published table")
    for key, value in actual.items():
        matches = math.isclose(value, expected[key], rel_tol=1e-12, abs_tol=1e-12) if isinstance(value, float) else value == expected[key]
        if not matches:
            raise ValueError(f"Recomputed primary score differs at {key}")


def verify(root):
    verify_members(root)
    config = load(root, "study.json")
    methods = config["primary_methods"]
    if len(set(methods)) != 7 or len(methods) != 7:
        raise ValueError("The study must declare seven distinct primary methods")
    forecasts = load(root, "forecasts.json")
    verify_forecasts(forecasts, config)
    groups = group_cases(forecasts, load(root, "outcomes.json"), config)
    reference = load(root, "reference.json")
    results = {}
    for split in SCOPE:
        heads = {}
        for head in HEADS:
            heads[head] = {}
            for team in TEAMS:
                roster = groups[split, team]
                matrix = {name: score_method(roster, head, name, config) for name in methods}
                published = reference[split]["metrics_by_model_and_team"][head][str(team)]
                if set(published) != set(matrix):
                    raise ValueError("Reference and recomputed method rosters differ")
                for name, row in matrix.items():
                    compare_row(row, published[name])
                heads[head][str(team)] = matrix
        results[split] = {"independent_source_groups": len(groups[split, 3]),
                          "recorded_variants_across_teams": sum(len(rows) for team in TEAMS for rows in groups[split, team].values()),
                          "metrics_by_model_and_team": heads}
        for field in ("independent_source_groups", "recorded_variants_across_teams"):
            if results[split][field] != reference[split][field]:
                raise ValueError(f"Recomputed primary cohort differs at {field}")
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = verify(args.root)
    if args.output:
        with args.output.open("x", encoding="utf-8") as stream:
            json.dump(result, stream, sort_keys=True, indent=2)
    for name, rows in result.items():
        print(f"{name}: verified {rows['independent_source_groups']} independent sources and "
              f"{rows['recorded_variants_across_teams']} team variants")


if __name__ == "__main__":
    main()
