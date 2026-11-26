"""Generate group-paired acquisition effects from saved independent predictions."""

import argparse
from collections import defaultdict
from pathlib import Path
from statistics import mean

from compare_primary_supervision import paired_interval
from fit_witness_transfer_selector import ARMS, KINDS, WINDOWS_MS, score_summary
from smart_swarm.episode import atomic, load, sha

PAIRS = (("empirical_expected_risk", "calibrated_fixed_selection"),
         ("learned_channel_expected_risk", "empirical_expected_risk"))


def metrics(case, kind, window, method):
    result = case["methods"][kind][window][method]
    answer, truth = result["decision"], case["truth"]
    cost = .25 if answer == "request_inspection" else float(answer != truth)
    return {"decision_cost": cost, "first_choice_accuracy": float(answer == truth),
            "total_serialized_bytes": result["serialized_bytes"]}


def paired_results(cases, left, right, head, window):
    groups = defaultdict(lambda: defaultdict(list))
    for case in cases:
        candidate, reference = (metrics(case, head, window, arm) for arm in (left, right))
        for key in candidate:
            groups[case["source_group"]][key].append(candidate[key] - reference[key])
    averages = {group: {key: mean(values) for key, values in entries.items()} for group, entries in groups.items()}
    fields = next(iter(averages.values()))
    return {"candidate": left, "reference": right, "group_differences": averages,
            "paired_intervals": {key: paired_interval([row[key] for row in averages.values()]) for key in fields}}


def report(path):
    data = load(path)
    if data["status"] != "evaluated":
        raise ValueError("An admitted independent acquisition result is required")
    cases = data["case_results"]
    groups = {row["source_group"] for row in cases}
    metrics_by_policy = score_summary(cases)
    if (len(groups) != data["independent_test_sources"] or len(cases) != data["test_variants"]
            or metrics_by_policy != data["source_weighted_metrics"]):
        raise ValueError("Independent source counts or per-case forecasts do not reproduce the saved score")
    pairs = {head: {str(window): [paired_results(cases, left, right, head, str(window)) for left, right in PAIRS]
                    for window in WINDOWS_MS} for head in KINDS}
    return {"score_sha256": sha(path), "independent_sources": len(groups), "paired_camera_variants": len(cases),
            "analysis_scope": "paired descriptive bootstrap over underlying scene groups, averaging camera variants within each group",
            "methods_by_head_and_window": metrics_by_policy, "paired_comparisons": pairs}


def markdown(data):
    lines = ["# Witness acquisition on reserved MuJoCo viewpoints", "",
             f"The test contains {data['independent_sources']} independent groups and {data['paired_camera_variants']} paired camera variants.",
             "", "| Evidence head | Reply window | Acquisition | Correct | Incorrect | Inspect | Mean bytes | Mean decision cost |",
             "| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |"]
    for head, windows in data["methods_by_head_and_window"].items():
        for window, methods in windows.items():
            for name in ARMS:
                row = methods[name]
                correct, wrong, inspect = [row["counts"].get(key, 0) for key in ("correct", "wrong", "inspection")]
                lines.append(f"| {head.replace('_', ' ')} | {window} ms | {name.replace('_', ' ')} | {correct} | {wrong} | "
                             f"{inspect} | {row['average_bytes']:.1f} | {row['cost']:.3f} |")
    lines.extend(["", "Mean costs and bytes weight each source group equally.",
                  "The paired bootstrap intervals use the underlying source group as the unit. The small test is descriptive."])
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--score", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--table", type=Path)
    args = parser.parse_args()
    data = report(args.score)
    atomic(args.output, data)
    if args.table:
        args.table.write_text(markdown(data))


if __name__ == "__main__":
    main()
