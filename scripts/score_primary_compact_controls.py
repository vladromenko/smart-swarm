"""Score the completed matched compact replay against isolated evaluator sidecars."""

import argparse
from hashlib import sha256
from pathlib import Path
from statistics import mean

from audit_primary_pairs import conservative_interval, paired_rows
from compare_primary_supervision import describe, evaluator_record, paired_interval, prediction_path, score_paths
from replay_primary_compact_controls import (
    ARMS, KINDS, PRIMARY_CAP, PROFILE, SCOPE, TEAMS, authority, public_inputs,
)
from score_primary_validation import CONFIG
from smart_swarm.episode import atomic, json_bytes, load, sha
from smart_swarm.evidence_budget_replay import wire_size


INSPECT = "request_inspection"
REFERENCE = "factor_coverage"
MEASURES = ("group_mean_nll", "group_mean_brier", "group_mean_decision_cost", "group_mean_bytes",
            "group_answer_coverage", "group_confident_wrong_rate", "group_mean_modeled_latency_ms")
METHOD_NAMES = {"factor_coverage": "Factor coverage", "central_identical_selection": "Central, identical selection",
                "broadcast_capped": "Capped broadcast", "fixed_original": "Fixed original witnesses",
                "most_recent": "Most recent", "random_eligible": "Random eligible",
                "validation_quality": "Validation quality", "all_witness_reference": "Uncapped all-witness"}


def sealed_groups(root, identity, traces):
    seal = load(root / "predictions.json")
    recipe = load(root / "release.json")
    charge = load(root / "budget.json")
    rows = seal["groups"]
    keys = [(row["source_group"], row["team_size"]) for row in rows]
    if (seal["scope"] != SCOPE or seal["source_groups"] != identity["source_groups"]
            or seal["eligible_variants"] != len(identity["manifests"]) or seal["optimizer_updates"] != 0
            or seal["evaluator_inputs_read_by_predictor"] is not False or
            sha256(json_bytes(rows)).hexdigest() != seal["groups_payload_sha256"] or
            set(keys) != set(traces) or len(keys) != len(traces)
            or "active_cap_s" in charge or not 0 < charge["used_s"] <= recipe["wall_limit_s"]):
        raise ValueError("A complete, bounded and sealed compact forecast is required before scoring")
    binding = {"release_sha256": sha(root / "release.json"),
               "primary_test_release_sha256": identity["primary_test_release_sha256"],
               "original_prediction_sha256": {k: sha(prediction_path(v)) for k, v in score_paths("test").items()}}
    for key, row in zip(keys, rows, strict=True):
        found = [{field: item[field] for field in ("path", "manifest_sha256")} for item in row["variants"]]
        if row["binding"] != binding or found != public_inputs(traces[key], identity["manifests"]):
            raise ValueError("The forecast seal lost its exact source binding")
    return rows, recipe, charge


def checked_methods(row):
    result = row["methods"]
    if set(result) != set(ARMS) or result[REFERENCE] != result["central_identical_selection"]:
        raise ValueError("The identical-input control or compact method membership changed")
    for method, forecast in result.items():
        wire = forecast["wire"]
        counts = {key: wire_size(entries) for key, entries in wire.items()}
        spent = sum(counts.values())
        latency, on_time = recorded_reply_timing(wire)
        within_limit = spent <= PRIMARY_CAP and on_time
        if (counts != forecast["bytes_by_kind"] or spent != forecast["serialized_bytes"]
                or latency != forecast["modeled_latency_ms"]
                or within_limit != forecast["within_cap"] or (method != "all_witness_reference" and not within_limit)):
            raise ValueError("A saved transport charge does not match the serialized bytes or original deadline")
    return result


def recorded_reply_timing(wire):
    requests, replies = wire["requests"], wire["replies"]
    if not requests and not replies:
        return 0., True
    request, = requests
    query = int(request["issued_ns"])
    deadline = query + 90_000_000
    if {row["sender"] for row in replies} != set(request["recipients"]) or len(replies) != len(request["recipients"]):
        raise ValueError("Reply senders differ from the saved request")
    timely = [int(row["delivered_ns"]) for row in replies
              if int(row["capture_end_ns"]) <= int(row["sent_ns"]) == query <= int(row["delivered_ns"]) <= deadline]
    return (max(timely, default=query) - query) / 1e6, len(timely) == len(replies)


def score_group(predicted, labels, kind):
    outcomes = []
    for row in predicted["variants"]:
        truth = labels[row["path"]]
        if truth["source"] != row["path"] or truth["source_manifest_sha256"] != row["manifest_sha256"]:
            raise ValueError("An evaluator row belongs to another forecast source")
        outcomes.append({**truth, **checked_methods(row["heads"][kind]),
                         "original_compact_senders_reproduced": row["heads"][kind]["original_compact_senders_reproduced"]})
    if len(outcomes) == 2:
        first, second = predicted["variants"]
        if (outcomes[0]["family"] != "F6" or outcomes[1]["family"] != "F6"
                or first["matched_private_input_sha256"] is None
                or first["matched_private_input_sha256"] != second["matched_private_input_sha256"]
                or first["heads"] != second["heads"] or outcomes[0]["truth"] == outcomes[1]["truth"]):
            raise ValueError("F6 needs identical permitted histories and divergent evaluator outcomes")
    return {"group": predicted["source_group"], "team_size": predicted["team_size"], "outcomes": outcomes}


def measures(groups, method, config):
    base = describe(groups, method, config)
    rows = [row for group in groups for row in group["outcomes"]]
    latency = mean(mean(row[method]["modeled_latency_ms"] for row in group["outcomes"]) for group in groups)
    payload = mean(mean(row[method]["bytes_by_kind"]["replies"] for row in group["outcomes"]) for group in groups)
    return {**base, "group_mean_modeled_latency_ms": latency, "group_mean_reply_bytes": payload,
            "surface_correct": sum(row["truth"] == row[method]["decision"] == "surface" for row in rows),
            "held_correct": sum(row["truth"] == row[method]["decision"] == "held" for row in rows),
            "outside_correct": sum(row["truth"] == row[method]["decision"] == "outside" for row in rows),
            "total_inspections": sum(row[method]["decision"] == INSPECT for row in rows),
            "private_f6_pairs_inspected": sum(group["outcomes"][0]["family"] == "F6"
                                              and all(row[method]["decision"] == INSPECT for row in group["outcomes"])
                                              for group in groups),
            "cases_within_original_cap": sum(row[method]["within_cap"] for row in rows)}


def comparisons(groups, config):
    per_source = {group["group"]: {arm: measures([group], arm, config) for arm in ARMS}
                  for group in groups}
    pairs = {}
    for arm in ARMS:
        if arm != REFERENCE:
            differences = {group: {field: row[arm][field] - row[REFERENCE][field] for field in MEASURES}
                           for group, row in per_source.items()}
            pairs[arm] = {"group_metrics_minus_factor_coverage": {
                              field: paired_interval([row[field] for row in differences.values()]) for field in MEASURES},
                          "container_correctness_minus_factor_coverage": conservative_interval(
                              paired_rows(groups, arm), paired_rows(groups, REFERENCE)),
                          "group_differences": differences}
    return pairs


def source_metrics(predictions, labels, config):
    unique_paths = {row["path"] for group in predictions for row in group["variants"]}
    if unique_paths != set(labels):
        raise ValueError("The score requires exactly the sealed forecast sources")
    results = {}
    for kind in KINDS:
        groups = [score_group(row, labels, kind) for row in predictions]
        results[kind] = {}
        for size in TEAMS:
            selected = [row for row in groups if row["team_size"] == size]
            results[kind][str(size)] = {"methods": {arm: measures(selected, arm, config) for arm in ARMS},
                                        "paired_to_factor_coverage": comparisons(selected, config),
                                        "reproduced_original_sender_records": sum(row["original_compact_senders_reproduced"]
                                            for group in selected for row in group["outcomes"])}
    return results


def score(root):
    identity, traces = authority(PROFILE, "test")
    predictions, recipe, charge = sealed_groups(root, identity, traces)
    labels = {row["path"]: evaluator_record(row, identity["manifests"]) for team in traces.values() for row in team}
    label_digest = sha256(json_bytes(labels) + b"\n").hexdigest()
    report = {"scope": SCOPE, "independent_sources": len(identity["source_groups"]),
            "source_and_team_groups": len(predictions), "team_condition_variants": len(identity["manifests"]),
            "score_implementation_sha256": sha(Path(__file__)), "release_sha256": sha(root / "release.json"),
            "predictions_sha256": sha(root / "predictions.json"), "evaluator_index_sha256": label_digest,
            "wall_seconds_charged": charge["used_s"],
            "optimizer_updates": 0, "communication_byte_cap": recipe["cap_bytes"], "reply_window_ms": 90,
            "peer_advertisement": "All arms pay for the same two-decimal visual confidence advertisement.",
            "unconstrained_reference": "The all-witness reference can exceed the communication cap.",
            "results": source_metrics(predictions, labels, load(CONFIG))}
    return report, labels


def markdown(data):
    header = ("| Evidence objective | Team size | Selector | First container choice | F6 pairs inspected | "
              "Wrong | Mean bytes | Mean decision cost | Within cap |")
    lines = ["# Matched compact selection on the consumed primary cohort", "",
             "The same 100 independent scenes appear at each team size. "
             "F6 alternatives stay within their source group. This analysis reopened the original test cohort.", "",
             header, "| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for kind in KINDS:
        for size in TEAMS:
            for arm in ARMS:
                row = data["results"][kind][str(size)]["methods"][arm]
                lines.append(f"| {kind.replace('_', ' ')} | {size} | {METHOD_NAMES[arm]} | "
                             f"{row['container_correct']}/{row['container_count']} | "
                             f"{row['private_f6_pairs_inspected']}/16 | {row['confident_wrong']} | "
                             f"{row['group_mean_bytes']:.1f} | {row['group_mean_decision_cost']:.3f} | "
                             f"{row['cases_within_original_cap']}/{row['variant_count']} |")
    lines.extend(["", "Each method receives the same locally computed likelihoods, metadata charges and calibrated fusion.",
                  "The identical-selection control uses the same witness packets. "
                  "The uncapped reference is outside the budgeted comparison."])
    return "\n".join(lines) + "\n"


def figure_effects(result, arm):
    pair = result["paired_to_factor_coverage"][arm]
    container = pair["container_correctness_minus_factor_coverage"]
    total = result["methods"][arm]["container_count"]
    difference = 100 * (container["wins"] - container["losses"]) / total
    bounds = [100 * value for value in container["conservative_paired_95pct"]]
    traffic = pair["group_metrics_minus_factor_coverage"]["group_mean_bytes"]
    return ((difference, bounds), (traffic["difference"], traffic["paired_group_bootstrap_95pct"]))


def figure(data, output):
    from study_figures import plt

    controls = tuple(arm for arm in ARMS if arm not in (REFERENCE, "all_witness_reference"))
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 4.3), sharey=True, layout="constrained")
    styles = (("Events only", "#235a78", "o", -.10), ("Events + reveal", "#a24a24", "s", .10))
    for kind, (label, color, marker, offset) in zip(KINDS, styles, strict=True):
        result = data["results"][kind]["8"]
        for line, arm in enumerate(controls):
            position = line + offset
            for axis, (effect, bounds) in zip(axes, figure_effects(result, arm), strict=True):
                axis.hlines(position, *bounds, color=color, linewidth=1.2)
                axis.scatter(effect, position, s=28, color=color, marker=marker,
                             label=label if line == 0 else None, zorder=3)
    axes[0].set_yticks(range(len(controls)), [METHOD_NAMES[arm] for arm in controls])
    axes[0].set_ylim(len(controls) - .55, -.45)
    axes[0].set_title("First correct container choice")
    axes[0].set_xlabel("Difference (percentage points)")
    axes[1].set_title("Group-weighted network traffic")
    axes[1].set_xlabel("Difference (bytes)")
    axes[1].legend(frameon=False, loc="best", fontsize=8)
    for axis in axes:
        axis.axvline(0, color="#454545", linewidth=.8, linestyle="--")
        axis.grid(axis="x", alpha=.17)
        axis.spines[["top", "right", "left"]].set_visible(False)
    fig.suptitle("Matched compact replay at N = 8\nComparison with factor coverage on consumed scenes", fontsize=11)
    fig.savefig(output, dpi=220, metadata={"CreationDate": None, "ModDate": None})
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--table", type=Path, required=True)
    parser.add_argument("--evaluator-index", type=Path, required=True)
    parser.add_argument("--figure", type=Path)
    args = parser.parse_args()
    targets = (args.output, args.evaluator_index, args.table, args.figure)
    if any(path is not None and path.exists() for path in targets):
        parser.error("Choose new output files to preserve earlier results")
    data, labels = score(args.root)
    atomic(args.evaluator_index, labels)
    atomic(args.output, data)
    args.table.write_text(markdown(data))
    if args.figure:
        figure(data, args.figure)
    print(args.output)


if __name__ == "__main__":
    main()
