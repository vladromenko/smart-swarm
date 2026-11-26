#!/usr/bin/env python3
"""Score the original transmitted witness probabilities separately from terminal decisions."""

import argparse
from collections import Counter, defaultdict
from hashlib import sha256
import json
from math import log
from pathlib import Path
from statistics import mean
import tarfile

from archive_primary_release import INDEX, KINDS, read_member
from archive_primary_results import verify
from audit_primary_delivery_delay import RESULTS, RESULTS_SHA256, forecast_inputs
from compare_primary_supervision import paired_interval
from score_primary_validation import SELECTIVE
from smart_swarm.contract import read_ns
from smart_swarm.episode import atomic, json_bytes
from smart_swarm.evidence_budget_replay import wire_size
from smart_swarm.evidence_comparison import LABELS, grouped_mean
from smart_swarm.particle_memory import FACTORS
from smart_swarm.physical_witness_score import SENDERS, matches, normalized_probabilities, terminal_pattern, wire_index

WINDOW_NS = 90_000_000
MEASURED = "labeled_visual_reply"
SIZES = (3, 4, 8)


def open_scores(archive, kind, binding):
    forecast = read_member(archive, f"scores/primary-test-{kind}-predictions.json")
    scored = json.loads(read_member(archive, f"scores/primary-test-{kind}.json"))
    manifests = {str(Path(path).parent): row["sha256"] for path, row in binding["original_recording_manifests"].items()}
    if (scored["predictions_sha256"] != sha256(forecast).hexdigest()
            or scored["checkpoint_kind"] != kind or scored["scope"] != "independent primary test"
            or scored["profiles_sha256"] != binding["profiles_sha256"]
            or scored["primary_test_release_sha256"] != binding["primary_test_release_sha256"]
            or scored["fit_on_inputs"] or scored["training_labels_read_by_predictor"] or scored["manifests"] != manifests):
        raise ValueError("Evaluator labels and sealed predictions refer to different test inputs")
    if any(row["source_manifest_sha256"] != manifests.get(row["source"])
           for group in scored["outcomes"] for row in group["outcomes"]):
        raise ValueError("An evaluator label does not bind its original recording manifest")
    return scored["outcomes"]


def witness_factor(factor, label, index, query, deadline):
    if label is not None and label not in FACTORS[factor]:
        raise ValueError("An archived event label is outside its original class vocabulary")
    ads, requested, replies = index
    result = {"label": label, "sender": SENDERS[factor]}
    if factor not in ads:
        return {**result, "status": "no_advertisement"}
    if SENDERS[factor] not in requested:
        return {**result, "status": "not_requested"}
    if factor not in replies:
        return {**result, "status": "no_reply"}
    ad, reply = ads[factor], replies[factor]
    capture, sent, delivered = (read_ns(reply[key]) for key in ("capture_end_ns", "sent_ns", "delivered_ns"))
    count = ad["frame_count"]
    if (not capture <= sent == query <= delivered or capture != read_ns(ad["capture_end_ns"])
            or delivered - sent != ad["delay_ns"] or len(reply["frame_ids"]) != count
            or len(set(reply["frame_ids"])) != count):
        raise ValueError("A saved witness reply disagrees with its advertised frames or modeled times")
    result.update(delay_ns=delivered - query, frame_count=count)
    if delivered > deadline:
        return {**result, "status": "after_deadline"}
    probabilities = normalized_probabilities(reply, factor, mapping=None)
    return {**result, "status": MEASURED if label is not None else "event_label_masked", "probabilities": probabilities}


def score_case(group, case, prediction, projection, kind):
    if ((group["group"], group["team_size"]) != (projection["scene_group_id"], projection["scaling_team_size"])
            or case[SELECTIVE] != prediction or set(case["event_labels"]) != set(LABELS.values())):
        raise ValueError("An evaluator row differs from the original scoped forecast or source")
    query = read_ns(projection["query_boundary"]["monotonic_ns"])
    deadline = read_ns(projection["decision_boundary"]["monotonic_ns"])
    if deadline - query != WINDOW_NS:
        raise ValueError("The archived decision deadline changed")
    wire = prediction["wire"]
    if sum(wire_size(wire[stage]) for stage in ("requests", "advertisements", "replies")) != prediction["serialized_bytes"]:
        raise ValueError("The transmitted replies disagree with archived communication charges")
    # Larger teams charge the unused advertisements as well as replies from the original witnesses.
    selected_ads = [row for row in wire["advertisements"] if row["sender"] == SENDERS.get(row["factor"])]
    index = wire_index({**wire, "advertisements": selected_ads}, query)
    factors = {factor: witness_factor(factor, case["event_labels"][event], index, query, deadline)
               for factor, event in LABELS.items()}
    diagnostic = {"terminal": case["truth"], "forecasts": {kind: prediction}, "witness_diagnostics": {"heads": {kind: factors}}}
    return {"scene_group_id": group["group"], "team_size": group["team_size"], "family": case["family"],
            "variant": case["variant"], "source": Path(case["source"]).name, "witnesses": factors,
            "terminal": case["truth"], "decision": prediction["decision"], "terminal_pattern": terminal_pattern(diagnostic, kind)}


def score_groups(groups, predictions, projections, kind):
    used, cases = set(), []
    for group in groups:
        outcomes = group["outcomes"]
        is_f6 = all(item["family"] == "F6" for item in outcomes)
        if len(outcomes) != (2 if is_f6 else 1):
            raise ValueError("An archived source has the wrong number of coupled variants")
        scored = []
        for item in outcomes:
            source = item["source"]
            if source in used:
                raise ValueError("A frozen forecast appeared twice in the evaluator table")
            used.add(source)
            scored.append(score_case(group, item, predictions[source], projections[source], kind))
        if is_f6:
            left, right = scored
            if (left["terminal"] == right["terminal"] or left["witnesses"] != right["witnesses"]
                    or left["witnesses"]["containment"]["label"] is not None
                    or left["witnesses"]["containment"]["status"] != "no_advertisement"):
                raise ValueError("An F6 pair does not preserve its withheld placement and matched replies")
        cases.extend(scored)
    if used != predictions.keys():
        raise ValueError("The archive omitted an original forecast in the evaluator join")
    return cases


def nll(row):
    return -log(max(row["probabilities"][row["label"]], 1e-12))


def brier(row):
    return sum((value - (name == row["label"])) ** 2 for name, value in row["probabilities"].items())


def factor_rows(cases, factor):
    return [{**case["witnesses"][factor], "scene_group_id": case["scene_group_id"], "source": case["source"]} for case in cases]


def factor_metrics(cases, factor):
    rows = factor_rows(cases, factor)
    measured = [row for row in rows if row["status"] == MEASURED]
    return {"statuses": dict(Counter(row["status"] for row in rows)),
            "scored_source_groups": len({row["scene_group_id"] for row in measured}), "scored_variants": len(measured),
            "correct_top_class_variants": sum(matches(row) for row in measured),
            "group_mean_accuracy": grouped_mean(measured, matches) if measured else None,
            "group_mean_nll": grouped_mean(measured, nll) if measured else None,
            "group_mean_brier": grouped_mean(measured, brier) if measured else None}


def paired_factors(first, second, factor):
    a, b = [sorted(factor_rows(rows, factor), key=lambda row: row["source"]) for rows in (first, second)]
    differences = {name: defaultdict(list) for name in ("accuracy", "nll", "brier")}
    for base, reveal in zip(a, b, strict=True):
        if any(base[key] != reveal[key] for key in ("source", "scene_group_id", "label", "status")):
            raise ValueError("The frozen supervision objectives do not share the same witness score support")
        if base["status"] == MEASURED:
            for name, metric in (("accuracy", matches), ("nll", nll), ("brier", brier)):
                differences[name][base["scene_group_id"]].append(metric(reveal) - metric(base))
    return {name: {"source_groups": len(groups), **paired_interval([mean(rows) for rows in groups.values()])}
            for name, groups in differences.items() if groups}


def analyses(cases):
    result = {}
    for size in SIZES:
        scope = {kind: [row for row in cases[kind] if row["team_size"] == size] for kind in KINDS}
        views = {}
        for kind, rows in scope.items():
            families = sorted({row["family"] for row in rows})
            views[kind] = {"witnesses": {factor: factor_metrics(rows, factor) for factor in LABELS},
                           "terminal_patterns": dict(Counter(row["terminal_pattern"] for row in rows)),
                           "by_family": {family: {factor: factor_metrics([row for row in rows if row["family"] == family], factor)
                                                  for factor in LABELS} for family in families}}
        effects = {factor: paired_factors(*(scope[kind] for kind in KINDS), factor) for factor in LABELS}
        result[str(size)] = {"by_head": views, "paired_reveal_minus_events": effects}
    return result


def report(path):
    original = verify(path)
    if original["archive_sha256"] != RESULTS_SHA256 or original["scope"] != "test" or original["independent_source_groups"] != 100:
        raise ValueError("Witness scoring requires the original sealed 100-source result archive")
    with tarfile.open(path, "r:gz") as archive:
        binding = json.loads(read_member(archive, INDEX))
    _, projections, predictions = forecast_inputs(path, binding)
    with tarfile.open(path, "r:gz") as archive:
        cases = {kind: score_groups(open_scores(archive, kind, binding), predictions[kind], projections, kind) for kind in KINDS}
    return {"schema": "smart-swarm-frozen-primary-witness-scores-v1",
            "scope": "Exploratory scoring of original selective replies on the consumed simulation test cohort",
            "source_archive_sha256": original["archive_sha256"], "independent_source_groups": 100,
            "original_camera_views_and_wire": True, "modeled_reply_window_ns": WINDOW_NS,
            "by_team_size": analyses(cases), "cases": cases}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", default=RESULTS, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = report(args.archive)
    if args.output.is_file() and args.output.read_bytes() != json_bytes(result) + b"\n":
        parser.error("Choose a new report path to preserve the existing result")
    if not args.output.exists():
        atomic(args.output, result)
    print(args.output)


if __name__ == "__main__":
    main()
