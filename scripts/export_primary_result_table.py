#!/usr/bin/env python3
"""Export the seven registered methods from the two sealed primary scores."""

import argparse
from collections import Counter
import math
from pathlib import Path
from tempfile import NamedTemporaryFile

from audit_primary_pairs import paired_rows
from compare_primary_supervision import (
    TEAMS, authority, describe, evaluator_record, read_scores, score_paths, verify_scores,
)
from freeze_primary_profiles import KINDS, PILOT
from score_primary_validation import CONFIG, on_time_compact
from smart_swarm.episode import atomic, json_bytes, load, sha

METHOD_NAMES = {
    "receiver_current_rgb": "Receiver current RGB",
    "receiver_private_temporal": "Receiver private history",
    "deterministic_broadcast_ledger": "Deterministic broadcast",
    "calibrated_broadcast_joint_particle_filter": "Calibrated particle broadcast",
    "central_full_rgb_joint_filter": "Central full RGB",
    "central_compact_budget_matched": "Central compact",
    "learned_selective_joint_memory": "Learned selective",
}
HEAD_NAMES = {"events_only": "Event-only", "events_plus_reveal": "Event + reveal"}
COMPACT_METHODS = ("central_compact_budget_matched", "learned_selective_joint_memory")
FULL_RGB_CONTROLS = ("central_full_rgb_joint_filter", "central_full_witnesses_without_receiver_cue")


def verify_metrics(recomputed, saved):
    for key, value in saved.items():
        actual = recomputed[key]
        same = math.isclose(actual, value, rel_tol=1e-12, abs_tol=1e-12) if isinstance(value, float) else actual == value
        if not same:
            raise ValueError(f"Sealed outcomes disagree with a reported study metric at {key}")


def association_replies(rows, method, reply_ns):
    results = Counter()
    for row in rows:
        wire = row[method]["wire"]
        issued = min(int(request["issued_ns"]) for request in wire["requests"]) if wire["requests"] else 0
        on_time = on_time_compact(wire["replies"], issued + reply_ns)
        association = [reply["sender"] for reply in on_time if reply["factor"] == "association"]
        results.update(association or ["no_reply"])
    return dict(results)


def controls(groups, config):
    eligible = {row["world_episode_id"] for row in paired_rows(groups, COMPACT_METHODS[0])}
    container = [row for group in groups for row in group["outcomes"] if row["source"] in eligible]
    f6 = [row for group in groups for row in group["outcomes"] if row["family"] == "F6"]
    reply_ns = config["communication_reply_window_ms"] * 1_000_000
    counts = {method: dict(Counter(row[method]["decision"] for row in f6)) for method in FULL_RGB_CONTROLS}
    return {"f6_full_rgb_and_receiver_ablation_choices": counts, "resolvable_container_groups": len(container),
            "on_time_association_replies_on_container_controls": {
                method: association_replies(container, method, reply_ns) for method in COMPACT_METHODS}}


def one_head(path, receipt, identity, truth, config):
    scores, predictions = receipt
    verified = verify_scores(scores, predictions, truth)
    saved = load(path)["metrics_by_team_size"]
    methods = config["primary_methods"]
    result, checks = {}, {}
    for size in TEAMS:
        published = saved[str(size)]
        if set(published) != set(methods):
            raise ValueError("A score differs from the seven registered comparison methods")
        groups = [verified[(name, size)] for name in identity["source_groups"]]
        matrix = {name: describe(groups, name, config) for name in methods}
        for name, row in matrix.items():
            verify_metrics(row, published[name])
            if row["f6_variants"] != 2 * row["families"]["F6"]["groups"]:
                raise ValueError("The table must retain both F6 alternatives inside each independent group")
        result[str(size)] = matrix
        checks[str(size)] = controls(groups, config)
    return result, checks


def collect(profile_path, scope):
    paths = score_paths(scope)
    identity, traces = authority(profile_path, scope)
    receipts = {kind: read_scores(path, kind, identity) for kind, path in paths.items()}
    truth = {key: [evaluator_record(trace, identity["manifests"]) for trace in rows] for key, rows in traces.items()}
    config = load(CONFIG)
    methods = config["primary_methods"]
    if set(methods) != set(METHOD_NAMES) or len(methods) != len(METHOD_NAMES):
        raise ValueError("Update the table method names against the registered study")
    runs = {kind: one_head(paths[kind], receipts[kind], identity, truth, config) for kind in KINDS}
    checksums = {kind: sha(path) for kind, path in paths.items()}
    return {"schema": "smart-swarm-primary-method-table-v1", "scope": identity["scope"],
            "independent_source_groups": len(identity["source_groups"]), "teams": list(TEAMS),
            "recorded_variants_across_teams": len(identity["manifests"]), "profiles_sha256": identity["profiles_sha256"],
            "primary_test_release_sha256": identity["primary_test_release_sha256"], "scored_receipts_sha256": checksums,
            "registered_method_order": methods, "metrics_by_model_and_team": {kind: row[0] for kind, row in runs.items()},
            "recorded_control_checks_by_model_and_team": {kind: row[1] for kind, row in runs.items()}}


def table_lines(report, size):
    title = f"### N = {size}"
    header = "| Objective | Method | Correct container | F6 inspect | Answer % | Wrong % | NLL | Brier | Cost | Mean bytes |"
    separator = "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"
    lines = [title, "", header, separator]
    for kind in KINDS:
        entries = report["metrics_by_model_and_team"][kind][str(size)]
        for method in report["registered_method_order"]:
            row = entries[method]
            columns = (HEAD_NAMES[kind], METHOD_NAMES[method], f"{row['container_correct']}/{row['container_count']}",
                       f"{row['f6_inspections']}/{row['f6_variants']}", f"{100 * row['group_answer_coverage']:.1f}",
                       f"{100 * row['group_confident_wrong_rate']:.1f}", f"{row['group_mean_nll']:.3f}",
                       f"{row['group_mean_brier']:.3f}", f"{row['group_mean_decision_cost']:.3f}", f"{row['group_mean_bytes']:.0f}")
            lines.append("| " + " | ".join(columns) + " |")
    return lines


def markdown(report):
    validation = report["primary_test_release_sha256"] is None
    title = "# Fixed-profile calibration results" if validation else "# Independent primary test results"
    scope = "This is a descriptive replay of the calibration cohort." if validation else "This is the released independent test cohort."
    intro = (f"{scope} The same {report['independent_source_groups']} independent source groups appear at each team size. "
             f"The archive contains {report['recorded_variants_across_teams']} recorded team variants.")
    note = ("Correct container counts the registered resolvable scenes and charges an inspection as a missed first choice. "
            "F6 inspect counts hidden variants, with two variants per source group. Answer and wrong rates, "
            "proper scores, decision cost and serialized bytes weight source groups equally within each team size. "
            "Raw RGB payload counts compressed frame bytes. Compact and broadcast rows count their declared messages.")
    lines = [title, "", intro, "", note, ""]
    for size in report["teams"]:
        lines.extend([*table_lines(report, size), ""])
    return "\n".join(lines)


def publish(report, output, table):
    if output.resolve() == table.resolve():
        raise ValueError("The metrics JSON and Markdown table require distinct paths")
    text = markdown(report)
    data = json_bytes(report) + b"\n"
    for path, target in ((output, data), (table, text.encode())):
        if path.exists() and path.read_bytes() != target:
            raise ValueError(f"Choose a new report path to preserve {path}")
    if not output.exists():
        atomic(output, report)
    if not table.exists():
        with NamedTemporaryFile(dir=table.parent, mode="w", encoding="utf-8", delete=False) as stream:
            stream.write(text)
            pending = Path(stream.name)
        pending.replace(table)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scope", choices=("validation", "test"), required=True)
    parser.add_argument("--profiles", type=Path, default=PILOT / "primary-profiles-frozen.json")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--table", type=Path, required=True)
    args = parser.parse_args()
    try:
        publish(collect(args.profiles, args.scope), args.out, args.table)
    except (OSError, ValueError, KeyError) as error:
        parser.exit(2, f"Primary results table withheld. {error}\n")
    print(args.table)


if __name__ == "__main__":
    main()
