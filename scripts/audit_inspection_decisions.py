"""Compare saved inspection choices with forced guessing and forced inspection."""

import argparse
from collections import Counter
import json
from math import hypot, isclose, isfinite, log
from pathlib import Path
from statistics import mean, median
from tempfile import TemporaryDirectory

from archive_rosmaster_inspection import ARMS, HEADS, LABELS, PILOT, read_verified, verify
from audit_rosmaster_inspection_results import paired_effect
from smart_swarm.episode import atomic, json_bytes
from smart_swarm.inspection_comparison import INSPECT, follow_up
from smart_swarm.particle_memory import LOCATIONS

DEFAULT = PILOT / "rosmaster-bounded-comparison-homebrain.tar.gz"
STRATEGIES = ("argmax", "selective", "always_inspect")
POLICY = "selective_with_observed_marker_support"


def checked_belief(row, threshold):
    belief = row["belief"]
    if (set(belief) != set(LOCATIONS) or not all(isfinite(value) and 0 <= value <= 1 for value in belief.values())
            or not isclose(sum(belief.values()), 1., rel_tol=0., abs_tol=1e-8)):
        raise ValueError("An archived pre-inspection belief is not a probability distribution")
    winner = max(belief, key=belief.get)
    if row["decision"] != (winner if belief[winner] >= threshold else INSPECT):
        raise ValueError("The archived decision does not reproduce the frozen inspection threshold")
    return winner


def disposition(choice, branch, expected):
    result = follow_up(choice, branch)
    answer = result["final_choice"]
    return {**result, "requested_inspection": choice == INSPECT, "correct": answer == expected,
            "wrong_answer": answer != INSPECT and answer != expected, "unresolved": answer == INSPECT}


def case_methods(heads, policy, expected, thresholds):
    results = {}
    for head, threshold in thresholds.items():
        early = heads[head]
        winner = checked_belief(early, threshold)
        choices = dict(zip(STRATEGIES, (winner, early["decision"], INSPECT), strict=True))
        use = {name: {arm: disposition(choice, policy[arm], expected) for arm in ARMS}
               for name, choice in choices.items()}
        for arm in ARMS:
            if follow_up(early["decision"], policy[arm]) != policy["methods"][head][arm]:
                raise ValueError("A derived inspection decision differs from the frozen policy")
        probabilities = early["belief"]
        results[head] = {"pre_inspection": early, "argmax": winner, "max_probability": probabilities[winner],
                         "entropy_nats": -sum(value * log(value) for value in probabilities.values() if value),
                         "terminal_nll": -log(max(probabilities[expected], 1e-12)), "strategies": use}
    return results


def branch_cues(branch):
    result = branch["post_inspection_decision"]
    return {"readouts": [sample["readout"] for sample in branch["samples"]],
            "any_strong_target_cue": any(sample["readout"]["target_color_detected"] for sample in branch["samples"]),
            "used_triangulation": result["used_triangulation"], "result": result["answer"]}


def archived_case(files, attempt, thresholds):
    name = attempt["source_group"]
    base = f"cases/{name}"
    policy = json.loads(files[f"{base}/policy.json"])
    heads = json.loads(files[f"{base}/pre-inspection-decision.json"])["learned"]
    evaluated = json.loads(files[f"{base}/evaluator/result.json"])["evaluation"]
    drive = policy["scan"]["motion"]["drive"]
    return {"source_group": name, "family": attempt["family"], "variant": attempt["variant"],
            "terminal": evaluated["location"], "models": case_methods(heads, policy, evaluated["location"], thresholds),
            "cues": {arm: branch_cues(policy[arm]) for arm in ARMS}, "paired_case_wall_seconds": attempt["wall_seconds"],
            "scan_base_displacement_m": hypot(*drive["measured_xy_m"]), "scan_peak_speed_m_s": drive["max_measured_speed_m_s"],
            "scan_max_arm_error_deg": max(point["max_error_deg"] for point in policy["scan"]["motion"]["arm"])}


def strategy_counts(cases, head, strategy, arm):
    decisions = [row["models"][head]["strategies"][strategy][arm] for row in cases]
    moved = [row for row, decision in zip(cases, decisions, strict=True)
             if decision["requested_inspection"] and arm == "scan"]
    keys = ("correct", "wrong_answer", "unresolved", "requested_inspection", "extra_rgb_frames")
    stats = {key: sum(row[key] for row in decisions) for key in keys}
    return {**stats, "cases": len(cases), "scan_executions": len(moved),
            "extra_simulated_seconds": sum(row["elapsed_ns"] for row in decisions) / 1e9,
            "visible_control_inspections": sum(row["family"] == "F1" and decision["requested_inspection"]
                                               for row, decision in zip(cases, decisions, strict=True)),
            "max_scan_base_displacement_m": max((row["scan_base_displacement_m"] for row in moved), default=0.),
            "max_scan_speed_m_s": max((row["scan_peak_speed_m_s"] for row in moved), default=0.),
            "max_scan_arm_error_deg": max((row["scan_max_arm_error_deg"] for row in moved), default=0.)}


def confidence(cases, head):
    forecasts = [row["models"][head] for row in cases]
    peaks = [row["max_probability"] for row in forecasts]
    return {"range": [min(peaks), max(peaks)], "median": median(peaks),
            "mean_entropy_nats": mean(row["entropy_nats"] for row in forecasts),
            "mean_terminal_nll": mean(row["terminal_nll"] for row in forecasts),
            "mean_serialized_bytes": mean(row["pre_inspection"]["serialized_bytes"] for row in forecasts),
            "modeled_reply_latency_ms": sorted({row["pre_inspection"]["modeled_latency_ms"] for row in forecasts})}


def compare_strategies(cases, head, alternative, arm, allocated):
    rows = []
    for case in cases:
        behavior = case["models"][head]["strategies"]
        left, right = behavior["selective"][arm], behavior[alternative][arm]
        fields = lambda result: {"correct": result["correct"], "answered": not result["unresolved"],
                                 "confident_wrong": result["wrong_answer"]}
        rows.append({"scan": fields(left), "stationary": fields(right)})
    metrics = paired_effect(rows, allocated)
    return {key: value for key, value in metrics.items() if key != "branches"}


def summarize_scope(cases, allocated):
    by_head = {}
    for head in HEADS:
        by_head[head] = {"pre_inspection": confidence(cases, head),
                         "strategies": {strategy: {arm: strategy_counts(cases, head, strategy, arm) for arm in ARMS}
                                        for strategy in STRATEGIES},
                         "selective_moving_vs_controls": {
                             strategy: compare_strategies(cases, head, strategy, "scan", allocated)
                             for strategy in ("argmax", "always_inspect")}}
    return {"cases": len(cases), "allocated": allocated, "terminal_counts": dict(Counter(row["terminal"] for row in cases)),
            "method_summaries": by_head,
            "optical_cues_by_arm": {arm: {key: sum(row["cues"][arm][key] for row in cases)
                                          for key in ("any_strong_target_cue", "used_triangulation")}
                                    for arm in ARMS}}


def report(archive):
    proof = verify(archive)
    files, _ = read_verified(archive)
    profile = json.loads(files["configuration/primary-profiles-frozen.json"])
    thresholds = {head: profile["fits"][head]["mount_and_receiver_profiles"]["3"]["decision_thresholds"][POLICY] for head in HEADS}
    comparison = json.loads(files["configuration/comparison.json"])
    original = json.loads(files["analysis/original-report.json"])
    cases = [archived_case(files, row, thresholds) for row in comparison["attempts"] if row["status"] == "sealed"]
    if len({row["source_group"] for row in cases}) != proof["paired_results_reproduced"] or len(cases) != proof["paired_results_reproduced"]:
        raise ValueError("Each analyzed scene must occur once as a matched optical pair")
    scopes = {name: summarize_scope([case for case in cases if name == "all" or case["family"] == name],
                                    original["scopes"][name]["optical"]["allocated"]) for name in ("all", "F1", "F6")}
    return {"schema": "smart-swarm-archived-inspection-decision-audit-v1", "scope": "Exploratory policy replay of used MuJoCo scenes",
            "inspection_archive_sha256": proof["sha256"], "thresholds": thresholds, "scopes": scopes, "cases": cases,
            "modeled_not_radio_latency": True, "counterfactual_controls_from_same_frames": True,
            "post_inspection_calibrated_distribution_available": False, "hardware_motion_requested": False,
            "new_simulator_steps": 0, "optimizer_updates": 0}


def strategy_table(result):
    scope = result["scopes"]["all"]
    by_head = scope["method_summaries"]
    columns = ("correct", "wrong_answer", "unresolved", "requested_inspection", "visible_control_inspections", "extra_rgb_frames")
    lines = ["| Evidence model | Replay decision rule | Correct | Wrong | Unresolved | Inspect | F1 inspections | Extra RGB | Extra sim s |",
             "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    names = (("argmax", "scan", "Forced direct answer"), ("selective", "scan", "Frozen selective, moving"),
             ("always_inspect", "scan", "Forced inspection, moving"), ("selective", "stationary", "Frozen selective, stationary"),
             ("always_inspect", "stationary", "Forced inspection, stationary"))
    for head in HEADS:
        for strategy, arm, name in names:
            row = by_head[head]["strategies"][strategy][arm]
            cells = " | ".join(str(row[key]) for key in columns)
            lines.append(f"| {LABELS[head]} | {name} | {cells} | {row['extra_simulated_seconds']:.2f} |")
    return lines


def markdown(result):
    scopes = result["scopes"]
    lines = ["# Inspection decisions and additional evidence", "",
             f"Exploratory replay of {scopes['all']['cases']} of {scopes['all']['allocated']} allocated MuJoCo sources.",
             "The forced controls use the recorded pre-inspection beliefs and the same two optical exposures. "
             "The visible F1 checks count inspections where both saved policies already answered correctly.", "",
             *strategy_table(result), "", "## Uncertainty before inspection", "",
             "| Source family | Evidence model | Peak probability range | Mean entropy (nats) | Mean bytes |",
             "| --- | --- | ---: | ---: | ---: |"]
    for family in ("F1", "F6"):
        for head in HEADS:
            row = scopes[family]["method_summaries"][head]["pre_inspection"]
            low, high = row["range"]
            lines.append(f"| {family} | {LABELS[head]} | {low:.3f}–{high:.3f} | {row['mean_entropy_nats']:.3f} | "
                         f"{row['mean_serialized_bytes']:.0f} |")
    lines.extend(["", "Each requested inspection uses two local frames and 27.09 additional simulated seconds in either branch. "
                  "Reply latency is modeled. The optical follow-up supplies a location or remains unresolved.", ""])
    return "\n".join(lines)


def figure(result, output):
    from study_figures import plt

    fig, axis = plt.subplots(figsize=(6.3, 3.2), layout="constrained")
    options = (("F1", "events_only"), ("F1", "events_plus_reveal"), ("F6", "events_only"), ("F6", "events_plus_reveal"))
    for index, (family, head) in enumerate(options):
        values = sorted(row["models"][head]["max_probability"] for row in result["cases"] if row["family"] == family)
        spread = [(i - (len(values) - 1) / 2) * .045 for i in range(len(values))]
        tone = "#245a75" if family == "F1" else "#a44730"
        axis.scatter(values, [index + offset for offset in spread], color=tone, s=22, alpha=.72)
    thresholds = sorted(set(result["thresholds"].values()))
    for limit in thresholds:
        axis.axvline(limit, color="#555555", linestyle="--", linewidth=1)
    axis.set(yticks=list(range(len(options))), yticklabels=[f"{family} · {LABELS[head]}" for family, head in options],
             xlim=(.45, 1.01), ylim=(3.6, -.6), xlabel="Largest pre-inspection location probability")
    axis.set_title("Uncertainty before simulated inspection\nDashed line shows the frozen threshold")
    axis.grid(axis="x", alpha=.16)
    axis.spines[["top", "right", "left"]].set_visible(False)
    fig.savefig(output, metadata={"CreationDate": None, "ModDate": None})
    plt.close(fig)


def publish(result, output, table, image):
    if len({path.resolve() for path in (output, table, image)}) != 3:
        raise ValueError("Choose distinct metrics, table and figure paths")
    with TemporaryDirectory(prefix="smart-swarm-inspection-decisions-") as directory:
        pdf = Path(directory) / "figure.pdf"
        figure(result, pdf)
        additions = {output: json_bytes(result) + b"\n", table: markdown(result).encode(), image: pdf.read_bytes()}
        for path, data in additions.items():
            if path.exists() and path.read_bytes() != data:
                raise ValueError(f"Choose a new publication path to preserve {path}")
        for path, data in additions.items():
            if not path.exists():
                if path == output:
                    atomic(output, result)
                else:
                    with path.open("xb") as stream:
                        stream.write(data)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=DEFAULT)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--table", type=Path, required=True)
    parser.add_argument("--figure", type=Path, required=True)
    args = parser.parse_args()
    publish(report(args.archive), args.output, args.table, args.figure)
    print(args.output)


if __name__ == "__main__":
    main()
