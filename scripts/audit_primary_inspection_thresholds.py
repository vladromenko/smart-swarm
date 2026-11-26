"""Measure inspection coverage and confidence overlap from the sealed primary scores."""

import argparse
from collections import Counter, defaultdict
from hashlib import sha256
from io import BytesIO
import json
import math
from pathlib import Path
from statistics import mean
import tarfile

from archive_primary_release import INDEX, KINDS, read_member, verify as verify_release
from archive_primary_results import verify as verify_results
from audit_primary_delivery_delay import CALIBRATION, MANIFEST, RESULTS, RESULTS_SHA256
from compare_primary_supervision import FIELDS, TEAMS, paired_interval
from export_primary_ablation_table import publish
from export_primary_result_table import HEAD_NAMES
from freeze_primary_profiles import PILOT
from score_primary_test import check_prediction_receipt
from score_primary_validation import IDENTITY, PARTICLE, SELECTIVE

CALIBRATION_SCORES = PILOT / "primary-calibration-results-homebrain.tar.gz"
CALIBRATION_SCORES_SHA256 = "57b7037c771a06c553bda0aef40591ebe845a4a6456443956eaad6ba238ca792"
METHODS = {SELECTIVE: "Learned selective", IDENTITY: "Visual marker check", PARTICLE: "Particle broadcast"}
INSPECT = "request_inspection"
RESOLVING = {"F1", "F2", "F3", "F5"}
METADATA = {"validation": (48, 168, "primary-frozen-rehearsal-forty-eight", "validation replay using fixed profiles"),
            "test": (100, 348, "primary-test", "independent primary test")}


def index_sources(archive, binding, split):
    result, paths = {}, {}
    for path, row in binding["original_recording_manifests"].items():
        source = str(Path(path).parent)
        manifest = json.loads(read_member(archive, row["archive_member"]))
        if manifest["lineage"]["split"] != split or manifest["world_episode_id"] != Path(source).name:
            raise ValueError("An archived observation escaped its original split or identity")
        result[source] = (manifest["scene_group_id"], manifest["scaling_team_size"], manifest["family"], manifest["evidence_variant"])
        paths[source] = row["sha256"]
    names = {row[0] for row in result.values()}
    groups, variants, _, label = METADATA[split]
    if len(names) != groups or len(result) != variants or {(v[0], v[1]) for v in result.values()} != {
            (name, size) for name in names for size in TEAMS}:
        raise ValueError("The archived cohort lacks its declared groups and team conditions")
    identity = {"scope": label, "profiles_sha256": binding["profiles_sha256"], "source_groups": sorted(names),
                "primary_test_release_sha256": binding["primary_test_release_sha256"], "manifests": paths}
    return identity, result


def sealed_sources(archive, identity, sources, split, kind):
    stem = f"scores/{METADATA[split][2]}-{kind}"
    raw = read_member(archive, f"{stem}-predictions.json")
    scored = json.loads(read_member(archive, f"{stem}.json"))
    expected = {**identity, "checkpoint_kind": kind}
    if (sha256(raw).hexdigest() != scored["predictions_sha256"] or scored["fit_on_inputs"] is not False
            or scored["training_labels_read_by_predictor"] is not False
            or any(scored[key] != value for key, value in expected.items())):
        raise ValueError("The archive's score disagrees with its pre-evaluation binding")
    predictions = check_prediction_receipt(json.loads(raw), expected, identity["manifests"])
    seen, groups = set(), defaultdict(list)
    for group in scored["outcomes"]:
        for case in group["outcomes"]:
            path = case["source"]
            key = group["group"], group["team_size"], case["family"], case["variant"]
            if (path in seen or sources[path] != key or case["source_manifest_sha256"] != identity["manifests"][path]
                    or {field: value for field, value in case.items() if field not in FIELDS} != predictions[path]):
                raise ValueError("A saved posterior differs from its manifest or forecast")
            seen.add(path)
        groups[group["team_size"]].append(group)
    if seen != set(sources) or any(sorted(group["group"] for group in groups[size]) != identity["source_groups"] for size in TEAMS):
        raise ValueError("The analysis omitted or duplicated an admitted source")
    return groups


def group_confidences(groups, method, frozen):
    material = []
    for group in groups:
        cases = []
        for row in group["outcomes"]:
            state = row[method]
            belief = state["belief"]
            if (row["truth"] not in belief or any(not math.isfinite(p) or not 0 <= p <= 1 for p in belief.values())
                    or not math.isclose(sum(belief.values()), 1., abs_tol=1e-8)):
                raise ValueError("Invalid terminal posterior or evaluator label")
            winner = max(belief, key=belief.get)
            confidence = belief[winner]
            if (winner if confidence >= frozen else INSPECT) != state["decision"]:
                raise ValueError("Recomputed threshold decision differs from the frozen policy")
            cases.append({"family": row["family"], "truth": row["truth"], "winner": winner,
                          "confidence": confidence, "weight": 1. / len(group["outcomes"]), "belief": belief})
        if cases[0]["family"] == "F6":
            if len(cases) != 2 or cases[0]["belief"] != cases[1]["belief"] or cases[0]["truth"] == cases[1]["truth"]:
                raise ValueError("An ambiguous pair lacks coupled posteriors and different hidden truths")
        elif len(cases) != 1:
            raise ValueError("Only F6 may contribute coupled variants in this cohort")
        material.append(cases)
    return material


def threshold_row(groups, threshold, config):
    cases = [row for group in groups for row in group]
    answering = [row for row in cases if row["confidence"] >= threshold]
    wrong = [row for row in answering if row["winner"] != row["truth"]]
    controls = [row for row in cases if row["family"] in RESOLVING]
    resolved = [row for row in answering if row["family"] in RESOLVING and row["winner"] == row["truth"]]
    paired = [group for group in groups if group[0]["family"] == "F6"]
    answered_weight = sum(row["weight"] for row in answering)
    error_weight = sum(row["weight"] for row in wrong)
    inspection_weight = len(groups) - answered_weight
    return {"threshold": threshold, "group_answer_coverage": answered_weight / len(groups),
            "risk_among_answers": error_weight / answered_weight if answered_weight else None,
            "group_mean_decision_cost": (inspection_weight * config["inspect_cost"] +
                                         error_weight * config["wrong_answer_cost"]) / len(groups),
            "wrong_variants": len(wrong), "wrong_by_family": dict(Counter(row["family"] for row in wrong)),
            "resolving_correct": len(resolved), "resolving_controls": len(controls),
            "container_correct": sum(row["truth"] in ("left", "right") for row in resolved),
            "container_controls": sum(row["truth"] in ("left", "right") for row in controls),
            "f6_inspected_pairs": sum(all(row["confidence"] < threshold for row in group) for group in paired),
            "f6_pairs": len(paired)}


def threshold_bounds(groups):
    cases = [row for group in groups for row in group]
    controls = [row for row in cases if row["family"] in RESOLVING]
    wrong = [row for row in cases if row["truth"] != row["winner"]]
    avoid = [row for row in cases if row["family"] == "F6" or row["truth"] != row["winner"]]
    low, high = max(avoid, key=lambda row: row["confidence"]), min(controls, key=lambda row: row["confidence"])
    control_errors = sum(row["truth"] != row["winner"] for row in controls)
    return {"must_exceed": low["confidence"], "must_exceed_family": low["family"], "at_most": high["confidence"],
            "at_most_family": high["family"], "resolving_argmax_errors": control_errors,
            "highest_wrong_confidence": max(row["confidence"] for row in wrong),
            "feasible": bool(not control_errors and low["confidence"] < high["confidence"])}


def measures(groups, method, threshold, config):
    confidences = group_confidences(groups, method, threshold)
    levels = sorted({row["confidence"] for group in confidences for row in group}, reverse=True)
    curve = [threshold_row(confidences, value, config) for value in (math.nextafter(1., math.inf), *levels)]
    fixed = threshold_row(confidences, threshold, config)
    return {"frozen": fixed, "risk_coverage": curve, "separation": threshold_bounds(confidences)}


def marker_cost_difference(groups, config):
    def loss(case, method):
        decision = case[method]["decision"]
        if decision == INSPECT:
            return config["inspect_cost"]
        return config["wrong_answer_cost"] * (decision != case["truth"])

    paired = [mean(loss(case, IDENTITY) - loss(case, SELECTIVE) for case in group["outcomes"]) for group in groups]
    return {**paired_interval(paired), "direction": "marker check minus learned selective",
            "independent_source_groups": len(groups)}


def analyze_archive(path, split, calibration, config):
    proof = verify_results(path)
    digest = RESULTS_SHA256 if split == "test" else CALIBRATION_SCORES_SHA256
    if proof["scope"] != split or proof["archive_sha256"] != digest:
        raise ValueError("Use the original primary score and calibration score archives")
    with tarfile.open(path) as archive:
        binding = json.loads(read_member(archive, INDEX))
        profile = json.loads(read_member(archive, MANIFEST))
        if (binding["profiles_sha256"] != calibration["profile_sha256"]
                or binding["external_calibration_archive"]["sha256"] != calibration["archive_sha256"]):
            raise ValueError("The calibrated profile differs between the two cohorts")
        identity, sources = index_sources(archive, binding, split)
        result = {}
        for kind in KINDS:
            groups = sealed_sources(archive, identity, sources, split, kind)
            mounts = profile["fits"][kind]["mount_and_receiver_profiles"]
            result[kind] = {str(size): {method: measures(groups[size], method, mounts[str(size)]["decision_thresholds"][method], config)
                                       for method in METHODS} for size in TEAMS}
            for size in TEAMS:
                result[kind][str(size)]["marker_cost_vs_selective"] = marker_cost_difference(groups[size], config)
    return {"archive_sha256": digest, "source_groups": identity["source_groups"], "by_head_and_team": result}


def collect(test, validation, calibration_path):
    calibration = verify_release(calibration_path)
    with tarfile.open(calibration_path) as archive:
        config = json.loads(read_member(archive, "configuration/primary-memory-study.json"))
    splits = {key: analyze_archive(path, key, calibration, config) for key, path in (("validation", validation), ("test", test))}
    if set(splits["validation"]["source_groups"]) & set(splits["test"]["source_groups"]):
        raise ValueError("Calibration and test sources overlap")
    return {"schema": "smart-swarm-frozen-inspection-thresholds-v1", "scope": "Descriptive analysis on the original scored cohorts",
            "profile_sha256": calibration["profile_sha256"], "original_thresholds_retained": True,
            "fit_performed": False, "threshold_selected_from_test": False,
            "costs": {key: config[key] for key in ("inspect_cost", "wrong_answer_cost")}, "splits": splits}


def markdown(report):
    lines = ["# Confidence cutoffs for inspection", "",
             "Saved posteriors for 48 calibration and 100 independent test sources. Team sizes reuse the same source groups. "
             "This table shows N=8 on the original test. Confidence equal to the threshold results in a direct answer. "
             "Coverage and the wrong share among answers give each source equal weight. Both F6 variants share one source weight.", "",
             "| Objective | Method | Frozen cutoff | Coverage | Wrong variants | Resolving correct | F6 pairs inspect | Cutoff strictly above | Cutoff at most |",
             "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for kind in KINDS:
        for method, name in METHODS.items():
            row = report["splits"]["test"]["by_head_and_team"][kind]["8"][method]
            point, bounds = row["frozen"], row["separation"]
            lines.append(f"| {HEAD_NAMES[kind]} | {name} | {point['threshold']:.2f} | {point['group_answer_coverage']:.1%} | "
                         f"{point['wrong_variants']} | {point['resolving_correct']}/{point['resolving_controls']} | "
                         f"{point['f6_inspected_pairs']}/{point['f6_pairs']} | {bounds['must_exceed']:.4f} | {bounds['at_most']:.4f} |")
    lines.extend(["", "The final two columns describe the bounds for retaining every resolving control while avoiding all wrong "
                  "direct choices and inspecting every F6 pair. Feasibility also requires a correct argmax on each resolving control. "
                  "The cutoff range is descriptive. All evaluated thresholds remain the ones selected on validation.", "",
                  "Dots on the coverage and risk figure identify the validation-selected operating points. The report includes "
                  "paired source bootstrap intervals for the change in decision cost with the marker check.", ""])
    return "\n".join(lines)


def figure(report, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (risk_axis, confidence_axis) = plt.subplots(1, 2, figsize=(9.2, 4.1), layout="constrained")
    test = report["splits"]["test"]["by_head_and_team"]
    labels = []
    for index, (kind, color) in enumerate(zip(KINDS, ("#245975", "#a24a24"), strict=True)):
        for offset, (method, style) in enumerate(((SELECTIVE, "-"), (IDENTITY, "--"))):
            name = HEAD_NAMES[kind] + (" with marker check" if offset else "")
            row = test[kind]["8"][method]
            points = [item for item in row["risk_coverage"] if item["risk_among_answers"] is not None]
            risk_axis.plot([100 * item["group_answer_coverage"] for item in points], [100 * item["risk_among_answers"] for item in points],
                           color=color, linestyle=style, label=name)
            frozen = row["frozen"]
            risk_axis.scatter(100 * frozen["group_answer_coverage"], 100 * frozen["risk_among_answers"], color=color, zorder=4, s=22)
            labels.append(name)
            position, bounds = index * 2 + offset, row["separation"]
            confidence_axis.scatter(bounds["must_exceed"], position, color=color, marker="x", s=34, zorder=4)
            confidence_axis.scatter(bounds["at_most"], position, color=color, marker="o", s=23, zorder=4)
            confidence_axis.hlines(position, *sorted((bounds["must_exceed"], bounds["at_most"])), color=color, linestyle=style)
    risk_axis.set(xlabel="Answered sources (%)", ylabel="Wrong among answers (%)", title="Coverage and selective risk")
    risk_axis.legend(loc="upper left", fontsize=7, frameon=False)
    confidence_axis.set(xlabel="Confidence   × lower bound    ● upper bound", title="Cutoff bounds on the observed test",
                        yticks=range(4), yticklabels=labels, xlim=(.46, 1.01))
    confidence_axis.invert_yaxis()
    fig.suptitle("Inspection at eight robots · 100 independent sources", fontsize=12)
    for axis in (risk_axis, confidence_axis):
        axis.grid(axis="x", alpha=.15)
        axis.spines[["top", "right"]].set_visible(False)
    output = BytesIO()
    fig.savefig(output, format="pdf", metadata={"CreationDate": None, "ModDate": None})
    plt.close(fig)
    if path.exists() and path.read_bytes() != output.getvalue():
        raise ValueError(f"Preserve the existing figure at {path}")
    if not path.exists():
        with path.open("xb") as stream:
            stream.write(output.getvalue())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test", type=Path, default=RESULTS)
    parser.add_argument("--validation", type=Path, default=CALIBRATION_SCORES)
    parser.add_argument("--calibration", type=Path, default=CALIBRATION)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--table", type=Path, required=True)
    parser.add_argument("--figure", type=Path, required=True)
    args = parser.parse_args()
    if len({path.resolve() for path in (args.output, args.table, args.figure)}) != 3:
        parser.error("The report, table and figure require distinct destinations")
    try:
        report = collect(args.test, args.validation, args.calibration)
        publish(report, args.output, args.table, render=markdown)
        figure(report, args.figure)
    except (OSError, ValueError, KeyError, tarfile.TarError) as error:
        parser.exit(2, f"Inspection analysis withheld. {error}\n")


if __name__ == "__main__":
    main()
