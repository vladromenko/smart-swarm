#!/usr/bin/env python3
"""Reproduce the registered three-robot controls from sealed simulation scores."""

import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import tarfile
from tempfile import NamedTemporaryFile

from archive_primary_release import INDEX, KINDS, read_member, verify as verify_calibration
from archive_primary_results import verify as verify_results
from audit_primary_ablations import oracles, report as original_report
from audit_primary_delivery_delay import CALIBRATION, MANIFEST, RESULTS, RESULTS_SHA256, frozen_parameters
from audit_primary_pairs import conservative_interval, paired_rows
from compare_primary_supervision import FIELDS, describe
from export_primary_result_table import HEAD_NAMES
from score_primary_test import check_prediction_receipt
from score_primary_validation import SELECTIVE
from smart_swarm.contract import json_bytes

METHODS = {
    SELECTIVE: "Learned selective",
    "learned_broadcast": "Learned broadcast",
    "single_information_gain": "One reply, validation information gain",
    "single_latest_witness": "One reply, latest witness",
    "reversed_witness_time": "Reversed witness clips",
    "withheld_association": "Association likelihood withheld",
    "particle_matched_learned_factors": "Particle filter, same learned likelihoods",
    "isolated_event_oracle": "Scoring-only event-label oracle",
    "isolated_terminal_oracle": "Scoring-only terminal oracle",
}


def bound_inputs(archive, calibration_path):
    index = json.loads(read_member(archive, INDEX))
    profile = json.loads(read_member(archive, MANIFEST))
    release = json.loads(read_member(archive, "configuration/primary-test-release.json"))
    calibration = verify_calibration(calibration_path)
    if (calibration["archive_sha256"] != index["external_calibration_archive"]["sha256"]
            or calibration["profile_sha256"] != index["profiles_sha256"]):
        raise ValueError("Calibration differs from the original test release")
    with tarfile.open(calibration_path) as bundle:
        raw = read_member(bundle, "configuration/primary-sim-source-allocation.json")
        config_raw = read_member(bundle, "configuration/primary-memory-study.json")
    if sha256(raw).hexdigest() != release["allocation_sha256"] or sha256(config_raw).hexdigest() != release["config_sha256"]:
        raise ValueError("Source allocation or configuration differs from the original release")
    slots = {row["source_group"]: row for row in json.loads(raw)["slots"] if row["split"] == "test"}
    subset = {name: row["family"] for name, row in slots.items() if row["frozen_n3_ablation_subset"]}
    if len(slots) != 100 or len(subset) != 50:
        raise ValueError("The released analysis subset changed")
    config, priors = frozen_parameters(calibration_path, profile)
    manifests = {str(Path(path).parent): row["sha256"] for path, row in index["original_recording_manifests"].items()}
    identity = {"scope": "independent primary test", "profiles_sha256": index["profiles_sha256"],
                "primary_test_release_sha256": index["primary_test_release_sha256"],
                "manifests": manifests, "source_groups": sorted(slots)}
    return identity, subset, profile, config, priors


def signed_groups(archive, kind, identity, subset):
    path = f"scores/primary-test-{kind}"
    receipt_bytes = read_member(archive, f"{path}-predictions.json")
    score = json.loads(read_member(archive, f"{path}.json"))
    digest = sha256(receipt_bytes).hexdigest()
    cache = {**identity, "checkpoint_kind": kind}
    if (score["predictions_sha256"] != digest or score["fit_on_inputs"] or score["training_labels_read_by_predictor"]
            or any(score[key] != value for key, value in cache.items())):
        raise ValueError("Original evaluation disagrees with its policy seal")
    predictions = check_prediction_receipt(json.loads(receipt_bytes), cache, identity["manifests"])
    groups, visited = {}, set()
    for group in score["outcomes"]:
        name, size = group["group"], group["team_size"]
        expected = size == 3 and name in subset
        for row in group["outcomes"]:
            source = row["source"]
            payload = {key: value for key, value in row.items() if key not in FIELDS}
            if (source in visited or predictions[source] != payload or ("bounded_ablations" in row) != expected
                    or row["source_manifest_sha256"] != identity["manifests"][source]):
                raise ValueError("The registered controls do not match the pre-evaluator trace")
            visited.add(source)
        if expected:
            if name in groups or any(row["family"] != subset[name] for row in group["outcomes"]):
                raise ValueError("A source group is repeated or assigned to the wrong family")
            groups[name] = group
    if set(groups) != set(subset) or visited != set(predictions):
        raise ValueError("The score omits a reserved source or forecast")
    return list(groups.values()), score["bounded_n3_ablations"], digest


def score_controls(groups, common, threshold, config):
    material = [{**group, "outcomes": [{**row, **row["bounded_ablations"], **oracles(row, common, threshold)}
                                      for row in group["outcomes"]]} for group in groups]
    ambiguous = [group for group in material if group["outcomes"][0]["family"] == "F6"]
    applicable = [name for name in METHODS if name != "isolated_terminal_oracle"]
    if not all(len(pair) == 2 and pair[0]["truth"] != pair[1]["truth"]
               and all(pair[0][method] == pair[1][method] for method in applicable)
               for pair in (group["outcomes"] for group in ambiguous)):
        raise ValueError("A paired pre-inspection control changed under indistinguishable histories")
    reference = paired_rows(material, SELECTIVE)
    return {method: {**describe(material, method, config), "first_choice_vs_selective":
                     conservative_interval(paired_rows(material, method), reference)} for method in METHODS}


def collect(results, calibration):
    integrity = verify_results(results)
    if integrity["archive_sha256"] != RESULTS_SHA256:
        raise ValueError("The analysis requires the original independent result archive")
    with tarfile.open(results) as archive:
        identity, subset, profile, config, priors = bound_inputs(archive, calibration)
        measurements, diagnostics, digests = {}, {}, {}
        for kind in KINDS:
            groups, original, digests[kind] = signed_groups(archive, kind, identity, subset)
            common = {**priors[kind], "inspect_cost": config["inspect_cost"], "wrong_cost": config["wrong_answer_cost"]}
            threshold = profile["fits"][kind]["mount_and_receiver_profiles"]["3"]["decision_thresholds"][SELECTIVE]
            diagnostics[kind] = recomputed = original_report(groups, common, threshold)
            if recomputed != original:
                raise ValueError(f"Regenerated controls disagree with the original score for {kind}")
            measurements[kind] = score_controls(groups, common, threshold, config)
    return {"schema": "smart-swarm-primary-ablation-table-v1", "primary_archive_sha256": RESULTS_SHA256,
            "profile_sha256": identity["profiles_sha256"], "prediction_seals": digests, "team_size": 3,
            "scope": "Registered secondary controls from the original sealed primary simulation score",
            "source_groups": sorted(subset), "independent_groups_by_family": dict(Counter(subset.values())),
            "bytes_cap": diagnostics[KINDS[0]]["common_cap_bytes"], "metrics_by_head": measurements,
            "predictions_reused": True, "optimizer_updates": 0, "new_simulation_steps": 0,
            "diagnostics": {kind: {key: value for key, value in report.items() if key not in ("metrics", "source_groups")}
                            for kind, report in diagnostics.items()}}


def lines_for_head(report, kind):
    rows = report["metrics_by_head"][kind]
    lines = [f"## {HEAD_NAMES[kind]}", "",
             "| Method | First container | Resolving correct | F6 pairs inspect | Wrong variants | NLL | Mean bytes |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for name, label in METHODS.items():
        row = rows[name]
        container = f"{row['container_correct']}/{row['container_count']}"
        resolved = f"{row['resolvable_correct']}/{row['resolvable_count']}"
        pairs = f"{row['f6_inspections'] // 2}/{row['f6_variants'] // 2}"
        lines.append(f"| {label} | {container} | {resolved} | {pairs} | {row['confident_wrong']} | "
                     f"{row['group_mean_nll']:.3f} | {row['group_mean_bytes']:.0f} |")
    return [*lines, ""]


def markdown(report):
    intro = ["# Registered three-robot controls", "",
             f"The two evidence heads share {len(report['source_groups'])} prespecified independent groups. "
             "The original frozen forecasts were reused. The two hidden variants of each F6 group count as one pair. "
             f"The common message ceiling is {report['bytes_cap']:,} bytes. NLL and byte means weight source groups equally.", "",
             "Reversing the witness clips changes frame order and the latest frame seen by each visual head. "
             "Withholding association drops its likelihood at fusion and keeps the original transmission charge. "
             "Only the scoring oracles access evaluator or event labels.", ""]
    notes = ["## Transport and exact final-view matches", ""]
    for kind in KINDS:
        stats = report["diagnostics"][kind]
        same = stats["same_final_receiver_observations"]["same_receiver_last_rgb_distinct_witness_histories_and_answers"]
        notes.append(f"- {HEAD_NAMES[kind]} had maximum joint total variation {stats['max_delivery_control_joint_tv']:.3g} "
                     f"after reordering, duplicating or adding late copies of the admitted messages. The matched particle "
                     f"filter differed from exact fusion by at most {stats['max_matched_particle_joint_tv']:.4f}. "
                     f"There were {len(same)} resolving pairs with exactly equal final receiver RGB, distinct witness histories "
                     "and different correct answers.")
    return "\n".join([*intro, *(line for kind in KINDS for line in lines_for_head(report, kind)), *notes, ""])


def publish(report, output, table, *, render=markdown):
    artifacts = ((output, json_bytes(report) + b"\n"), (table, render(report).encode()))
    for path, data in artifacts:
        if path.exists() and path.read_bytes() != data:
            raise ValueError(f"Choose a new publication path to preserve {path}")
    for path, data in artifacts:
        with NamedTemporaryFile(dir=path.parent, prefix=path.name + ".") as staged:
            staged.write(data)
            staged.flush()
            try:
                path.hardlink_to(staged.name)
            except FileExistsError:
                if path.read_bytes() != data:
                    raise ValueError(f"A concurrent report used {path} for different content")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=RESULTS)
    parser.add_argument("--calibration", type=Path, default=CALIBRATION)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--table", type=Path, required=True)
    args = parser.parse_args()
    if args.out.resolve() == args.table.resolve():
        parser.error("Choose distinct paths for the JSON report and the table")
    try:
        publish(collect(args.results, args.calibration), args.out, args.table)
    except (OSError, ValueError, KeyError, tarfile.TarError) as error:
        parser.exit(2, f"Primary control table withheld. {error}\n")
    print(args.table)


if __name__ == "__main__":
    main()
