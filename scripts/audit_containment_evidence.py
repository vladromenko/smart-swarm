#!/usr/bin/env python3
"""Audit SO101 target support and replay uncertainty diagnostics on recorded inputs."""

import argparse
from collections import Counter
from pathlib import Path
from statistics import median

import numpy as np

from smart_swarm.contract import json_bytes
from smart_swarm.episode import Replay, atomic, load, sha
from smart_swarm.evidence_inputs import clip_packets
from smart_swarm.memory import TRACKS
from smart_swarm.vision import center, insertion

ROOT = Path(__file__).resolve().parents[1]
PILOT = ROOT / "evidence/learning-pilot-v3"
WORK = ROOT.parent.parent / "work"
STATION = "so101-1/top"


def station_support(root):
    client = Replay(root, "so101-1")
    positions = []
    image = None
    packets = clip_packets(client)
    for packet in packets:
        frame = next(row for row in packet["cameras"] if row["sensor_id"] == STATION)
        image = np.asarray(client.image(frame["frame_id"]))
        positions.append(center(image, "cube"))
    supported = image is not None and all(center(image, track) is not None for track in TRACKS)
    cue = insertion(packets[-1], image) if supported else None
    return {"clip_frames": len(positions), "cube_detected_frames": sum(point is not None for point in positions),
            "last_cube_xy_pixels": positions[-1] if positions else None,
            "last_station_rule": max(cue.hypotheses, key=cue.hypotheses.get) if cue else None}


def support_summary(items):
    final = [row["last_cube_xy_pixels"] for row in items if row["last_cube_xy_pixels"] is not None]
    return {"source_groups": len({row["group"] for row in items}), "recorded_variants": len(items),
            "groups_with_any_cube_detection": len({row["group"] for row in items if row["cube_detected_frames"]}),
            "groups_with_last_cube_detection": len({row["group"] for row in items if row["last_cube_xy_pixels"]}),
            "last_position_min_median_max_xy": [[min(axis), median(axis), max(axis)] for axis in zip(*final)],
            "isolated_placement_labels_by_variant": dict(Counter(str(row["label"]) for row in items))}


def pilot_coverage(study):
    path = WORK / "evidence-pilot-v3/portfolio.json"
    if study["input_portfolio_sha256"] != sha(path):
        raise ValueError("The original fitting portfolio changed")
    candidates = [row for row in load(path)["records"] if row["split"] in ("train", "validation")]
    data = [{**station_support(Path(row["path"])), "group": row["scene_group_id"], "family": row["family"],
             "split": row["split"], "label": row["settled_placement"]} for row in candidates]
    coverage = {part: {f"F{i}": support_summary([row for row in data if row["split"] == part and row["family"] == f"F{i}"])
                       for i in range(1, 7)} for part in ("train", "validation")}
    return {"portfolio_sha256": sha(path), "summary_by_split_and_family": coverage, "observed": data}


def archived_cases(pilot, kind):
    validation = load(PILOT / f"primary-frozen-rehearsal-eighteen-{kind}.json")
    groups = [row for row in validation["outcomes"] if row["team_size"] == 3]
    result = []
    for group in groups:
        for row in group["outcomes"]:
            if sha(Path(row["source"]) / "manifest.json") != row["source_manifest_sha256"]:
                raise ValueError("A fixed validation recording changed")
            result.append({"group": group["group"], "truth": row["truth"], "family": row["family"],
                           "path": row["source"], "fixed": row["learned_selective_joint_memory"]})
    for name, condition in pilot["conditions"].items():
        receipt = Path(condition["receipt"])
        forecasts = receipt.with_name("predictions.json")
        if sha(forecasts) != condition["prediction_sha256"] or sha(receipt) != condition["receipt_sha256"]:
            raise ValueError("A matched control receipt changed")
        cases = zip(condition["records"], load(forecasts)[kind], ("F1", "F2"), strict=True)
        for row, forecast, family in cases:
            result.append({"group": name, "truth": row["outcome"], "family": family, "path": row["raw_path"],
                           "fixed": forecast["learned_selective_joint_memory"]})
    return result


def neutral_report(replies, observed, reference, *, check_conflict):
    adjusted = []
    for reply in replies:
        change = False
        if reply["factor"] == "containment":
            scores = reply["probabilities"]
            learned = max(scores, key=lambda label: float(scores[label]))
            cue = observed["last_station_rule"]
            change = observed["cube_detected_frames"] == 0 or bool(check_conflict and cue and cue != learned)
        adjusted.append({**reply, "probabilities": reference} if change else reply)
    return adjusted


def score_case(row, fit):
    from score_primary_validation import choose, exact_from_reports, on_time_compact

    root = Path(row["path"])
    deadline = Replay(root, "rosmaster-1").decision
    original = row["fixed"]
    replies = on_time_compact(original["wire"]["replies"], deadline)
    support = station_support(root)
    common = fit["common"]
    reference = {name: format(value, ".8f") for name, value in common["reference"]["containment"].items()}
    outcomes = {}
    for label, inputs in (("fixed", replies), ("visibility", neutral_report(replies, support, reference, check_conflict=False)),
                          ("visibility_and_conflict", neutral_report(replies, support, reference, check_conflict=True))):
        if len(json_bytes(inputs)) != len(json_bytes(replies)):
            raise ValueError("A containment diagnostic changed the size of the original wire")
        current = choose(exact_from_reports(inputs, fit["association"], common), fit["threshold"])
        outcomes[label] = {"decision": current["decision"], "belief": current["belief"], "neutralized": inputs != replies}
    fixed = outcomes["fixed"]
    if (fixed["decision"] != original["decision"]
            or any(abs(value - original["belief"][label]) > 1e-12 for label, value in fixed["belief"].items())):
        raise ValueError("The diagnostic must reproduce the real frozen prediction before changing a report")
    return {key: value for key, value in row.items() if key != "fixed"} | {"support": support, "outcomes": outcomes}


def summarize_scored(rows):
    controls = [row for row in rows if row["family"] in ("F1", "F2", "F3", "F5")]
    f6 = [row for row in rows if row["family"] == "F6"]
    names = ("fixed", "visibility", "visibility_and_conflict")
    result = {}
    for name in names:
        decisions = [row["outcomes"][name]["decision"] for row in rows]
        result[name] = {"resolvable_correct": sum(row["outcomes"][name]["decision"] == row["truth"] for row in controls),
                        "f6_inspections": sum(row["outcomes"][name]["decision"] == "request_inspection" for row in f6),
                        "confident_wrong": sum(choice not in (row["truth"], "request_inspection")
                                               for choice, row in zip(decisions, rows, strict=True)),
                        "inspections": decisions.count("request_inspection"),
                        "neutralized_reports": sum(row["outcomes"][name]["neutralized"] for row in rows)}
    for group in {row["group"] for row in f6}:
        pair = [row for row in f6 if row["group"] == group]
        if len(pair) != 2 or pair[0]["outcomes"] != pair[1]["outcomes"]:
            raise ValueError("The diagnostic changed an output across identical F6 inputs")
    return {"independent_groups": len({row["group"] for row in rows}), "resolvable": len(controls),
            "f6_variants": len(f6), "methods": result}


def frozen_fit(study, profile, kind):
    from audit_evidence_replay import verified_model
    from score_primary_validation import SELECTIVE
    from smart_swarm.evidence_comparison import learned_prior_map

    sealed, _ = verified_model(next(row for row in study["models"] if row["kind"] == kind))
    mount = profile["fits"][kind]["mount_and_receiver_profiles"]["3"]
    return {"common": {"state_prior": sealed["state_prior"], "reference": learned_prior_map(sealed["priors"])},
            "association": mount["association"]["compact"], "threshold": mount["decision_thresholds"][SELECTIVE]}


def evaluate(study, profile, control):
    result = {}
    for kind in profile["fits"]:
        fit = frozen_fit(study, profile, kind)
        rows = [score_case(row, fit) for row in archived_cases(control, kind)]
        validation = [row for row in rows if row["group"].startswith("primary-validation-")]
        results = {"validation": summarize_scored(validation), "rows": rows}
        for name in control["conditions"]:
            results[name] = summarize_scored([row for row in rows if row["group"] == name])
        result[kind] = results
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--coverage-only", action="store_true", help="Audit archived RGB inputs without reloading checkpoints")
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Choose a new report path to preserve the original diagnostic")
    study = load(PILOT / "story-learning.json")
    result = {"scope": "Development and existing validation only", "pilot_inputs": pilot_coverage(study)}
    if not args.coverage_only:
        from freeze_primary_profiles import require_profiles

        profile_path = PILOT / "primary-profiles-eighteen-draft.json"
        control_path = PILOT / "primary-matched-history-controls.json"
        profile = require_profiles(profile_path, rehearsal=True)
        result.update(diagnostic=evaluate(study, profile, load(control_path)),
                      profile_sha256=sha(profile_path), matched_development_source_groups=1,
                      changed_weights_or_primary_protocol=False)
    atomic(args.output, result)
    print(args.output)


if __name__ == "__main__":
    main()
