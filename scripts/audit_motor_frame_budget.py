#!/usr/bin/env python3
"""Use preserved motor RGB to bound the potential benefit of exact-frame reuse."""

import argparse
from hashlib import sha256
from pathlib import Path

import numpy as np

from smart_swarm.episode import atomic, load, sha

PILOT = Path(__file__).resolve().parents[1] / "evidence/learning-pilot-v3"
RAW = PILOT.parents[3] / "work/motor-aligned-v3-raw"
GOALS = {"train": 80, "validation": 20}


def frame_counts(path):
    info = load(path)
    if info["status"] != "accepted" or not info["physics"]["released_stable_containment"]:
        raise ValueError(f"{path.parent.name} failed the teacher contact gate")
    archive_path = path.with_name("motor.npz")
    if sha(archive_path) != info["sha256"]:
        raise ValueError(f"{path.parent.name} differs from its original dataset hash")
    with np.load(archive_path, allow_pickle=False) as archive:
        counts, lengths = {}, set()
        for name in ("wrist", "top"):
            images = archive[name]
            counts[name] = len({sha256(frame).digest() for frame in images})
            lengths.add(len(images))
    if lengths != {info["sample_count"]}:
        raise ValueError(f"{path.parent.name} has an incomplete camera stream")
    return {"source": info["scene_group_id"], "dataset_sha256": info["sha256"],
            "frames_per_camera": info["sample_count"], "distinct_rgb_by_camera": counts}


def report(raw, season, reference):
    splits = {name: [frame_counts(path) for path in sorted((raw / name).glob("so101-*/manifest.json"))]
              for name in GOALS}
    if any(len(splits[name]) > GOALS[name] for name in GOALS):
        raise ValueError("The fixed generation target and the retained split disagree")
    rows = [item for records in splits.values() for item in records]
    sizes = {row["frames_per_camera"] for row in rows}
    if len(sizes) != 1:
        raise ValueError("The upper bound uses a single verified teacher duration")
    samples, = sizes
    by_camera = {sensor: sum(row["distinct_rgb_by_camera"][sensor] for row in rows) for sensor in ("wrist", "top")}
    unique = sum(by_camera.values())
    frames = len(rows) * 2 * samples
    ratio = min(by_camera.values()) / (frames / 2)
    remaining = {name: GOALS[name] - len(splits[name]) for name in GOALS}
    left = season["wall_cap_s"] - season["generation_elapsed_s"]
    missing = sum(remaining.values())
    return {"scope": "Preserved raw train and validation only", "accepted_by_split": {k: len(v) for k, v in splits.items()},
            "remaining_by_split": remaining, "remaining_generation_wall_seconds": left, "recorded_images": frames,
            "distinct_images_counting_each_episode_and_camera_separately": unique,
            "distinct_fraction_by_camera": {key: value / (frames / 2) for key, value in by_camera.items()},
            "best_case_reuse_speedup_if_each_camera_has_fixed_cost": 1 / ratio,
            "allowed_wall_seconds_per_remaining_episode": left / missing if missing else None,
            "allowed_wall_seconds_per_remaining_pair_including_all_overhead": left / (missing * samples) if missing else None,
            "archived_reference_seconds_per_pair": reference,
            "optimistic_reference_wall_per_rollout_if_repeats_are_free_and_each_camera_has_fixed_cost":
                reference * ratio * samples,
            "per_attempt_wall_cap_seconds": season["rollout_timeout_s"], "frames": splits}


def evaluation_projection(reference, fit):
    remaining = {"act": 7200 - fit["cumulative_fit_wall_s"], "smolvla": 7200}
    validation_trials = 20
    fresh_trials = 20
    pairs_per_trial = 510
    models = {}
    for kind, candidates in (("act", 2), ("smolvla", 1)):
        stage_available = remaining[kind]
        training_cap = min(3900, int(stage_available - 1200))
        pairs = (validation_trials * candidates + fresh_trials) * pairs_per_trial
        models[kind] = {"validation_checkpoint_candidates": candidates, "evaluation_camera_pairs": pairs,
                        "remaining_end_to_end_cap_s": stage_available, "allowed_optimizer_wall_cap_s": training_cap,
                        "projected_render_only_wall_s_at_reference_median": reference * pairs,
                        "available_mean_seconds_per_pair_if_optimizer_uses_cap": (stage_available - training_cap) / pairs}
    return {"scope": "Projection for the current full evaluator from measured stock CPU renderer timing",
            "reference_median_pair_seconds": reference, "validation_resets_per_checkpoint": validation_trials,
            "fresh_resets_for_selected_checkpoint": fresh_trials, "maximum_camera_pairs_per_reset": pairs_per_trial,
            "excludes": ["processor time", "policy inference", "physics", "I/O", "load time"], "models": models}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Choose a new report path")
    budget = PILOT / "motor-full-capture.json"
    baseline = PILOT / "mesa-motor-render-profile.json"
    fit = PILOT / "act-clock-fit.json"
    seconds = load(baseline)["results"]["default"]["median_two_camera_pair_s"]
    result = report(RAW, load(budget), seconds)
    hashes = {"generation_budget_sha256": sha(budget), "reference_measurement_sha256": sha(baseline),
              "clock_fit_sha256": sha(fit), "motor_comparison_source_sha256": sha(PILOT.parents[1] / "src/smart_swarm/motor_full.py")}
    atomic(args.output, {**result, **hashes, "current_full_evaluator_projection": evaluation_projection(seconds, load(fit))})
    print(args.output)


if __name__ == "__main__":
    main()
