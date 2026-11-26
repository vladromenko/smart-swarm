#!/usr/bin/env python3
"""Localize the frozen ACT closure forecast with paired edits to archived wrist pixels."""

import argparse
from hashlib import sha256
from itertools import product
from pathlib import Path

import numpy as np
import torch

from audit_act_closure import CUE_ISSUES, FIT, PARITY_LIMIT, PILOT, checkpoint_visual_binding, decode, payload
from export_lerobot import KEYS, load_episode, make_frame
from policy_worker import load_policy
from smart_swarm.episode import atomic, load, sha
from smart_swarm.motor_input import CLOCKED_PROFILE

REFERENCE = PILOT / "act-clock-closure-window.json"
EXPECTED_REFERENCE = "7562a4112b80aa6f14a9b238346238faeb9ac4f3e5a48ca4fb777e47228e2339"
REGIONS = ("cube_without_jaw_overlap", "jaw_without_cube_overlap", "cube_and_jaw_overlap", "remaining_pixels")
PALETTE = {"cube": "r >= 95, r > 2g, r > 2b", "jaw": "r >= 75, g >= 55, 3r <= 5g, g <= r, 5b < 3g"}


def colors(image):
    red, green, blue = image.astype(np.int16).transpose(2, 0, 1)
    cube = (red >= 95) & (red > 2 * green) & (red > 2 * blue)
    jaw = (red >= 75) & (green >= 55) & (5 * green >= 3 * red) & (green <= red) & (5 * blue < 3 * green)
    return cube, jaw


def masks_for(early, later):
    first_cube, first_jaw = colors(early)
    last_cube, last_jaw = colors(later)
    cube = first_cube | last_cube
    jaw = first_jaw | last_jaw
    within_cube = cube & ~jaw
    regions = dict(zip(REGIONS, (within_cube, jaw & ~cube, cube & jaw, ~(cube | jaw)), strict=True))
    cube_transitions = {"new_cube_pixels": within_cube & ~first_cube & last_cube,
                        "departed_cube_pixels": within_cube & first_cube & ~last_cube,
                        "still_cube_pixels": within_cube & first_cube & last_cube}
    for partition, expected in ((regions, np.ones_like(cube)), (cube_transitions, within_cube)):
        if not np.array_equal(np.sum(list(partition.values()), axis=0), expected.astype(int)):
            raise ValueError("The color regions overlap or omit pixels from their declared domain")
    return regions, cube_transitions


def combined(edited, masks, shape):
    mask = np.zeros(shape, dtype=bool)
    for name in edited:
        mask |= masks[name]
    return mask


def swap_cases(early, later, frame, masks, native):
    requests, records = [], []
    for selection in product((False, True), repeat=len(masks)):
        edited = [name for name, enabled in zip(masks, selection, strict=True) if enabled]
        mask = combined(edited, masks, early.shape[:2])
        image = np.where(mask[..., None], later, early)
        requests.append(payload({**frame, KEYS["wrist"]: image}))
        records.append({"later_regions": edited, "wrist_sha256": sha256(image.tobytes()).hexdigest(),
                        "changed_pixels": int(np.count_nonzero(np.any(image != early, axis=-1)))})
    outputs = decode(requests, *native)
    return [{**record, "first_gripper_target_percent": float(chunk[0, -1]),
             "issued_first_six_gripper_targets_percent": chunk[:6, -1].tolist()}
            for record, chunk in zip(records, outputs, strict=True)]


def endpoint_check(reference, results, early, later):
    expected = []
    for issue in CUE_ISSUES:
        expected.append(next(row["first_native_gripper_target_percent"] for row in reference
                             if row["wrist"] == issue and all(row[name] == CUE_ISSUES[0] for name in ("top", "state", "clock"))))
    for goal, row, image in zip(expected, (results[0], results[-1]), (early, later), strict=True):
        if row["wrist_sha256"] != sha256(image.tobytes()).hexdigest() or abs(row["first_gripper_target_percent"] - goal) > PARITY_LIMIT:
            raise ValueError("Wrist-only endpoints disagree with the original component audit")


def geometry_summary(early, later, masks):
    difference = np.max(np.abs(early.astype(np.int16) - later), axis=-1)
    counts = {name: {"pixels": int(mask.sum()), "pixels_with_rgb_change_at_least_20": int((mask & (difference >= 20)).sum())}
              for name, mask in masks.items()}
    centroids = []
    for image in (early, later):
        y, x = np.nonzero(colors(image)[0])
        if len(x) < 1000:
            raise ValueError("The original scene has too few distinguishable cube-color pixels")
        centroids.append([float(x.mean()), float(y.mean())])
    return {"region_sizes": counts, "cube_color_centroid_xy": centroids}


def source_report(entry, reference, native):
    root = Path(entry["raw_path"])
    meta = load(root / "manifest.json")
    if (meta["sha256"] != reference["source_sha256"] or meta["sha256"] != entry["source_sha256"]
            or meta["seed"] != reference["seed"] or not meta["physics"]["released_stable_containment"]):
        raise ValueError("The stock teacher source changed from the closure audit")
    data = load_episode(root, meta, aligned=True)
    early, later = data["wrist"][list(CUE_ISSUES)]
    frame = make_frame(data, meta, CUE_ISSUES[0], CLOCKED_PROFILE)
    regions, cube_transitions = masks_for(early, later)
    swaps = swap_cases(early, later, frame, regions, native)
    cube_swaps = swap_cases(early, later, frame, cube_transitions, native)
    endpoint_check(reference["teacher_input_component_swaps"], swaps, early, later)
    full_cube = next(row for row in swaps if row["later_regions"] == [REGIONS[0]])
    if full_cube["wrist_sha256"] != cube_swaps[-1]["wrist_sha256"]:
        raise ValueError("Separating the cube changes did not retain the same composite cube-only input")
    return {"source_sha256": meta["sha256"], "seed": meta["seed"], "split": entry["source_split"],
            "region_analysis": geometry_summary(early, later, {**regions, **cube_transitions}),
            "region_swaps": swaps, "within_cube_swaps": cube_swaps}


def audit():
    torch.set_num_threads(1)
    fit, previous = load(FIT), load(REFERENCE)
    checkpoint, dataset = (Path(fit[key]) for key in ("checkpoint", "dataset"))
    if sha(REFERENCE) != EXPECTED_REFERENCE or sha(FIT) != previous["fit_receipt_sha256"]:
        raise ValueError("The original aligned closure receipt changed")
    if any(sha(checkpoint / name) != checksum for name, checksum in previous["artifact_sha256"].items()):
        raise ValueError("The frozen ACT checkpoint or processor artifacts changed")
    visual = checkpoint_visual_binding(checkpoint, dataset)
    config, model, pre, post = load_policy(checkpoint)
    if config.type != "act" or config.n_action_steps != 6:
        raise ValueError("The archived ACT control queue changed")
    cohorts = {row["seed"]: row for row in previous["sources"]}
    sources = []
    for split in ("train", "validation"):
        manifest = dataset / split / "smart-swarm-lineage.json"
        if sha(manifest) != previous["teacher_lineage_sha256_by_split"][split]:
            raise ValueError("The source grouping changed")
        for entry in load(manifest)["episodes"]:
            seed = int(entry["source_world_episode_id"].rsplit("-", 1)[1])
            sources.append(source_report(entry, cohorts[seed], (model, pre, post)))
    return {"scope": "Exploratory archived teacher wrist edits, frozen ACT", **visual,
            "previous_closure_audit_sha256": EXPECTED_REFERENCE, "checkpoint_sha256": previous["artifact_sha256"]["model.safetensors"],
            "held_fixed_issue": CUE_ISSUES[0], "late_pixel_source_issue": CUE_ISSUES[1],
            "palette_masks": PALETTE, "sources": sources, "new_optimizer_updates": 0, "new_closed_loop_trials": 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Choose a new wrist region audit path")
    atomic(args.output, audit())
    print(args.output)


if __name__ == "__main__":
    main()
