#!/usr/bin/env python3
"""Audit the archived optical development probe against public fixture geometry."""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

from smart_swarm.episode import atomic, load, sha

POCKET = np.array([.044, -.258, .252])
SCREEN_WALLS = {
    "south": ((-.020, -.312, .220), (.108, -.304, .308)),
    "north": ((-.020, -.212, .220), (.108, -.204, .308)),
    "east": ((.100, -.308, .220), (.108, -.208, .308)),
}


def obstacles(origin):
    origin = np.asarray(origin)
    direction = POCKET - origin
    moving = direction != 0
    intersections = []
    for wall, ends in SCREEN_WALLS.items():
        lower, upper = np.asarray(ends)
        if moving.any() and np.all((lower[~moving] <= origin[~moving]) & (origin[~moving] <= upper[~moving])):
            values = np.sort((np.asarray(ends)[:, moving] - origin[moving]) / direction[moving], axis=0)
            enter = max(0., float(values[0].max()))
            exit_ = min(1., float(values[1].min()))
            if enter < exit_:
                height = origin[2] + enter * direction[2]
                intersections.append({"wall": wall, "line_entry_height_m": height,
                                      "entry_below_wall_top_m": ends[1][2] - height})
    return intersections


def frame_result(root, case, sample, visibility):
    camera = sample["private_rgb"]
    folder = root / case["variant"] / "arm-camera-scan"
    path = (folder / camera["frame"]).resolve()
    if not path.is_relative_to(folder) or sha(path) != camera["sha256"] or visibility["at_ns"] != camera["at_ns"]:
        raise ValueError("The archived image or exposure identity changed")
    with Image.open(path) as image:
        r, g, b = np.asarray(image.convert("RGB"), dtype=np.int16).transpose(2, 0, 1)
    mask = (r > 140) & (r - g > 77) & (g - b > 10)
    count = int(mask.sum())
    if camera["readout"]["target_color_detected"] != (count >= 7):
        raise ValueError("The RGB support no longer reproduces the saved readout")
    return {"frame_path": str(path), "frame_sha256": camera["sha256"], "rgb_orange_mask_pixels": count,
            "minimum_pixels_for_readout": 7, "evaluator_target_pixels": visibility["target_pixels"],
            "camera_origin_m": camera["calibration"]["origin_m"], "locally_answered": camera["readout"]["answer"],
            "public_pocket_center_line_occlusion": obstacles(camera["calibration"]["origin_m"])
            if case["family"] == "F6" else None}


def audit(probe):
    data = load(probe)
    if not data["complete"] or not data["f6_scope"]["pre_inspection_permitted_input_match"]:
        raise ValueError("A complete development probe with matched F6 inputs is required")
    cases = []
    for case in data["cases"]:
        scan = case["inspection_branch"]
        images = [frame_result(probe.parent, case, sample, view)
                  for sample, view in zip(scan["active_scan"], scan["evaluator_target_visibility"], strict=True)]
        cases.append({"family": case["family"], "variant": case["variant"], "sealed_outcome": case["sealed_evaluator_outcome"],
                      "initial_decision": case["decision"], "frames": images})
    return {"scope": "development only, archived optical evidence and a nominal public pocket center line",
            "probe_sha256": sha(probe), "source_seed": data["seed"], "independent_sources": 1,
            "cases": cases, "reserved_primary_and_active_test_sources_used": 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Choose a new optics audit path")
    atomic(args.output, audit(args.probe.resolve()))
    print(args.output)


if __name__ == "__main__":
    main()
