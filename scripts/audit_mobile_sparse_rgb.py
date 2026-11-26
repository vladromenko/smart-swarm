#!/usr/bin/env python3
"""Reopen the sealed development images and test conservative sparse orange cues."""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

from smart_swarm.episode import atomic, load, sha
from smart_swarm.inspection_triangulation import MIN_WEAK_CUBE_PIXELS, extent, intersect, locate, pixel_ray
from smart_swarm.inspection_view import CUBE_CENTER_HEIGHT_M, RIM_MARKER_HEIGHT_M, bin_choice, interpret

ARMS = ("scan", "stationary")
FRAMES = ("carrier.png", "surface.png")


def sparse_pixels(rgb):
    r, g, b = rgb.astype(np.int16).transpose(2, 0, 1)
    y, x = np.where((r > 140) & (r - g > 77) & (g - b > 10))
    if not 2 <= x.size < MIN_WEAK_CUBE_PIXELS or max(np.ptp(x), np.ptp(y)) > 4:
        return []
    return list(zip(x, y, strict=True))


def unanimous_container(first, weak, candidates, markers):
    rgb, camera = weak
    bounds = [intersect(first, pixel_ray(pixel, camera, rgb.shape[:2][::-1]), CUBE_CENTER_HEIGHT_M)
              for pixel in candidates]
    if any(point is None for point in bounds):
        return None
    answers = {bin_choice(point, *markers) for point in bounds}
    return next(iter(answers)) if len(answers) == 1 else None


def locate_with_sparse_support(samples):
    original = locate(samples)
    if (original["answer"] != "unresolved" or original["conflicting_localizations"]
            or not original["visible_target_cue"]):
        return original
    offsets = [(index, interpret(rgb, camera)) for index, (rgb, camera) in enumerate(samples)]
    markers = [extent(samples, [report for _, report in offsets], cue, RIM_MARKER_HEIGHT_M)
               for cue in ("cyan", "violet")]
    strong = [(index, report["rgb_marker_centers_px"]["cube"]) for index, report in offsets
              if report["target_color_detected"]]
    if len(strong) != 1 or any(marker is None for marker in markers):
        return original
    index, pixel = strong[0]
    rgb, camera = samples[index]
    weak = samples[1 - index]
    candidates = sparse_pixels(weak[0])
    ray = pixel_ray(pixel, camera, rgb.shape[:2][::-1])
    answer = unanimous_container(ray, weak, candidates, markers) if candidates else None
    if answer is None:
        return original
    return {**original, "answer": answer, "used_triangulation": True, "sparse_epipolar_pixels": len(candidates)}


def stored_pair(directory, branch):
    images = []
    for row, name in zip(branch["samples"], FRAMES, strict=True):
        source = directory / name
        if row["frame"] != name or sha(source) != row["sha256"]:
            raise ValueError(f"Sealed RGB frame has changed at {source}")
        with Image.open(source) as image:
            images.append((np.asarray(image).copy(), row["calibration"]))
    return images


def audit_case(path):
    receipt = load(path / "receipt.json")
    if receipt["status"] != "sealed":
        return {"status": receipt["status"], "reason": receipt.get("reason")}
    sealed = path / "policy.json"
    if sha(sealed) != receipt["policy_sha256"]:
        raise ValueError(f"The original post-inspection policy changed at {path}")
    policy = load(sealed)
    alternatives = {arm: locate_with_sparse_support(stored_pair(path / arm, policy[arm])) for arm in ARMS}
    expected = receipt["evaluation"]["location"]
    return {"status": "evaluated", "sealed_policy_sha256": sha(sealed), "expected": expected,
            "saved_choices": {arm: policy[arm]["post_inspection_decision"] for arm in ARMS},
            "sparse_support_choices": alternatives, "scan_correct": alternatives["scan"]["answer"] == expected}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--development", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Choose a new sparse RGB audit output")
    variants = ("visible-control", "hidden-contained", "hidden-surface")
    paths = {v: args.development / f"mobile-rehearsal-{v}" for v in variants}
    results = {variant: audit_case(path) for variant, path in paths.items()}
    atomic(args.output, {"scope": "sealed optical development evidence", "results": results,
                         "prediction_inputs": ["two native RGB exposures", "own measured camera calibration"],
                         "independent_cases_opened": 0, "deployed": False, "optimizer_updates": 0})
    print(args.output)


if __name__ == "__main__":
    main()
