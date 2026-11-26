#!/usr/bin/env python3
"""Reopen the original mobile development RGB and test two-view depth without privileged inputs."""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

from check_mobile_inspection_rehearsal import EXPOSURES, STATION, verify_identity, verify_route, verify_task, verify_views
from rehearse_rosmaster_mobility import FAMILY, read_waypoint
from smart_swarm.episode import atomic, load, sha
from smart_swarm import inspection_triangulation


def prediction(directory, observations):
    samples = []
    for row, name in zip(observations, EXPOSURES, strict=True):
        if row["frame"] != name or sha(directory / name) != row["sha256"]:
            raise ValueError("The post-inspection RGB reference changed")
        with Image.open(directory / name) as frame:
            samples.append((np.asarray(frame).copy(), row["calibration"]))
    return inspection_triangulation.locate(samples)


def inspect_case(root, variant, route):
    path = root / variant
    receipt = path / "rehearsal.json"
    original = load(receipt)
    verify_identity(original, variant, route)
    views = {name: verify_views(path / name, original[name]) for name in ("active", "stationary")}
    projected = {name: prediction(path / name, original[name]["policy_observations"]) for name in views}
    target = verify_route(original, route)["same_location"]
    verify_task(original["teacher_contact_gate"], variant, target)
    return {"reference": target, "receipt_sha256": sha(receipt), "views": views, "triangulated": projected,
            "elapsed_wall_seconds": original["elapsed_wall_seconds"],
            "active_correct": projected["active"]["answer"] == target,
            "stationary_unresolved": projected["stationary"]["answer"] == "unresolved"}


def audit(root, waypoint):
    route = read_waypoint(waypoint, STATION)
    results, refused = {}, {}
    for variant in FAMILY:
        try:
            row = inspect_case(root, variant, route)
            results[variant] = row
            if not row["active_correct"] or not row["stationary_unresolved"]:
                refused[variant] = "The two exposure branches did not show the expected optical contrast"
        except (OSError, KeyError, TypeError, ValueError) as error:
            refused[variant] = f"{type(error).__name__}: {error}"
    return {"scope": "original simulation development optics", "ready": not refused, "results": results,
            "refused": refused, "implementation_sha256": sha(Path(inspection_triangulation.__file__)),
            "prediction_inputs": ["RGB exposures", "own measured camera calibration"],
            "independent_sources_used": 0, "optimizer_updates": 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rehearsals", type=Path, required=True)
    parser.add_argument("--waypoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Choose a new development audit output")
    result = audit(args.rehearsals, args.waypoint)
    atomic(args.output, result)
    print(args.output)
    raise SystemExit(0 if result["ready"] else 2)


if __name__ == "__main__":
    main()
