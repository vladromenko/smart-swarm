#!/usr/bin/env python3
"""Verify the actual development RGB before opening independent mobile inspection cases."""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

import rehearse_rosmaster_mobility as rehearsal
from smart_swarm.contract import json_bytes
from smart_swarm.episode import load, sha
from smart_swarm.inspection_comparison import LOCATIONS, MAX_DRIFT_M, MAX_TRACKING_ERROR_DEG
from smart_swarm.inspection_view import combine, interpret

STATION = "west_surface"
EXPOSURES = ("carrier.png", "surface.png")
SCENE_POSITIONS = {"visible-control": ("left", "right"), "hidden-contained": ("left", "right"),
                   "hidden-surface": ("surface",)}


def verify_views(directory, branch):
    frames = branch["policy_observations"]
    if tuple(frame["frame"] for frame in frames) != EXPOSURES:
        raise ValueError("Both original wrist image exposures are required")
    readouts, times, hashes = [], [], []
    for frame in frames:
        file = (directory / frame["frame"]).resolve()
        if not file.is_relative_to(directory.resolve()) or sha(file) != frame["sha256"]:
            raise ValueError("A saved development RGB frame changed")
        with Image.open(file) as image:
            if image.format != "PNG" or image.mode != "RGB" or image.size != (320, 240):
                raise ValueError("The saved inspection frame differs from the native RGB format")
            readout = interpret(np.asarray(image), frame["calibration"])
        if json_bytes(readout) != json_bytes(frame["readout"]):
            raise ValueError("The saved private RGB readout cannot be reproduced")
        readouts.append(readout)
        times.append(frame["at_ns"])
        hashes.append(frame["sha256"])
    if not branch["start_ns"] < times[0] < times[1] == branch["stop_ns"]:
        raise ValueError("The two RGB frames need fresh, ordered exposures")
    answer = combine(readouts)
    if answer != branch["localized"]:
        raise ValueError("The saved combined RGB decision cannot be reproduced")
    return {"at_ns": times, "rgb_sha256": hashes, "decision": answer}


def verify_route(report, waypoint):
    active, idle = (report[key] for key in ("active", "stationary"))
    active_motion = active["motion"]
    arm = active_motion["arm"]
    targets = [rehearsal.SCAN[0][1], rehearsal.HOME_DEG, waypoint["home_joints_deg"],
               waypoint["waypoint_deg"], waypoint["goal_joints_deg"]]
    if len(arm) != len(targets) or any(row["requested_deg"] != list(target)
                                       or not 0 <= row["max_error_deg"] <= MAX_TRACKING_ERROR_DEG
                                       for row, target in zip(arm, targets, strict=True)):
        raise ValueError("The inspection arm did not execute the fixed waypoint route within its limit")
    drive = active_motion["drive"]
    if (drive["requested_xy_m"] != waypoint["base_translation_xy_m"]
            or not 0 <= drive["tracking_error_m"] <= .005 or not 0 <= drive["max_measured_speed_m_s"] <= .13):
        raise ValueError("The inspection chassis did not reach the fixed station within its limits")
    if idle["motion"] is not None or any(row[key] for row in (active, idle)
                                        for key in ("external_contact_pairs", "self_contact_pairs")):
        raise ValueError("Both matched inspection branches must preserve the scene without contacts")
    comparison = rehearsal.compare(active, idle)
    duration = active["stop_ns"] - active["start_ns"]
    steps = round(duration / (rehearsal.STEP * 1e9))
    drift = [row["evaluator"][key] for row in (active, idle)
             for key in ("cube_displacement_m", "carrier_displacement_m")]
    if (comparison != report["comparison"] or not all(0 <= value <= MAX_DRIFT_M for value in drift)
            or not duration > 0 or any(row["checked_physics_steps"] != steps for row in (active, idle))):
        raise ValueError("The saved matched contact and exposure evidence changed")
    return comparison


def verify_identity(report, variant, waypoint):
    if report["status"] != "complete":
        raise ValueError(f"The development rehearsal failed: {report.get('reason')}")
    if not report["rendered_rgb"] or not report["reservation"]:
        raise ValueError("The geometry-only rehearsal acquired zero RGB frames")
    if (report["variant"] != variant or report["family"] != rehearsal.FAMILY[variant]
            or report["seed"] != rehearsal.SEED or report["inspection_station"] != STATION
            or report["physical_motion_requested"] is not False
            or report["implementation_sha256"] != sha(Path(rehearsal.__file__))
            or report["waypoint_sha256"] != waypoint["source_sha256"]):
        raise ValueError("The development result used a different renderer path or scene recipe")


def read_case(directory, variant, waypoint):
    receipt = directory / "rehearsal.json"
    report = load(receipt)
    verify_identity(report, variant, waypoint)
    comparison = verify_route(report, waypoint)
    active, idle = (verify_views(directory / arm, report[arm]) for arm in ("active", "stationary"))
    location = comparison["same_location"]
    verify_task(report["teacher_contact_gate"], variant, location)
    if active["decision"]["answer"] != location:
        raise ValueError(f"Fresh RGB returned {active['decision']['answer']} for the {variant} control at {location}")
    return {"variant": variant, "receipt_sha256": sha(receipt), "active": active, "stationary": idle,
            "evaluator_location": location, "matched_virtual_duration_s": comparison["same_duration_s"]}


def verify_task(teacher, variant, location):
    contained = variant != "hidden-surface"
    expected_relation = "carrier" if contained else "surface"
    sustained_carry = teacher["carry_contact_samples"] >= 5 if contained else teacher["carry_contact_samples"] == 0
    if (location not in LOCATIONS or location not in SCENE_POSITIONS[variant]
            or teacher["first_insertion"] is not contained or not sustained_carry
            or teacher["final_relation"] != expected_relation or not teacher["displacement_m"] >= .12):
        raise ValueError("The physical state did not preserve the contact-verified development contrast")


def audit(root, waypoint_file):
    waypoint = rehearsal.read_waypoint(waypoint_file, STATION)
    accepted, refused = {}, {}
    for variant in rehearsal.FAMILY:
        try:
            accepted[variant] = read_case(root / variant, variant, waypoint)
        except (OSError, KeyError, TypeError, ValueError) as error:
            refused[variant] = f"{type(error).__name__}: {error}"
    return {"ready": not refused, "stage": "matched development RGB and route admission",
            "results": accepted, "refused": refused, "physical_motion_requested": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rehearsals", type=Path, required=True)
    parser.add_argument("--waypoint", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.rehearsals, args.waypoint)
    print(json.dumps(result, indent=2, sort_keys=True))
    raise SystemExit(0 if result["ready"] else 2)


if __name__ == "__main__":
    main()
