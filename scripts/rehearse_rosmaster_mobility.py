#!/usr/bin/env python3
"""Exercise the isolated planar ROSMASTER through contact-checked inspection branches."""

import argparse
from collections import defaultdict
from contextlib import closing, nullcontext
from pathlib import Path
from time import monotonic
from types import SimpleNamespace

import numpy as np
from PIL import Image

from audit_rosmaster_inspection_optics import obstacles
from probe_rosmaster_inspection import CAMERA, SCAN, SEED, check_reset, count_visible_target, evaluator_state
from smart_swarm.cli import settings
from smart_swarm.episode import atomic, load, sha
from smart_swarm.evidence_rollouts import physical_checks, settled_truth
from smart_swarm.inspection_mobility import (ARM_SPEED_RAD_S, STATIONS, SceneContactWatch, drive_to, mounted_feedback,
                                             prepare_mobile, slew)
from smart_swarm.inspection_view import combine, interpret
from smart_swarm.physics import STEP
from smart_swarm.rendering import configure_renderer
from smart_swarm.research_jobs import admitted
from smart_swarm.scaling import SLOT_STEPS, WINDOW_SLOTS
from smart_swarm.teacher import Teacher, move_box, run_insertion, truth

FAMILY = {"visible-control": "F1", "hidden-contained": "F6", "hidden-surface": "F6"}
HOME_DEG = (0., 40., -85., -55.)
SURFACE_POSES = {"surface": (75.491262, -17.693978, -29.569158, -89.307325),
                 "west_surface": (54.828946, -56.637978, 21.274819, -80.181543)}


def decision_snapshot(world, pickup, family, variant):
    samples = defaultdict(list)

    def record(phase, index, *_):
        if index % 20 == 19:
            samples[phase].append(truth(world))

    world.tank_target[:] = [.10, 0.]
    world.step(1000)
    world.tank_target[:] = 0.
    world.step(1000)
    run_insertion(Teacher(world), record, pickup=pickup)
    world.step(200)
    first = truth(world)
    settled = settled_truth(world)
    if family == "F6":
        world.step(SLOT_STEPS)
    world.step(WINDOW_SLOTS * SLOT_STEPS)
    move_box(world, lambda: None)
    world.step(200)
    world.step(WINDOW_SLOTS * SLOT_STEPS)
    world.step(180)
    reference = SimpleNamespace(world=world, family=family, miss=variant == "hidden-surface", stage_data=samples)
    return world.snapshot(), physical_checks(reference, first, settled)


def arm_motion(world, watch, target):
    requested = np.deg2rad(target)
    axes = len(requested)
    elapsed = slew(world, watch, world.ros_actuators[:axes], requested, ARM_SPEED_RAD_S, .75)
    measured = world.data.qpos[world.ros_joints[:axes]]
    error = float(np.rad2deg(np.max(np.abs(measured - requested))))
    if error > 1.:
        raise ValueError(f"Inspection arm tracking error reached {error:.3f} degrees")
    return {"requested_deg": list(target), "max_error_deg": error, "elapsed_ns": elapsed}


def observation(world, folder, name):
    camera = mounted_feedback(world, CAMERA)
    result = {"calibration": camera, "at_ns": world.ns}
    evaluator = {"pocket_line_intersections": obstacles(camera["origin_m"])}
    if folder is not None:
        rgb = world.image(CAMERA)
        image = folder / f"{name}.png"
        Image.fromarray(rgb).save(image)
        result.update(frame=image.name, sha256=sha(image), readout=interpret(rgb, camera))
        evaluator["target_pixels"] = count_visible_target(world, CAMERA)
    return result, evaluator


def mobile_scan(world, snapshot, folder, station, waypoint):
    check_reset(world, snapshot)
    events = SceneContactWatch(world)
    before = evaluator_state(world)
    start = world.ns
    motions = [arm_motion(world, events, SCAN[0][1])]
    carrier, carrier_visibility = observation(world, folder, "carrier")
    motions.append(arm_motion(world, events, HOME_DEG))
    if waypoint:
        motions.append(arm_motion(world, events, waypoint["home_joints_deg"]))
    drive = drive_to(world, events, station)
    if waypoint:
        motions.append(arm_motion(world, events, waypoint["waypoint_deg"]))
    target = waypoint["goal_joints_deg"] if waypoint else SURFACE_POSES[station]
    motions.append(arm_motion(world, events, target))
    surface, surface_visibility = observation(world, folder, "surface")
    motion = {"arm": motions, "drive": drive}
    return outcome(world, before, start, events, [carrier, surface], [carrier_visibility, surface_visibility], motion)


def stationary(world, snapshot, schedule, folder):
    check_reset(world, snapshot)
    events = SceneContactWatch(world)
    before = evaluator_state(world)
    start = world.ns
    private, visibility = [], []
    for label, deadline in zip(("carrier", "surface"), schedule, strict=True):
        events.advance(round((deadline - world.ns) / (STEP * 1e9)))
        frame, evaluator = observation(world, folder, label)
        private.append(frame)
        visibility.append(evaluator)
    return outcome(world, before, start, events, private, visibility, None)


def outcome(world, before, start, events, private, visibility, motion):
    after = evaluator_state(world)
    result = {"start_ns": start, "stop_ns": world.ns, "policy_observations": private, "motion": motion,
              "checked_physics_steps": events.active_steps, "external_contact_pairs": sorted(events.pairs),
              "self_contact_pairs": sorted(events.self_pairs), "peak_chassis_speed_m_s": events.peak_speed_m_s}
    evaluator = {"before": before["terminal_location"], "after": after["terminal_location"], "visibility": visibility,
                 "cube_displacement_m": float(np.linalg.norm(np.subtract(before["cube_xyz_m"], after["cube_xyz_m"]))),
                 "carrier_displacement_m": float(np.linalg.norm(np.subtract(before["carrier_xyz_m"], after["carrier_xyz_m"])))}
    readouts = [row["readout"] for row in private if "readout" in row]
    return {**result, "localized": combine(readouts) if readouts else None, "evaluator": evaluator}


def compare(active, control):
    active_clock = [row["at_ns"] for row in active["policy_observations"]]
    control_clock = [row["at_ns"] for row in control["policy_observations"]]
    active_truth, control_truth = active["evaluator"], control["evaluator"]
    locations = {row[phase] for row in (active_truth, control_truth) for phase in ("before", "after")}
    max_drift = max(row[key] for key in ("cube_displacement_m", "carrier_displacement_m")
                    for row in (active_truth, control_truth))
    if active["start_ns"] != control["start_ns"] or active["stop_ns"] != control["stop_ns"] or active_clock != control_clock:
        raise ValueError("The matched mobile and stationary optics have different clocks")
    if len(locations) != 1 or max_drift > .005:
        raise ValueError("Inspection changed the preserved object or carrier relation")
    return {"same_reset_and_exposure_times": True, "same_location": locations.pop(), "max_scene_drift_m": max_drift,
            "same_duration_s": (active["stop_ns"] - active["start_ns"]) / 1e9, "retrieval_attempted": False}


def read_waypoint(path, station):
    if path is None:
        return None
    route = load(path)
    if (station != route["station"] or list(STATIONS[station]) != route["base_translation_xy_m"]
            or list(HOME_DEG) != route["home_joints_deg"][:4] or list(SURFACE_POSES[station]) != route["goal_joints_deg"][:4]
            or not route["waypoint_deg"] or route["endpoint_external_contacts"] or route["fine_samples_per_segment"] < 100):
        raise ValueError("The offline waypoint does not match the requested public inspection station")
    return {**route, "source_sha256": sha(path)}


def exercise(config, variant, output, render, station, waypoint):
    family = FAMILY[variant]
    world, _, pickup = prepare_mobile(config["asset_root"], SEED, family, variant, seconds=100)
    with closing(world):
        world.frame_size = (320, 240)
        source, gate = decision_snapshot(world, pickup, family, variant)
        paths = {name: output / name if render else None for name in ("active", "stationary")}
        if render:
            for path in paths.values():
                path.mkdir()
        active = mobile_scan(world, source, paths["active"], station, waypoint)
        schedule = [row["at_ns"] for row in active["policy_observations"]]
        control = stationary(world, source, schedule, paths["stationary"])
        return {"family": family, "variant": variant, "seed": SEED, "parent_model": world.manifest["parent_model_sha256"],
                "mobile_model": world.manifest["model_sha256"], "scope": "isolated simulation development",
                "rendered_rgb": render, "inspection_station": station,
                "waypoint_sha256": waypoint["source_sha256"] if waypoint else None, "teacher_contact_gate": gate,
                "active": active, "stationary": control,
                "comparison": compare(active, control), "physical_motion_requested": False}


def run(args):
    config = settings()
    args.output.mkdir(parents=True, exist_ok=False)
    receipt = args.output / "rehearsal.json"
    if args.render:
        configure_renderer(config)
    reservation = (admitted(config, "smart-swarm-mobile-inspection-development") if args.render
                   else nullcontext((None, lambda: None)))
    with reservation as (grant, heartbeat):
        started = monotonic()
        try:
            heartbeat()
            waypoint = read_waypoint(args.waypoint, args.station)
            report = exercise(config, args.variant, args.output, args.render, args.station, waypoint)
        except (Exception, KeyboardInterrupt) as error:
            atomic(receipt, {"status": "failed", "reason": str(error), "variant": args.variant,
                             "inspection_station": args.station, "rendered_rgb": args.render,
                             "elapsed_wall_seconds": monotonic() - started})
            raise
        atomic(receipt, {**report, "status": "complete", "reservation": grant,
                         "elapsed_wall_seconds": monotonic() - started, "implementation_sha256": sha(Path(__file__))})
    print(receipt)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=tuple(FAMILY), default="hidden-surface")
    parser.add_argument("--station", choices=tuple(SURFACE_POSES), default="west_surface")
    parser.add_argument("--waypoint", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--render", action="store_true")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
