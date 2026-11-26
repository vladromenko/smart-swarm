#!/usr/bin/env python3
"""Rehearse fresh inspection branches with the existing simulated ROSMASTER wrist camera."""

import argparse
from contextlib import closing
from pathlib import Path

import numpy as np
from PIL import Image

from smart_swarm.cli import settings
from smart_swarm.episode import atomic, sha
from smart_swarm.evidence_inputs import runtime_signature
from smart_swarm.inspection_view import combine, interpret, mounted_calibration
from smart_swarm.rendering import configure_renderer
from smart_swarm.research_jobs import admitted

CAMERA = "ros/front_cam"
SEED = 416_031
PERIOD_S = 0.05
MAX_SPEED_DEG_S = 35
SETTLE_S = 0.5
SCAN = (
    ("containers", (88.56, -13.98, -28.02, -60.86)),
    ("surface", (-43.02, -0.26, 9.14, 90.0)),
)


def aim(world, degrees):
    from smart_swarm.physics import STEP

    indices = world.ros_actuators[:4]
    limits = world.model.actuator_ctrlrange[indices]
    desired = np.deg2rad(degrees).clip(*limits.T)
    maximum = np.deg2rad(MAX_SPEED_DEG_S) * PERIOD_S
    steps = round(PERIOD_S / STEP)
    requested = world.data.ctrl[indices].copy()
    ticks = int(np.ceil(np.max(np.abs(requested - desired)) / maximum))
    for _ in range(ticks):
        requested += (desired - requested).clip(-maximum, maximum)
        world.data.ctrl[indices] = requested
        world.step(steps)
    world.step(round(SETTLE_S / STEP))
    measured = world.data.qpos[world.ros_joints[:4]]
    return {"requested_deg": np.rad2deg(desired).tolist(), "measured_deg": np.rad2deg(measured).tolist(),
            "max_tracking_error_deg": float(np.max(np.rad2deg(np.abs(measured - desired)))),
            "speed_limit_deg_s": MAX_SPEED_DEG_S, "elapsed_s": (ticks * steps + round(SETTLE_S / STEP)) * STEP}


def check_reset(world, snapshot):
    world.restore(snapshot)
    restored = world.snapshot()
    arrays = ("integration", "pending", "filtered", "tank_target")
    if (any(not np.array_equal(getattr(restored, key), getattr(snapshot, key)) for key in arrays)
            or any(not np.array_equal(a, b) for a, b in zip(restored.camera_pose, snapshot.camera_pose, strict=True))
            or restored.queue != snapshot.queue or restored.controls != snapshot.controls
            or restored.rng != snapshot.rng or restored.physics_steps != snapshot.physics_steps):
        raise ValueError("The matched inspection branch could not restore its decision snapshot")


def capture_view(world, directory, label, camera):
    rgb = world.image(camera)
    calibration = mounted_calibration(world, camera)
    readout = interpret(rgb, calibration)
    file = directory / f"{label}-rgb.png"
    Image.fromarray(rgb).save(file)
    return {"frame": file.name, "sha256": sha(file), "camera": camera,
            "at_ns": world.ns, "calibration": calibration, "readout": readout}


def count_visible_target(world, camera):
    import mujoco

    renderer = world.renderer
    renderer.enable_segmentation_rendering()
    try:
        renderer.update_scene(world.data, camera=camera)
        segmentation = renderer.render()
    finally:
        renderer.disable_segmentation_rendering()
    target_id = world.model.geom(f"{world.truth_target}_geom").id
    kind = mujoco.mjtObj.mjOBJ_GEOM
    return int(np.count_nonzero((segmentation[..., 0] == target_id) & (segmentation[..., 1] == kind)))


def evaluator_state(world):
    from smart_swarm.evidence_rollouts import settled_truth

    state = settled_truth(world)
    carrier = float(world.data.body("carrier").xpos[0])
    decoy = float(world.data.body("decoy").xpos[0])
    contained = "left" if carrier < decoy else "right"
    return {**state, "terminal_location": contained if state["inside_carrier"] else state["cube_relation"]}


def hold(world, duration_ns):
    from smart_swarm.physics import STEP

    steps = round(duration_ns / (STEP * 1e9))
    for offset in range(0, steps, 2000):
        world.step(min(2000, steps - offset))


def branch(world, snapshot, directory, inspect, *, idle_ns=None):
    check_reset(world, snapshot)
    directory.mkdir()
    start = world.ns
    before = evaluator_state(world)
    active, visibility = [], []
    if inspect:
        for label, goal in SCAN:
            motion = aim(world, goal)
            view = capture_view(world, directory, label, CAMERA)
            active.append({"motion": motion, "private_rgb": view})
            visibility.append({"at_ns": view["at_ns"], "target_pixels": count_visible_target(world, CAMERA)})
    else:
        hold(world, idle_ns)
    prediction = combine([item["private_rgb"]["readout"] for item in active]) if inspect else None
    after = evaluator_state(world)
    return {"reset_exact": True, "start_ns": start, "stop_ns": world.ns, "active_scan": active,
            "post_inspection_decision": prediction, "evaluator_target_visibility": visibility,
            "evaluator_start_location": before["terminal_location"], "evaluator_end_location": after["terminal_location"],
            "start_relation": before["cube_relation"], "end_relation": after["cube_relation"],
            "target_displacement_m": float(np.linalg.norm(np.subtract(before["cube_xyz_m"], after["cube_xyz_m"]))),
            "retrieval_attempted": False, "assistance": []}


def sealed_decision(story, pickup):
    from smart_swarm.evidence_rollouts import reveal

    try:
        story.scene(pickup)
        query, decision, report = story.ask()
        snapshot = story.world.snapshot()
        finish, reveal_record, _ = reveal(story, decision)
        story.recorder.meta.update(status="valid", experiment_role="development_active_inspection_geometry")
        story.recorder.close(query, decision, finish)
        return snapshot, report["query_all"]["decision"], reveal_record["ground_truth"]
    except (Exception, KeyboardInterrupt) as error:
        story.recorder.quarantine(str(error) or type(error).__name__)
        raise


def evaluate_inspection(fresh, stationary):
    observed = fresh["post_inspection_decision"]["answer"]
    start = fresh["evaluator_start_location"]
    stable = fresh["evaluator_end_location"] == stationary["evaluator_end_location"] == start
    evaluated = (observed == start) if stable and observed != "unresolved" else None
    return {"same_endpoint_location": stable, "fresh_frames": len(fresh["active_scan"]),
            "admitted_target_color_cue": fresh["post_inspection_decision"]["visible_target_cue"],
            "evaluator_target_visible": any(row["target_pixels"] > 0 for row in fresh["evaluator_target_visibility"]),
            "answers_location": observed != "unresolved", "correct_location_if_answered": evaluated,
            "retrieval_attempted": False, "actual_retrieval_success": None}


def run_case(config, folder, family, variant):
    from smart_swarm.evidence_rollouts import EvidenceStory, prepare_world

    world, _, pickup = prepare_world(config["asset_root"], SEED, family, variant=variant)
    folder.mkdir()
    with closing(world):
        story = EvidenceStory(folder / "pre-inspection", family, world, SEED, variant == "hidden-surface")
        snapshot, decision, expected = sealed_decision(story, pickup)
        fresh = branch(world, snapshot, folder / "arm-camera-scan", True)
        elapsed = fresh["stop_ns"] - fresh["start_ns"]
        control = branch(world, snapshot, folder / "stationary", False, idle_ns=elapsed)
        if control["stop_ns"] != fresh["stop_ns"] or fresh["evaluator_start_location"] != expected:
            raise ValueError("The branch clock or start location differs from the original decision snapshot")
        return {"family": family, "variant": variant, "decision": decision, "sealed_evaluator_outcome": expected,
                "contact_gate": story.recorder.meta["physical_story_gate"],
                "selective_rule_would_inspect": decision == "request_inspection",
                "stationary_branch": control, "inspection_branch": fresh,
                "inspection_evaluation": evaluate_inspection(fresh, control)}


def paired_scope(directory, cases):
    pair = [case for case in cases if case["family"] == "F6"]
    signatures = [runtime_signature(directory / case["variant"] / "pre-inspection") for case in pair]
    same = signatures[0] == signatures[1]
    decisions = [case["decision"] for case in pair]
    outcomes = [case["sealed_evaluator_outcome"] for case in pair]
    return {"pre_inspection_permitted_input_match": same, "same_pre_inspection_rule_decision": len(set(decisions)) == 1,
            "different_sealed_outcomes": len(set(outcomes)) == 2, "f6_decisions": decisions,
            "f6_sealed_outcomes": outcomes}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = settings()
    configure_renderer(config)
    with admitted(config, "smart-swarm-rosmaster-inspection-probe") as (grant, heartbeat):
        args.output.mkdir(parents=True, exist_ok=False)
        cases = []
        recipes = (("F1", "visible-control"), ("F6", "hidden-contained"), ("F6", "hidden-surface"))
        for family, variant in recipes:
            heartbeat()
            cases.append(run_case(config, args.output / variant, family, variant))
            complete = len(cases) == len(recipes)
            scope = paired_scope(args.output, cases) if complete else None
            atomic(args.output / "probe.json", {"scope": "development only", "seed": SEED, "cases": cases,
                   "complete": complete, "f6_scope": scope, "reserved_primary_test_sources_used": 0,
                   "physical_motion_requested": False, "renderer_reservation": grant, "implementation_sha256": sha(Path(__file__))})
    print(args.output / "probe.json")


if __name__ == "__main__":
    main()
