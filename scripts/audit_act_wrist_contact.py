#!/usr/bin/env python3
"""Reproduce the teacher contacts at the two archived ACT wrist observations."""

import argparse
from pathlib import Path

import numpy as np

from smart_swarm.arm_executor import ArmExecutor
from smart_swarm.cli import ROOT, settings
from smart_swarm.episode import atomic, load, sha
from smart_swarm.motor_diagnostics import read_teacher, reset_world
from smart_swarm.motor_input import verify_issue_time
from smart_swarm.teacher import Teacher, run_insertion, truth

PILOT = ROOT / "evidence/learning-pilot-v3"
REFERENCE = PILOT / "act-wrist-region-attribution.json"
FIT = PILOT / "act-clock-fit.json"
STATE_TOLERANCE = .0001
WINDOW = range(114, 151)


class WindowObserved(Exception):
    """The original teacher has reached the last issue needed by this audit."""


def projected_cube(cube, pose):
    rotation = np.asarray(pose["wrist_camera_orientation"]).reshape(3, 3)
    relative = rotation.T @ (np.asarray(cube) - pose["wrist_xyz_m"])
    width, height = pose["wrist_frame_width_height"]
    focal = height / (2 * np.tan(np.deg2rad(pose["vertical_fov_degrees"]) / 2))
    if relative[2] >= 0:
        raise ValueError("The cube moved behind the wrist camera in the selected replay window")
    return (np.asarray([width / 2, height / 2]) + (focal / -relative[2]) * relative[:2] * [1, -1]).tolist()


def observe(world, issue, camera):
    seen = truth(world)
    mount = world.data.camera(camera)
    pose = {"wrist_camera_orientation": mount.xmat.tolist(), "wrist_xyz_m": mount.xpos.tolist(),
            "wrist_frame_width_height": list(world.frame_size), "vertical_fov_degrees": float(world.model.camera(camera).fovy[0])}
    return {"issue": issue, "at_ns": world.ns, "cube_xyz_m": seen["cube_xyz_m"],
            "cube_camera_center_px": projected_cube(seen["cube_xyz_m"], pose), **pose,
            "cube_speed_m_s": seen["cube_speed_m_s"], "contact_bodies": seen["contact_bodies"],
            "so101_contact_bodies": seen["gripper_contact_bodies"]}


def recorded_command_prefix(world, arrays):
    engine = ArmExecutor(world, "aligned-sample-v2")
    engine.prime()
    camera = world.cameras["so101-1"]["so101-1/wrist"]
    feedback, residuals = [], []
    for issue in range(WINDOW.stop):
        state = world.state("so101-1")[0]
        residuals.append(float(np.max(np.abs(np.asarray(state) - arrays["state"][issue]))))
        if engine.observation_index() != issue or world.ns != int(arrays["timestamps_ns"][issue]):
            raise ValueError("Physics replay no longer matches the original command clock")
        if issue in WINDOW:
            feedback.append(observe(world, issue, camera) | {"teacher_gripper_measured_percent": state[-1],
                                                            "teacher_gripper_target_percent": float(arrays["requested"][issue, -1])})
        engine.submit(arrays["requested"][issue])
    return feedback, max(residuals), world.physics_steps


def original_teacher_prefix(world, arrays, pickup):
    camera = world.cameras["so101-1"]["so101-1/wrist"]
    feedback, residuals = [], []

    def sample(phase, _index, interval, asked, applied):
        issue = len(residuals)
        state = world.state("so101-1")[0]
        residuals.append(float(np.max(np.abs(np.asarray(state) - arrays["state"][issue]))))
        if (world.ns != int(arrays["timestamps_ns"][issue]) or interval != int(arrays["applied_interval_start_ns"][issue])
                or not np.array_equal(np.asarray(asked, dtype=np.float32), arrays["requested"][issue])
                or not np.array_equal(np.asarray(applied, dtype=np.float32), arrays["applied_previous"][issue])):
            raise ValueError(f"The original teacher no longer reproduces the archived controls at issue {issue}")
        if issue in WINDOW:
            feedback.append(observe(world, issue, camera) | {"teacher_gripper_measured_percent": state[-1],
                                                            "teacher_gripper_target_percent": asked[-1], "teacher_phase": phase})
        if issue + 1 == WINDOW.stop:
            raise WindowObserved

    try:
        run_insertion(Teacher(world), sample, pickup=pickup)
    except WindowObserved:
        pass
    if world.renderer is not None or len(residuals) != WINDOW.stop or max(residuals) > STATE_TOLERANCE:
        raise ValueError(f"The reconstructed teacher's no-render state differs by {max(residuals):.8f}")
    return feedback, max(residuals), world.physics_steps


def first_contact(trace, count):
    return next((row["issue"] for row in trace if len(row["so101_contact_bodies"]) >= count), None)


def camera_effect(first, last):
    actual_early = first["cube_camera_center_px"][0]
    actual_late = last["cube_camera_center_px"][0]
    late_view_early_cube = projected_cube(first["cube_xyz_m"], last)[0]
    early_view_late_cube = projected_cube(last["cube_xyz_m"], first)[0]
    return {"actual_horizontal_shift_px": actual_late - actual_early,
            "late_camera_only_horizontal_shift_px": late_view_early_cube - actual_early,
            "late_cube_position_only_horizontal_shift_px": early_view_late_cube - actual_early}


def report_source(asset_root, entry, source, early, later):
    record, arrays, _ = read_teacher(Path(entry["raw_path"]))
    if (record["sha256"] != source["source_sha256"] or record["sha256"] != entry["source_sha256"]
            or record["seed"] != source["seed"]):
        raise ValueError("The original teacher material changed")
    verify_issue_time(arrays["timestamps_ns"], arrays["applied_interval_start_ns"])
    world = reset_world(asset_root, record)
    try:
        camera = world.cameras["so101-1"]["so101-1/wrist"]
        parent = world.model.body(int(world.model.cam_bodyid[world.model.camera(camera).id])).name
        initial = world.snapshot()
        rounded, archived_replay_residual, _ = recorded_command_prefix(world, arrays)
        world.restore(initial)
        trace, residual, steps = original_teacher_prefix(world, arrays, record["physics"]["pickup_xy_m"])
    finally:
        world.close()
    epochs = {row["issue"]: row for row in trace}
    first, last = epochs[early], epochs[later]
    return {"seed": record["seed"], "split": entry["source_split"], "raw_sha256": record["sha256"],
            "contact_window": trace, "first_sampled_contact_in_window": first_contact(trace, 1),
            "first_sampled_bilateral_in_window": first_contact(trace, 2),
            "pre_issue_state_max_abs_residual": residual, "solver_steps": steps,
            "float32_command_replay_state_max_abs_residual": archived_replay_residual,
            "float32_command_replay_first_contact": first_contact(rounded, 1),
            "float32_command_replay_first_bilateral": first_contact(rounded, 2),
            "cube_motion_between_observations_mm": float(1000 * np.linalg.norm(np.subtract(last["cube_xyz_m"], first["cube_xyz_m"]))),
            "wrist_motion_between_observations_mm": float(1000 * np.linalg.norm(np.subtract(last["wrist_xyz_m"], first["wrist_xyz_m"]))),
            "camera_mount_body": parent, "projected_center_intervention": camera_effect(first, last)}


def audit():
    fit, receipt = load(FIT), load(REFERENCE)
    cohorts = {row["seed"]: row for row in receipt["sources"]}
    early, later = receipt["held_fixed_issue"], receipt["late_pixel_source_issue"]
    sources = []
    for split in ("train", "validation"):
        filename = Path(fit["dataset"]) / split / "smart-swarm-lineage.json"
        for entry in load(filename)["episodes"]:
            seed = int(entry["source_world_episode_id"].rsplit("-", 1)[1])
            sources.append(report_source(settings()["asset_root"], entry, cohorts[seed], early, later))
    if len(sources) != 7:
        raise ValueError("The frozen wrist attribution covers seven independent teacher sources")
    return {"scope": "Evaluator only, open-loop physics prefix from original and historical float32 teacher controls",
            "region_receipt_sha256": sha(REFERENCE),
            "first_observation_issue": early, "second_observation_issue": later, "sources": sources,
            "new_rgb_frames": 0, "new_optimizer_updates": 0, "new_learned_rollouts": 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Choose a new wrist contact audit path")
    atomic(args.output, audit())
    print(args.output)


if __name__ == "__main__":
    main()
