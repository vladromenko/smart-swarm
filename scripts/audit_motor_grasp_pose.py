#!/usr/bin/env python3
"""Vary only gripper targets at reconstructed archived ACT poses in MuJoCo."""

import argparse
import math
from pathlib import Path

from smart_swarm.cli import ROOT, settings
from smart_swarm.episode import atomic, load, sha
from smart_swarm.rendering import configure_renderer
from smart_swarm.research_jobs import admitted

PILOT = ROOT / "evidence/learning-pilot-v3"
OBSERVED_TICK = 150
STEPS = 45


def sources(record):
    root = Path(record["dataset"])
    result = {}
    for role in ("train", "validation"):
        lineage = load(root / role / "smart-swarm-lineage.json")
        for item in lineage["episodes"]:
            raw = Path(item["raw_path"])
            manifest = load(raw / "manifest.json")
            result[manifest["seed"]] = raw
    return result


def set_pose(world, native, cube):
    import mujoco

    pose = world.arm_to_mj(native)
    cube_joint = int(world.model.joint("cube_pose").qposadr[0])
    world.data.qpos[world.arm_joints] = pose
    world.data.qpos[cube_joint:cube_joint + 3] = cube
    world.data.qvel[:] = 0
    world.filtered = pose.copy()
    world.pending = pose.copy()
    world.data.ctrl[world.arm_actuators] = pose
    mujoco.mj_forward(world.model, world.data)


def grip_trial(world, native, grip):
    from smart_swarm.arm_executor import ArmExecutor
    from smart_swarm.motor_diagnostics import observe

    command = [*native[:5], grip]
    executive = ArmExecutor(world, "aligned-sample-v2")
    executive.prime()
    observations = []
    for tick in range(STEPS):
        executive.submit(command)
        observation = observe(world, tick + 1)
        observations.append({"tick": tick + 1, "contacts": observation["gripper_contact_bodies"],
                             "height_m": observation["cube_height_above_table_m"],
                             "tcp_to_cube_m": observation["tcp_to_cube_m"]})
    held = [row for row in observations if len(row["contacts"]) >= 2]
    return {"gripper_goal_native_percent": grip, "bilateral_contact_samples": len(held),
            "first_bilateral_contact_tick": held[0]["tick"] if held else None,
            "final_applied_gripper_percent": float(world.arm_from_mj(world.filtered)[-1]),
            "observations": observations}


def one_pose(assets, case, raw):
    from smart_swarm.motor_diagnostics import read_teacher, reset_world

    meta, arrays, _ = read_teacher(raw)
    sampled = next(row for row in case["feedback"] if row["tick"] == OBSERVED_TICK)
    preceding = next(row for row in case["feedback"] if row["tick"] == OBSERVED_TICK - 15)
    drift = math.dist(sampled["cube_xyz_m"], preceding["cube_xyz_m"])
    if drift > .001 or sampled["cube_speed_m_s"] > .01:
        raise ValueError("This archived sample cannot reconstruct a stationary cube")
    target = float(arrays["requested"][OBSERVED_TICK - 1][-1])
    if abs(target - sampled["teacher_requested_at_index"][-1]) > 1e-4:
        raise ValueError("Archived ACT comparison does not match the recorded teacher command")
    native = sampled["previous_observed_state"]
    world = reset_world(assets, meta)
    trial = {}
    try:
        initial = world.snapshot()
        for name, grip in (("archived_act_command", sampled["requested"][-1]), ("teacher_grip_command", target)):
            world.restore(initial)
            set_pose(world, native, sampled["cube_xyz_m"])
            trial[name] = grip_trial(world, native, grip)
    finally:
        world.close()
    return {"seed": case["seed"], "archived_tick": OBSERVED_TICK, "arm_native_pose": native,
            "archived_cube_xyz_m": sampled["cube_xyz_m"], "recorded_teacher_source": str(raw),
            "archived_tcp_to_cube_m": sampled["tcp_to_cube_m"], "cube_drift_since_previous_sample_m": drift,
            "trials": trial}


def run(config, source, grant, heartbeat):
    record = load(source)
    before = sources(record)
    cases = []
    for case in record["fitting_cases"]:
        heartbeat()
        cases.append(one_pose(config["asset_root"], case, before[case["seed"]]))
    return {"scope": "MuJoCo static pose intervention, no learned rollout or training",
            "act_fit_receipt_sha256": sha(source), "reservation": grant, "cases": cases,
            "interpretation_bound": "Measured arm and cube positions with zero velocities and initial reset cube orientation. "
                                    "All five non-gripper targets stay fixed. No carry, release or containment is tested."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=PILOT / "act-grasp-pose-intervention.json")
    args = parser.parse_args()
    config = settings()
    with admitted(config, "smart-swarm-static-grasp-diagnosis") as (grant, heartbeat):
        configure_renderer(config)
        result = run(config, PILOT / "act-clock-fit.json", grant, heartbeat)
    atomic(args.output, result)
    print(args.output)


if __name__ == "__main__":
    main()
