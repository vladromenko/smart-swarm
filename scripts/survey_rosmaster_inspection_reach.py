#!/usr/bin/env python3
"""Survey the nominal surface pocket from the compiled ROSMASTER camera mount."""

import argparse
from contextlib import closing
from pathlib import Path

import mujoco
import numpy as np

from audit_rosmaster_inspection_optics import POCKET, SCREEN_WALLS
from probe_rosmaster_inspection import CAMERA, SCAN, SEED
from smart_swarm.cli import settings
from smart_swarm.episode import atomic, sha
from smart_swarm.evidence_rollouts import prepare_world

RANDOM_POSES = 8_192
RANDOM_SEED = 218
PUBLIC_SAMPLES = POCKET + np.asarray([(x, y, z) for x in (-.013, 0., .013)
                                      for y in (-.013, 0., .013) for z in (-.006, .006)])
BASE_OFFSETS = [(0., 0.)] + [(x, y) for x in (-.16, -.32, -.48, -.64) for y in (0., .08, .14)]


def camera_poses(world):
    scratch = mujoco.MjData(world.model)
    indices = world.ros_joints[:4]
    limits = world.model.actuator_ctrlrange[world.ros_actuators[:4]]
    camera = world.model.camera(CAMERA).id
    generator = np.random.default_rng(RANDOM_SEED)
    sampled = generator.uniform(limits[:, 0], limits[:, 1], size=(RANDOM_POSES, 4))
    nominal = np.deg2rad(np.asarray([pose for _, pose in SCAN])).clip(*limits.T)
    joints = np.vstack((nominal, sampled))
    origins, rotations = [], []
    for pair in joints:
        scratch.qpos[indices] = pair
        mujoco.mj_kinematics(world.model, scratch)
        mujoco.mj_camlight(world.model, scratch)
        origins.append(scratch.cam_xpos[camera].copy())
        rotations.append(scratch.cam_xmat[camera].reshape(3, 3).copy())
    return joints, np.stack(origins), np.stack(rotations), float(world.model.cam_fovy[camera])


def hidden_by_wall(origins, delta, bounds):
    minimum, maximum = np.asarray(bounds)
    origin = origins[:, None, :]
    moving = np.abs(delta) > 1e-12
    intersects_parallel = np.all(moving | ((minimum <= origin) & (origin <= maximum)), axis=-1)
    distance = np.where(moving, delta, 1.)
    lower = (minimum - origin) / distance
    upper = (maximum - origin) / distance
    near = np.where(moving, np.minimum(lower, upper), -np.inf).max(axis=-1)
    far = np.where(moving, np.maximum(lower, upper), np.inf).min(axis=-1)
    return intersects_parallel & (np.maximum(near, 0.) < np.minimum(far, 1.))


def station_support(poses, offset):
    joints, static_origin, rotation, fovy = poses
    origin = static_origin + [*offset, 0.]
    delta = PUBLIC_SAMPLES[None, :, :] - origin[:, None, :]
    projected = np.einsum("nsi,nij->nsj", delta, rotation)
    depth = -projected[..., 2]
    half_angle = np.tan(np.deg2rad(fovy) / 2.)
    xy = np.abs(projected[..., :2]) / np.maximum(depth[..., None], .001)
    inside = (xy <= half_angle * np.asarray([.85 * 4 / 3, .85])).all(axis=-1) & (depth > .04)
    occluded = np.logical_or.reduce([hidden_by_wall(origin, delta, box) for box in SCREEN_WALLS.values()])
    counts = (inside & ~occluded & (origin[:, 2, None] > .308)).sum(axis=-1)
    best = int(counts.argmax())
    return {"base_translation_xy_m": list(offset), "sampled_poses": len(joints),
            "max_unoccluded_public_samples": int(counts[best]), "full_support_poses": int((counts == len(PUBLIC_SAMPLES)).sum()),
            "best_candidate_arm_deg": np.rad2deg(joints[best]).tolist(), "camera_origin_m": origin[best].tolist(),
            "original_scan_unoccluded_samples": counts[:len(SCAN)].tolist()}


def run():
    world, _, _ = prepare_world(settings()["asset_root"], SEED, "F6", variant="hidden-surface")
    with closing(world):
        poses = camera_poses(world)
        stations = [station_support(poses, offset) for offset in BASE_OFFSETS]
        return {"scope": "kinematic public aperture survey, no rendered frames or chassis rollout",
                "model_sha256": world.manifest["model_sha256"], "implementation_sha256": sha(Path(__file__)),
                "pose_random_seed": RANDOM_SEED, "public_surface_samples_m": PUBLIC_SAMPLES.tolist(),
                "stations": stations, "visibility_constraints": "Nominal FOV, aperture walls and margin only",
                "world_truth_state_read": False, "physical_motion_requested": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Choose a new kinematic survey path")
    atomic(args.output, run())
    print(args.output)


if __name__ == "__main__":
    main()
