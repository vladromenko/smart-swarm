#!/usr/bin/env python3
"""Screen simulated wrist waypoints against the original fixture contacts."""

import argparse
from contextlib import closing
from pathlib import Path

import mujoco
import numpy as np

from audit_rosmaster_inspection_optics import obstacles
from probe_rosmaster_inspection import CAMERA, SEED
from rehearse_rosmaster_mobility import HOME_DEG, SURFACE_POSES
from smart_swarm.cli import settings
from smart_swarm.episode import atomic, sha
from smart_swarm.inspection_mobility import SceneContactWatch, STATIONS, mounted_feedback, prepare_mobile

STATION = "west_surface"
ROLL_DEG = 85.
PLAN_SEED = 618_033
COARSE_SAMPLES = 25
FINE_SAMPLES = 161


def collisions(world, watch, joints):
    world.data.qpos[world.ros_joints[:5]] = np.deg2rad(joints)
    mujoco.mj_forward(world.model, world.data)
    watch.pairs.clear()
    watch.sample()
    return sorted(watch.pairs)


def segment_clear(world, watch, start, end, samples):
    return not any(collisions(world, watch, position) for position in np.linspace(start, end, samples))


def candidates(home, goal):
    midpoint = (home[:4] + goal[:4]) / 2.
    fixed = np.asarray([[0, -40, 0, 0], [0, -50, 20, -5], [0, -60, 25, -10], [0, -35, -20, 25],
                        [-25, -45, 15, 20], [20, 20, -40, 15]]) + midpoint
    generator = np.random.default_rng(PLAN_SEED)
    sampled = generator.normal(midpoint, [26., 38., 42., 28.], size=(600, 4))
    options = np.vstack((fixed, sampled)).clip(-88., 88.)
    order = np.linalg.norm(options - home[:4], axis=1) + np.linalg.norm(goal[:4] - options, axis=1)
    return [np.r_[options[index], ROLL_DEG] for index in order.argsort()]


def search(world):
    world.data.qpos[world.mobile_joints] = STATIONS[STATION]
    watch = SceneContactWatch(world)
    home = np.asarray((*HOME_DEG, ROLL_DEG))
    goal = np.asarray((*SURFACE_POSES[STATION], ROLL_DEG))
    end_contacts = collisions(world, watch, goal)
    camera = mounted_feedback(world, CAMERA)
    straight = segment_clear(world, watch, home, goal, FINE_SAMPLES)
    states = candidates(home, goal)
    coarse = (point for point in states if segment_clear(world, watch, home, point, COARSE_SAMPLES)
              and segment_clear(world, watch, point, goal, COARSE_SAMPLES))
    waypoint = next((point for point in coarse if segment_clear(world, watch, home, point, FINE_SAMPLES)
                     and segment_clear(world, watch, point, goal, FINE_SAMPLES)), None)
    return {"station": STATION, "base_translation_xy_m": list(STATIONS[STATION]), "home_joints_deg": home.tolist(),
            "goal_joints_deg": goal.tolist(), "waypoint_deg": None if waypoint is None else waypoint.tolist(),
            "endpoint_external_contacts": end_contacts, "direct_path_clear": straight,
            "candidate_limit": len(states), "fine_samples_per_segment": FINE_SAMPLES,
            "candidate_camera_pose": camera, "public_pocket_center_line_occlusion": obstacles(camera["origin_m"])}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Choose a new waypoint report path")
    world, _, _ = prepare_mobile(settings()["asset_root"], SEED, "F6", "hidden-surface")
    with closing(world):
        report = search(world)
        atomic(args.output, {**report, "scope": "offline contact screening at fixed public station",
                             "seed": SEED, "plan_random_seed": PLAN_SEED, "source_model_sha256": world.manifest["model_sha256"],
                             "implementation_sha256": sha(Path(__file__)), "dynamic_route_run": False})
    print(args.output)


if __name__ == "__main__":
    main()
