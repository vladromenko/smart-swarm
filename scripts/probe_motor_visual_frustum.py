#!/usr/bin/env python3
"""Rehearse a reduced drawing model with conservative per-camera clipping."""

import argparse
from contextlib import closing
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory
import time
from types import SimpleNamespace

import numpy as np

from probe_motor_rendering import PILOT, RAW, excluded, mesh_bounds, motion, render_candidate
from smart_swarm.cli import settings
from smart_swarm.episode import atomic, load, sha
from smart_swarm.rendering import configure_renderer
from smart_swarm.research_jobs import admitted


def geometry_renderer(model, data, camera, size):
    import mujoco

    scene = mujoco.MjvScene(model, maxgeom=1000)
    mount = mujoco.MjvCamera()
    mount.type = mujoco.mjtCamera.mjCAMERA_FIXED
    mount.fixedcamid = model.camera(camera).id
    mujoco.mjv_updateScene(model, data, mujoco.MjvOption(), None, mount, mujoco.mjtCatBit.mjCAT_ALL, scene)
    width, height = size
    return SimpleNamespace(scene=scene, width=width, height=height)


def counted_faces(draw, scene, ids):
    import mujoco

    mesh_geoms = [part.objid for part in scene.geoms[:scene.ngeom]
                  if part.type == mujoco.mjtGeom.mjGEOM_MESH and part.objtype == mujoco.mjtObj.mjOBJ_GEOM]
    original = sum(int(draw.mesh_facenum[draw.geom_dataid[geom]]) for geom in mesh_geoms)
    occluded = set(ids.tolist())
    retained = sum(int(draw.mesh_facenum[draw.geom_dataid[geom]]) for geom in mesh_geoms if geom not in occluded)
    return original, retained


def geometry_audit(world, drawing, snapshots, cameras, bounds):
    result = []
    for tick, state in snapshots.items():
        world.restore(state)
        views = {}
        for sensor, camera in cameras.items():
            projection = geometry_renderer(drawing, world.data, camera, world.frame_size)
            ids, removed = excluded(world, projection, bounds)
            original, retained = counted_faces(drawing, projection.scene, ids)
            if original - retained != removed:
                raise ValueError("Conservative clipping selected geometry outside the drawing scene")
            views[sensor] = {"reduced_faces": original, "reduced_faces_after_clipping": retained,
                             "removed_geometry_instances": len(ids)}
        if not np.array_equal(world.snapshot().integration, state.integration):
            raise ValueError("Dry drawing scene changed the archived solver state")
        result.append({"control_issue": tick, "views": views})
    return result


def clipped_image(world, renderer, name, bounds):
    picture, _ = render_candidate(world, name, bounds, renderer=renderer)
    return picture


def compare_images(world, renderer, snapshots, cameras, bounds, heartbeat):
    pairs = []
    for tick, snapshot in snapshots.items():
        heartbeat()
        world.restore(snapshot)
        for camera in cameras.values():
            renderer.update_scene(world.data, camera=camera)
            exact = renderer.render()
            candidate = clipped_image(world, renderer, camera, bounds)
            if not np.array_equal(exact, candidate):
                raise ValueError(f"Clipping changed reduced pixels at control issue {tick} on {camera}")
            pairs.append({"control_issue": tick, "camera": camera, "equal_rgb": True,
                          "sha256": sha256(candidate.tobytes()).hexdigest()})
    return pairs


def dense_rollout(world, renderer, reset, arrays, cameras, bounds, heartbeat):
    from smart_swarm.arm_executor import ArmExecutor
    from smart_swarm.motor_diagnostics import contact_valid_placement, observe, summarize

    world.restore(reset)
    started = time.monotonic()
    issue = ArmExecutor(world, "aligned-sample-v2")
    issue.prime()
    images = {sensor: [] for sensor in cameras}
    measured, applied, feedback = [], [], []
    pair_seconds = []
    for index, command in enumerate(arrays["requested"]):
        if index % 120 == 0:
            heartbeat()
        if world.ns != int(arrays["timestamps_ns"][index]):
            raise ValueError("Dense visual rollout drifted from the recorded issue clock")
        before = time.perf_counter()
        for sensor, camera in cameras.items():
            images[sensor].append(clipped_image(world, renderer, camera, bounds))
        pair_seconds.append(time.perf_counter() - before)
        measured.append(world.state("so101-1")[0])
        applied.append(world.arm_from_mj(world.filtered).tolist())
        issue.submit(command)
        if index % 15 == 14:
            feedback.append(observe(world, index + 1))
    success = contact_valid_placement(summarize(feedback))
    if not success:
        raise ValueError("Dense draw rehearsal lost the original contact-valid teacher trajectory")
    with TemporaryDirectory(prefix="smart-swarm-motor-drawing-") as directory:
        target = Path(directory) / "dense-camera-bytes.npz"
        np.savez(target, wrist=images["so101-1/wrist"], top=images["so101-1/top"],
                 requested=arrays["requested"], applied_previous=np.asarray(applied, dtype=np.float32),
                 state=np.asarray(measured, dtype=np.float32), timestamps_ns=arrays["timestamps_ns"],
                 applied_interval_start_ns=arrays["applied_interval_start_ns"])
        elapsed = time.monotonic() - started
        byte_count = target.stat().st_size
    return {"contact_valid": success, "recorded_camera_pairs": len(pair_seconds), "temporary_npz_bytes": byte_count,
            "elapsed_with_archive_write_s": elapsed, "median_pair_seconds": float(np.median(pair_seconds)),
            "p95_pair_seconds": float(np.percentile(pair_seconds, 95))}


def run(config, *, geometry_only, heartbeat):
    import mujoco

    from smart_swarm.motor import ARM
    from smart_swarm.motor_diagnostics import read_teacher, reset_world
    from smart_swarm.motor_visual_mesh import visual_copy

    started = time.monotonic()
    meta, arrays, _ = read_teacher(RAW / "train/so101-10000")
    with closing(reset_world(config["asset_root"], meta)) as world:
        reset = world.snapshot()
        snapshots, physics_s = motion(world, arrays)
        construction = time.monotonic()
        drawing, meshes = visual_copy(world.model)
        bounds = mesh_bounds(drawing)
        setup = time.monotonic() - construction
        cameras = world.cameras[ARM]
        result = {"source": meta["scene_group_id"], "dataset_sha256": meta["sha256"],
                  "original_contact_valid_replay": True, "separate_drawing_model": True, "altered_meshes": len(meshes),
                  "geometry_only": geometry_only, "geometry": geometry_audit(world, drawing, snapshots, cameras, bounds),
                  "physics_only_replay_seconds": physics_s, "drawing_copy_setup_seconds": setup}
        if not geometry_only:
            width, height = world.frame_size
            with closing(mujoco.Renderer(drawing, height=height, width=width)) as renderer:
                result["setup_including_world_and_renderer_seconds"] = time.monotonic() - started - physics_s
                result["compared_rgb"] = compare_images(world, renderer, snapshots, cameras, bounds, heartbeat)
                result["dense_rollout"] = dense_rollout(world, renderer, reset, arrays, cameras, bounds, heartbeat)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--geometry-only", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Choose a new report path")
    config = settings()
    if args.geometry_only:
        result = run(config, geometry_only=True, heartbeat=lambda: None)
    else:
        configure_renderer(config)
        with admitted(config, "smart-swarm-combined-motor-renderer-probe") as (grant, heartbeat):
            result = {**run(config, geometry_only=False, heartbeat=heartbeat), "reservation": grant}
    budget = load(PILOT / "motor-frame-budget-audit.json")
    target = budget["allowed_wall_seconds_per_remaining_episode"]
    if "dense_rollout" in result:
        elapsed = result["dense_rollout"]["elapsed_with_archive_write_s"] + result["setup_including_world_and_renderer_seconds"]
        result.update(dense_generation_estimate_s=elapsed, dense_estimate_under_generation_target=elapsed < target)
    atomic(args.output, {**result, "generation_wall_target_seconds_per_episode": target, "study_split_modified": False,
                         "model_optimizer_updates": 0, "implementation_sha256": sha(Path(__file__))})
    print(args.output)


if __name__ == "__main__":
    main()
