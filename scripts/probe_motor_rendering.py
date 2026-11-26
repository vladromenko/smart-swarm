#!/usr/bin/env python3
"""Benchmark conservative culling and reduced drawing meshes on archived SO101 motion."""

import argparse
from contextlib import closing
from hashlib import sha256
import os
from pathlib import Path
import time

import numpy as np

from smart_swarm.cli import settings
from smart_swarm.episode import atomic, load, sha
from smart_swarm.motor_renderer import excluded, mesh_bounds, render_candidate
from smart_swarm.rendering import configure_renderer
from smart_swarm.research_jobs import admitted

PILOT = Path(__file__).resolve().parents[1] / "evidence/learning-pilot-v3"
RAW = PILOT.parents[3] / "work/motor-aligned-v3-raw"
SNAPSHOTS = (0, 60, 150, 240, 330, 450, 509)


def motion(world, arrays):
    from smart_swarm.arm_executor import ArmExecutor
    from smart_swarm.motor_diagnostics import contact_valid_placement, observe, summarize

    started = time.monotonic()
    executor = ArmExecutor(world, "aligned-sample-v2")
    executor.prime()
    selected, feedback = {}, []
    for index, command in enumerate(arrays["requested"]):
        if index in SNAPSHOTS:
            if world.ns != int(arrays["timestamps_ns"][index]):
                raise ValueError("Snapshot no longer matches the original action issue time")
            selected[index] = world.snapshot()
        executor.submit(command)
        if index % 15 == 14:
            feedback.append(observe(world, index + 1))
    if set(selected) != set(SNAPSHOTS) or not contact_valid_placement(summarize(feedback)):
        raise ValueError("The archived probe motion failed its complete teacher replay")
    return selected, time.monotonic() - started


def camera_measurement(world, camera, bounds):
    started = time.perf_counter()
    stock = world.image(camera)
    normal = time.perf_counter() - started
    started = time.perf_counter()
    filtered, faces = render_candidate(world, camera, bounds)
    candidate = time.perf_counter() - started
    return {"stock_seconds": normal, "candidate_seconds": candidate, "culled_mesh_faces": faces,
            "stock_rgb_sha256": sha256(stock.tobytes()).hexdigest(),
            "candidate_rgb_sha256": sha256(filtered.tobytes()).hexdigest(),
            "changed_rgb_elements": int(np.count_nonzero(stock != filtered))}


def segmented(renderer):
    renderer.enable_segmentation_rendering()
    try:
        return renderer.render()
    finally:
        renderer.disable_segmentation_rendering()


def task_regions(model):
    names = {model.geom(index).name: index for index in range(model.ngeom)}
    return {"cube": [names["cube_geom"]],
            **{name: [index for key, index in names.items() if key.startswith(f"{name}_")]
               for name in ("carrier", "decoy")}}


def geometry_visibility(regions, stock, reduction):
    import mujoco

    kind = mujoco.mjtObj.mjOBJ_GEOM
    result = {}
    for label, identities in regions.items():
        original = np.isin(stock[..., 0], identities) & (stock[..., 1] == kind)
        candidate = np.isin(reduction[..., 0], identities) & (reduction[..., 1] == kind)
        union = int(np.count_nonzero(original | candidate))
        shared = int(np.count_nonzero(original & candidate))
        result[label] = {"stock_pixels": int(original.sum()), "reduced_pixels": int(candidate.sum()),
                         "mask_iou": shared / union if union else None}
    return result


def teaching_view(world, renderer, name, regions, stock_rgb):
    import mujoco

    from smart_swarm.vision import center

    started = time.perf_counter()
    renderer.update_scene(world.data, camera=name)
    reduced = renderer.render()
    elapsed = time.perf_counter() - started
    world.renderer.update_scene(world.data, camera=name)
    fallback = segmented(world.renderer)
    segmentation = segmented(renderer)
    cameras = zip(renderer.scene.camera, world.renderer.scene.camera, strict=True)
    aligned = all(np.array_equal(a.pos, b.pos) and np.array_equal(a.forward, b.forward) and
                  np.array_equal(a.up, b.up) and a.frustum_near == b.frustum_near and
                  a.frustum_top == b.frustum_top and a.frustum_bottom == b.frustum_bottom
                  and a.frustum_center == b.frustum_center and a.frustum_far == b.frustum_far for a, b in cameras)
    if not aligned or renderer.model.geom_type[world.model.geom("cube_geom").id] != mujoco.mjtGeom.mjGEOM_BOX:
        raise ValueError("The reduced drawing changed the issued camera or task geometry")
    cues = {label: {"stock_px": center(stock_rgb, label), "reduced_px": center(reduced, label)}
            for label in ("cube", "cyan", "violet")}
    return {"reduced_seconds": elapsed, "rgb_sha256": sha256(reduced.tobytes()).hexdigest(),
            "changed_rgb_elements": int(np.count_nonzero(reduced != stock_rgb)),
            "camera_extrinsics_and_frustum_exact": aligned,
            "task_visibility": geometry_visibility(regions, fallback, segmentation), "rgb_marker_centers": cues}


def mesh_measurements(world, snapshots, heartbeat, cameras):
    import mujoco

    from smart_swarm.motor_visual_mesh import VISUAL_GRID, visual_copy

    started = time.monotonic()
    drawing, changes = visual_copy(world.model)
    regions = task_regions(world.model)
    width, height = world.frame_size
    traces = []
    with closing(mujoco.Renderer(drawing, height=height, width=width)) as renderer:
        world.restore(snapshots[0])
        renderer.update_scene(world.data, camera=next(iter(cameras.values())))
        renderer.render()
        setup = time.monotonic() - started
        for index, state in snapshots.items():
            heartbeat()
            world.restore(state)
            samples = {sensor: teaching_view(world, renderer, camera, regions, world.image(camera))
                       for sensor, camera in cameras.items()}
            if not np.array_equal(world.snapshot().integration, state.integration):
                raise ValueError("Drawing the alternate model changed the solver state")
            traces.append({"sample": index, "camera_pairs": samples, "simulation_time_ns": world.ns})
    return {"probe_status": "measured", "grid_resolution": VISUAL_GRID, "setup_seconds": setup,
            "original_unique_mesh_faces": int(world.model.mesh_facenum.sum()),
            "reduced_unique_mesh_faces": int(drawing.mesh_facenum.sum()), "updated_meshes": changes,
            "checked_frames": traces}


def prototype(assets, root, heartbeat):
    from smart_swarm.motor import ARM, SENSORS
    from smart_swarm.motor_diagnostics import read_teacher, reset_world

    started = time.monotonic()
    meta, arrays, _ = read_teacher(root)
    with closing(reset_world(assets, meta)) as world:
        setup = time.monotonic() - started
        snapshots, physics = motion(world, arrays)
        bounds = mesh_bounds(world.model)
        cameras = world.cameras[ARM]
        world.restore(snapshots[0])
        world.views(ARM)
        checked = []
        for index, state in snapshots.items():
            heartbeat()
            world.restore(state)
            samples = {sensor: camera_measurement(world, cameras[sensor], bounds) for sensor in SENSORS}
            checked.append({"sample": index, "camera_pairs": samples, "simulation_time_ns": world.ns})
    return {"source": meta["scene_group_id"], "dataset_sha256": meta["sha256"],
            "setup_with_archive_read_seconds": setup, "recorded_actions_replay_without_rgb_seconds": physics,
            "original_action_rows": len(arrays["requested"]), "mesh_faces_in_model": int(bounds["face_count"].sum()),
            "checked_frames": checked}


def reduced_prototype(config, root, heartbeat):
    from smart_swarm.motor import ARM
    from smart_swarm.motor_diagnostics import read_teacher, reset_world

    meta, arrays, _ = read_teacher(root)
    with closing(reset_world(config["asset_root"], meta)) as world:
        snapshots, physics = motion(world, arrays)
        try:
            reduced = mesh_measurements(world, snapshots, heartbeat, world.cameras[ARM])
        except (ValueError, RuntimeError, OSError) as error:
            reduced = {"probe_status": "stopped", "reason": str(error)}
    return {"source": meta["scene_group_id"], "dataset_sha256": meta["sha256"],
            "recorded_actions_replay_without_rgb_seconds": physics, "original_action_rows": len(arrays["requested"]),
            "reduced_drawing_meshes": reduced}


def summary(rows, budget):
    mismatches = [item["changed_rgb_elements"] for row in rows for frame in row["checked_frames"]
                  for item in frame["camera_pairs"].values()]
    times = {}
    for mode in ("stock", "candidate"):
        observed = [sum(sensor[f"{mode}_seconds"] for sensor in frame["camera_pairs"].values())
                    for row in rows for frame in row["checked_frames"]]
        times[mode] = {"median_pair_seconds": float(np.median(observed)),
                       "p95_pair_seconds": float(np.percentile(observed, 95))}
    frames = max(row["original_action_rows"] for row in rows)
    overhead = max(row["recorded_actions_replay_without_rgb_seconds"] for row in rows)
    fits = overhead + frames * times["candidate"]["p95_pair_seconds"]
    limit = budget["allowed_wall_seconds_per_remaining_episode"]
    return {"all_sampled_frames_bit_exact": max(mismatches) == 0, "max_changed_rgb_elements": max(mismatches),
            "timing": times, "simple_rollout_time_projection_excluding_setup_and_write_seconds": fits,
            "generation_wall_target_seconds_per_episode": limit,
            "eligible_for_full_trajectory_comparison": max(mismatches) == 0 and fits < limit}


def reduced_summary(rows, budget):
    probes = [row["reduced_drawing_meshes"] for row in rows]
    if any(probe["probe_status"] != "measured" for probe in probes):
        return {"status": "incomplete", "available_receipts": sum(p["probe_status"] == "measured" for p in probes)}
    cameras = [frame["camera_pairs"] for probe in probes for frame in probe["checked_frames"]]
    pair_seconds = [sum(view["reduced_seconds"] for view in camera.values()) for camera in cameras]
    visibility = [view["task_visibility"] for camera in cameras for view in camera.values()]
    p95 = float(np.percentile(pair_seconds, 95))
    count = max(row["original_action_rows"] for row in rows)
    physics = max(row["recorded_actions_replay_without_rgb_seconds"] for row in rows)
    projected = p95 * count + physics
    return {"status": "development", "nonstandard_rgb_profile": True, "sampled_camera_pairs": len(cameras),
            "median_pair_seconds": float(np.median(pair_seconds)), "p95_pair_seconds": p95,
            "projected_rollout_seconds_excluding_setup_and_write": projected,
            "current_generation_wall_target_seconds": budget["allowed_wall_seconds_per_remaining_episode"],
            "fully_lost_stock_cube_detections": sum(row["cube"]["stock_pixels"] > 0 and row["cube"]["reduced_pixels"] == 0
                                                     for row in visibility),
            "simulated_contact_tested_on_stock_physics": True}


def measure_reduced(config, path, budget, identity, grant, heartbeat):
    companion = path.with_name(path.stem + "-reduced" + path.suffix)
    if companion.exists():
        raise ValueError("Choose a new lower-triangle timing report path")
    implementation = Path(__file__).resolve().parents[1] / "src/smart_swarm/motor_visual_mesh.py"
    metadata = {**identity, "scope": "Development lower-triangle drawing profile",
                "reservation": grant, "algorithm_sha256": sha(implementation), "original_rgb_admissible": False}
    samples = []
    for seed in (10000, 10001):
        samples.append(reduced_prototype(config, RAW / "train" / f"so101-{seed}", heartbeat))
        atomic(companion, {**metadata, "completed_source_attempts": len(samples), "complete": len(samples) == 2,
                           "measurements": samples,
                           "reduced_rgb_profile": reduced_summary(samples, budget)})
    print(companion, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Choose a new timing report path")
    bounds = PILOT / "motor-frame-budget-audit.json"
    budget = load(bounds)
    identity = {"software_threads_requested": os.environ.get("LP_NUM_THREADS"), "budget_sha256": sha(bounds)}
    config = settings()
    configure_renderer(config)
    with admitted(config, "smart-swarm-exact-motor-frame-probe") as (grant, heartbeat):
        samples = [prototype(config["asset_root"], RAW / "train" / f"so101-{seed}", heartbeat)
                   for seed in (10000, 10001)]
        result = {**identity, "scope": "Development renderer probe with two archived teach trajectories",
                  "measurements": samples, "reservation": grant, **summary(samples, budget)}
        atomic(args.output, result)
        print(args.output, flush=True)
        measure_reduced(config, args.output, budget, identity, grant, heartbeat)


if __name__ == "__main__":
    main()
