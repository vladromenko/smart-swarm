#!/usr/bin/env python3
"""Reproduce the two sealed moving exposures, then compare simultaneous higher-resolution RGB."""

import argparse
from contextlib import closing
from hashlib import sha256
from pathlib import Path
from time import monotonic

import numpy as np
from PIL import Image

from smart_swarm.cli import ROOT, settings
from smart_swarm.contract import json_bytes
from smart_swarm.episode import atomic, load, sha
from smart_swarm.inspection_triangulation import locate
from smart_swarm.rendering import configure_renderer
from smart_swarm.research_jobs import admitted

WORK = ROOT.parent.parent / "work"
DEVELOPMENT = WORK / "active-inspection-comparison-mobile-v2"
WAYPOINT = ROOT / "evidence/learning-pilot-v3/rosmaster-inspection-west-waypoint.json"


def paired_branch(world, output, exercise):
    import mujoco

    normal = output / "rgb-320"
    larger = output / "rgb-640"
    normal.mkdir()
    larger.mkdir()
    base_image = world.image
    captures = []
    with mujoco.Renderer(world.model, height=480, width=640) as renderer:
        def capture(camera):
            low = base_image(camera)
            before = world.snapshot().integration
            started = monotonic()
            renderer.update_scene(world.data, camera=camera)
            high = renderer.render().copy()
            elapsed = monotonic() - started
            if not np.array_equal(before, world.snapshot().integration):
                raise ValueError("The simultaneous RGB render changed simulated integration state")
            captures.append((world.ns, high, elapsed))
            return low

        world.image = capture
        try:
            branch = exercise(normal)
        finally:
            world.image = base_image
    return branch, captures, larger


def orange_pixels(rgb):
    r, g, b = rgb.astype(np.int16).transpose(2, 0, 1)
    return int(np.count_nonzero((r > 140) & (r - g > 77) & (g - b > 10)))


def compare_images(observations, captures, directory, source, output):
    rows, high_samples = [], []
    for given, old, (at, image, elapsed) in zip(observations, source, captures, strict=True):
        if at != given["at_ns"]:
            raise ValueError("The two resolution measurements must use the exact same exposure time")
        filename = given["frame"]
        high = directory / filename
        Image.fromarray(image).save(high)
        with Image.open(output / "rgb-320" / filename) as low:
            baseline_pixels = orange_pixels(np.asarray(low))
        high_samples.append((image, given["calibration"]))
        rows.append({"frame": filename, "at_ns": at, "matches_archived_frame": given["sha256"] == old["sha256"],
                     "matches_archived_time": at == old["at_ns"],
                     "matches_archived_pose": given["calibration"] == old["calibration"], "high_sha256": sha(high),
                     "low_orange_pixels": baseline_pixels, "high_orange_pixels": orange_pixels(image),
                     "high_render_seconds": elapsed})
    return rows, locate(high_samples)


def agrees_with_archive(rows):
    return all(row["matches_archived_frame"] and row["matches_archived_pose"] and row["matches_archived_time"]
               for row in rows)


def simulate_branches(world, snapshot, output, route, policy, include_control):
    from rehearse_rosmaster_mobility import compare, mobile_scan, stationary

    def scan(directory):
        return mobile_scan(world, snapshot, directory, route["station"], route)

    branch, captures, larger = paired_branch(world, output, scan)
    rows, high = compare_images(branch["policy_observations"], captures, larger, policy["scan"]["samples"], output)
    result = {"archived_exposures_reproduced": agrees_with_archive(rows), "two_native_resolutions": rows,
              "high_resolution_prediction": high, "branch_checked_ticks": branch["checked_physics_steps"],
              "motion_reached_same_endpoint": branch["evaluator"]["before"] == branch["evaluator"]["after"]}
    if include_control:
        control = output / "stationary"
        control.mkdir()
        timing = [row["at_ns"] for row in branch["policy_observations"]]

        def wait(directory):
            return stationary(world, snapshot, timing, directory)

        idle, captures, larger = paired_branch(world, control, wait)
        idle_rows, idle_high = compare_images(idle["policy_observations"], captures, larger,
                                             policy["stationary"]["samples"], control)
        result["archived_exposures_reproduced"] &= agrees_with_archive(idle_rows)
        result["stationary"] = {"two_native_resolutions": idle_rows, "high_resolution_prediction": idle_high,
                                "branch_checked_ticks": idle["checked_physics_steps"]}
        result["matched_branch_comparison"] = compare(branch, idle)
    return result


def check_pre_inspection(original, reproduction):
    from audit_primary_scope import verify_projection

    frozen = load(original / "pre-inspection-decision.json")
    signature, frames = verify_projection(reproduction)
    digest = sha256(json_bytes(signature)).hexdigest()
    if digest != frozen["permitted_input_sha256"] or frames != frozen["permitted_rgb_frames"]:
        raise ValueError("The regenerated recording changed the frozen permitted pre-inspection inputs")
    return {"permitted_input_sha256": digest, "permitted_rgb_frames": frames,
            "frozen_decision_sha256": sha(original / "pre-inspection-decision.json")}


def draw_without_offscreen_meshes(stock):
    from smart_swarm.motor_renderer import mesh_bounds, render_candidate

    original = stock.image
    bounds = mesh_bounds(stock.model)

    def capture(camera):
        if stock.renderer is None or camera == "tank/story_camera":
            return original(camera)
        image, _ = render_candidate(stock, camera, bounds)
        return image.copy()

    return capture


def record_stock(stock, variation, pickup, receipt, source, output, cull_proof):
    from mobile_inspection_case import record_decision

    original = stock.image
    if cull_proof:
        measured = load(cull_proof)
        frozen = load(source / "pre-inspection-decision.json")
        if (measured["status"] != "verified" or measured["source"] != str(source)
                or measured["pre_inspection_scope"]["permitted_input_sha256"] != frozen["permitted_input_sha256"]):
            raise ValueError("Draw-only culling must match the archived complete world inputs")
        stock.image = draw_without_offscreen_meshes(stock)
    try:
        return record_decision(stock, variation, pickup, receipt, output)
    finally:
        stock.image = original


def probe(config, source, output, heartbeat, include_control, cull_proof=None):
    from rehearse_rosmaster_mobility import read_waypoint
    from smart_swarm.evidence_rollouts import prepare_world
    from smart_swarm.inspection_mobility import prepare_mobile
    from smart_swarm.mobile_decision_state import transplant

    receipt = load(source / "receipt.json")
    policy_path = source / "policy.json"
    if receipt["status"] != "sealed" or sha(policy_path) != receipt["policy_sha256"]:
        raise ValueError("A sealed integrated development case is required")
    reference = load(policy_path)
    evaluator = load(source / "evaluator/result.json")
    route = read_waypoint(WAYPOINT, "west_surface")
    family, seed, variant = (receipt[key] for key in ("family", "seed", "variant"))
    assets = config["asset_root"]
    deadline = monotonic() + 100.
    stock, variation, pickup = prepare_world(assets, seed, family, variant=variant)
    with closing(stock):
        fixed, _, expected, _ = record_stock(stock, variation, pickup, receipt, source, output / "pre-inspection", cull_proof)
        input_scope = check_pre_inspection(source, output / "pre-inspection")
        stock.close()
        heartbeat()
        world, _, _ = prepare_mobile(assets, seed, family, variant, observed_steps=True)
        with closing(world):
            world.frame_size = (320, 240)
            world.deadline = min(world.deadline, deadline)
            snapshot = transplant(stock, world, fixed)
            result = simulate_branches(world, snapshot, output, route, reference, include_control)
    if monotonic() > deadline:
        raise TimeoutError("The exact development rerender reached the original 100-second case cap")
    if expected != receipt["evaluation"]["location"]:
        raise ValueError("The simulation development outcome changed before optical evaluation")
    return {"scope": "one previously sealed development case", "qualified_seed": seed, "variant": variant,
            "reference_policy_sha256": sha(policy_path), **result, "preserved_stationary_control": include_control,
            "pre_inspection_scope": input_scope, "stock_culling_proof_sha256": sha(cull_proof) if cull_proof else None,
            "archived_reset_state_reproduced": sha256(snapshot.integration.tobytes()).hexdigest() == evaluator["snapshot_sha256"],
            "target_for_evaluation": expected, "high_resolution_correct": result["high_resolution_prediction"]["answer"] == expected,
            "independent_cases_opened": 0, "optimizer_updates": 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=("hidden-contained", "hidden-surface", "visible-control"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--matched-control", action="store_true", help="Also reproduce the two matched stationary exposures")
    parser.add_argument("--cull-stock-proof", type=Path, help="Reuse a verified byte-exact stock frustum timing report")
    args = parser.parse_args()
    config = settings()
    configure_renderer(config)
    with admitted(config, "smart-swarm-rosmaster-dual-resolution-development") as (grant, heartbeat):
        args.output.mkdir()
        result_path = args.output / "probe.json"
        started = monotonic()
        source = DEVELOPMENT / f"mobile-rehearsal-{args.variant}"
        try:
            result = probe(config, source, args.output, heartbeat, args.matched_control, args.cull_stock_proof)
        except (OSError, KeyError, ValueError, RuntimeError, KeyboardInterrupt) as error:
            atomic(result_path, {"status": "failed", "reason": str(error), "wall_seconds": monotonic() - started})
            raise
        atomic(result_path, {"status": "recorded", **result, "wall_seconds": monotonic() - started,
                             "reservation": grant})
        if not result["archived_exposures_reproduced"] or not result["archived_reset_state_reproduced"]:
            raise ValueError("The archived reset or reference exposures did not reproduce")
    print(result_path)


if __name__ == "__main__":
    main()
