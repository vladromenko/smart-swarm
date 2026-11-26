#!/usr/bin/env python3
"""Verify exact camera and action trajectories when reusing only graphics contexts."""

import argparse
from contextlib import closing
from pathlib import Path
from time import monotonic

import numpy as np

from smart_swarm.cli import ROOT, settings
from smart_swarm.episode import atomic, sha
from smart_swarm import motor_draw_cache
from smart_swarm.motor_visual_profile import COMPACT, visual_binding
from smart_swarm.rendering import configure_renderer
from smart_swarm.research_jobs import admitted

RAW = ROOT.parent.parent / "work/motor-compact-rehearsal-v2/raw"
RECIPES = (("train/so101-10000", "train/so101-10002"),
           ("validation/so101-20001", "validation/so101-20003"))


def prime(assets, path, cache, deadline):
    from smart_swarm.arm_executor import ArmExecutor
    from smart_swarm.motor import ARM
    from smart_swarm.motor_diagnostics import read_teacher, reset_world

    meta, arrays, _ = read_teacher(path)
    with closing(reset_world(assets, meta)) as world:
        world.deadline = min(world.deadline, deadline)
        executor = ArmExecutor(world, "aligned-sample-v2")
        executor.prime()
        for action in arrays["requested"]:
            executor.submit(action)
        started = monotonic()
        world.views(ARM)
        seconds = monotonic() - started
        cache.release(world)
    return {"source": str(path), "sha256": meta["sha256"], "fresh_first_render_s": seconds}


def check_frame(world, arrays, wanted, index):
    from smart_swarm.motor import ARM, SENSORS

    state = np.asarray(world.state(ARM)[0], dtype=np.float32)
    applied = world.arm_from_mj(world.filtered).astype(np.float32)
    started = monotonic()
    frames = world.views(ARM)
    elapsed = monotonic() - started
    if (world.ns != arrays["timestamps_ns"][index] or not np.array_equal(state, arrays["state"][index])
            or not np.array_equal(applied, arrays["applied_previous"][index])
            or not all(np.array_equal(frames[name], rgb[index]) for name, rgb in zip(SENSORS, wanted, strict=True))):
        raise ValueError(f"A cached drawing, observation or action state diverged at native issue {index}")
    return elapsed


def verify(assets, path, cache, deadline):
    from smart_swarm.arm_executor import ArmExecutor
    from smart_swarm.motor_diagnostics import contact_valid_placement, observe, read_teacher, reset_world, summarize

    meta, arrays, _ = read_teacher(path)
    started, feedback, times = monotonic(), [], []
    with closing(reset_world(assets, meta)) as world, np.load(path / "motor.npz", allow_pickle=False) as archive:
        world.deadline = min(world.deadline, deadline)
        if not cache.attach(world) or world.renderer.model is world.model:
            raise ValueError("The exact simulation needs its own fresh physical model and a matching drawing context")
        wanted = [archive[name] for name in ("wrist", "top")]
        executor = ArmExecutor(world, "aligned-sample-v2")
        executor.prime()
        for index, command in enumerate(arrays["requested"]):
            times.append(check_frame(world, arrays, wanted, index))
            executor.submit(command)
            if index % 15 == 14:
                feedback.append(observe(world, index + 1))
        if not contact_valid_placement(summarize(feedback)):
            raise ValueError("A camera cache changed the successful physical replay")
        cache.release(world)
    return {"source": str(path), "sha256": meta["sha256"], "marker": meta["target_marker"],
            "exact_camera_and_control_rows": len(times), "contact_valid": True, "reused_first_render_s": times[0],
            "warm_median_pair_s": float(np.median(times[1:])), "total_checked_wall_s": monotonic() - started}


def run(config, heartbeat):
    deadline = monotonic() + 120.
    cases = []
    with closing(motor_draw_cache.CompactDrawCache()) as cache:
        for first, later in RECIPES:
            heartbeat()
            source = prime(config["asset_root"], RAW / first, cache, deadline)
            result = verify(config["asset_root"], RAW / later, cache, deadline)
            cases.append({"priming": source, "checked": result})
    return {"status": "verified", "cache_profile": motor_draw_cache.PROFILE,
            "cache_source_sha256": sha(Path(motor_draw_cache.__file__)), "visual": visual_binding(COMPACT),
            "cases": cases, "fresh_physics_for_each_source": True, "optimizer_updates": 0, "new_sources": 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Choose a new compact draw-context verification output")
    config = settings()
    configure_renderer(config)
    with admitted(config, "smart-swarm-compact-draw-cache-verification") as (grant, heartbeat):
        started = monotonic()
        try:
            result = run(config, heartbeat)
        except (OSError, ValueError, RuntimeError, KeyboardInterrupt) as error:
            atomic(args.output, {"status": "failed", "reason": str(error), "wall_s": monotonic() - started})
            raise
        atomic(args.output, {**result, "wall_s": monotonic() - started, "reservation": grant})
    print(args.output)


if __name__ == "__main__":
    main()
