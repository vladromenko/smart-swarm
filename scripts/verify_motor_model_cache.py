#!/usr/bin/env python3
"""Verify exact physics, controller and full RGB tracks from fresh model copies."""

import argparse
from contextlib import closing
from pathlib import Path
from time import monotonic

import numpy as np

from smart_swarm.cli import ROOT, settings
from smart_swarm.episode import atomic, sha
from smart_swarm.motor_visual_profile import COMPACT, visual_binding
from smart_swarm.rendering import configure_renderer
from smart_swarm.research_jobs import admitted

RAW = ROOT.parent.parent / "work/motor-compact-rehearsal-v2/raw"
SOURCES = ("train/so101-10002", "validation/so101-20003")


def prime(assets, seed, pool, draw, size):
    from smart_swarm.motor import ARM
    from smart_swarm.motor_renderer import MotorWorld

    with closing(pool.world(seed - 2, 100.)):
        pass
    started = monotonic()
    with closing(MotorWorld(assets, seed=seed, swap_markers=bool(seed % 2), visual_profile=COMPACT,
                            frame_size=size)) as fresh:
        cold = monotonic() - started
        snapshot = fresh.snapshot()
        fresh.views(ARM)
        draw.release(fresh)
    return cold, snapshot


def compare_archive(capture, path, arrays):
    from smart_swarm.motor import SENSORS

    now, interval, state, requested, applied = zip(*capture.records, strict=True)
    for key, values in (("timestamps_ns", now), ("applied_interval_start_ns", interval), ("state", state),
                        ("requested", requested), ("applied_previous", applied)):
        if not np.array_equal(np.asarray(values, dtype=arrays[key].dtype), arrays[key]):
            raise ValueError(f"A copied motor world changed archived {key}")
    with np.load(path / "motor.npz", allow_pickle=False) as archive:
        for key, sensor in zip(("wrist", "top"), SENSORS, strict=True):
            if not np.array_equal(np.asarray(capture.frames[sensor]), archive[key]):
                raise ValueError(f"A copied motor world changed the {key} RGB sequence")


def verify(assets, path, pool, draw, deadline):
    from smart_swarm.motor import IMAGE_SIZE, MotorCapture, record_insertion, variation
    from smart_swarm.motor_diagnostics import read_teacher

    meta, arrays, _ = read_teacher(path)
    seed = meta["seed"]
    cold, initial = prime(assets, seed, pool, draw, IMAGE_SIZE)
    started = monotonic()
    with closing(pool.world(seed, max(.01, deadline - started))) as world:
        warm = monotonic() - started
        if (not world.motor_model_cache_hit or not draw.attach(world)
                or world.manifest["model_sha256"] != meta["model_sha256"]
                or not np.array_equal(world.snapshot().integration, initial.integration)):
            raise ValueError("The initialized model copy did not match its independently compiled start")
        settings = variation(world)
        if any(value != meta["physics"][key] for key, value in settings.items()):
            raise ValueError("The model copy changed independent source variation")
        world.step(200)
        collector = MotorCapture(world)
        record_insertion(world, collector.sample, settings["pickup_xy_m"])
        world.step(600)
        success, physics = collector.result()
        if not success or any(value != meta["physics"][key] for key, value in physics.items()):
            raise ValueError("The regenerated motor trajectory changed its contact or containment outcome")
        compare_archive(collector, path, arrays)
        draw.release(world)
    return {"source": str(path), "sha256": meta["sha256"], "marker": meta["target_marker"],
            "independent_physics": True, "exact_rgb_and_controls": len(collector.records),
            "cold_world_seconds": cold, "pristine_copy_seconds": warm, "wall_seconds_with_copy": monotonic() - started}


def run(config, heartbeat):
    from smart_swarm.motor import IMAGE_SIZE
    from smart_swarm.motor_draw_cache import CompactDrawCache
    from smart_swarm import motor_model_cache

    deadline = monotonic() + 120.
    cases = []
    with closing(motor_model_cache.CompactModelCache(config["asset_root"], IMAGE_SIZE)) as pool, closing(CompactDrawCache()) as draw:
        for source in SOURCES:
            heartbeat()
            cases.append(verify(config["asset_root"], RAW / source, pool, draw, deadline))
    return {"status": "verified", "model_profile": motor_model_cache.PROFILE,
            "implementation_sha256": sha(Path(motor_model_cache.__file__)), "visual": visual_binding(COMPACT),
            "cases": cases, "optimizer_updates": 0, "new_sources": 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Choose a new copy-verification result path")
    cfg = settings()
    configure_renderer(cfg)
    with admitted(cfg, "smart-swarm-fresh-model-copy-verification") as (grant, heartbeat):
        started = monotonic()
        try:
            result = run(cfg, heartbeat)
        except (OSError, KeyError, ValueError, RuntimeError, KeyboardInterrupt) as error:
            atomic(args.output, {"status": "failed", "error": str(error), "wall_seconds": monotonic() - started})
            raise
        atomic(args.output, {**result, "wall_seconds": monotonic() - started, "reservation": grant})
    print(args.output)


if __name__ == "__main__":
    main()
