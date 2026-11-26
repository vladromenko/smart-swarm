#!/usr/bin/env python3
"""Time CPU rasterization on exact admitted compact motor observations."""

import argparse
from contextlib import closing
import json
import os
from pathlib import Path
import subprocess
import sys
from time import monotonic

import numpy as np

from smart_swarm.cli import ROOT, settings
from smart_swarm.episode import atomic
from smart_swarm.rendering import configure_renderer
from smart_swarm.research_jobs import admitted

RAW = ROOT.parent.parent / "work/motor-compact-rehearsal-v2/raw"
SOURCES = ("train/so101-10000", "validation/so101-20001")
ISSUES = (0, 3, 41, 58, 80, 119, 132, 150, 178, 221, 240, 275, 330, 357, 418, 450, 484, 509)
THREADS = ("default", "1", "2", "4", "8")


def measure_source(assets, path):
    from smart_swarm.arm_executor import ArmExecutor
    from smart_swarm.motor import ARM, SENSORS
    from smart_swarm.motor_diagnostics import read_teacher, reset_world

    meta, arrays, _ = read_teacher(path)
    elapsed, difference = [], 0
    with np.load(path / "motor.npz", allow_pickle=False) as archive, closing(reset_world(assets, meta)) as world:
        expected = [archive[name] for name in ("wrist", "top")]
        executor = ArmExecutor(world, "aligned-sample-v2")
        executor.prime()
        for index, command in enumerate(arrays["requested"]):
            if index in ISSUES:
                before = monotonic()
                frames = world.views(ARM)
                elapsed.append(monotonic() - before)
                difference += sum(int(np.count_nonzero(frames[camera] != rgb[index]))
                                  for camera, rgb in zip(SENSORS, expected, strict=True))
            executor.submit(command)
    return {"source": str(path), "raw_sha256": meta["sha256"], "pairs_checked": len(elapsed),
            "changed_rgb_values": difference, "first_render_seconds": elapsed[0],
            "median_seconds": float(np.median(elapsed[1:])), "p95_seconds": float(np.percentile(elapsed[1:], 95))}


def one_setting(threads, cut_off, heartbeat):
    heartbeat()
    started = monotonic()
    env = os.environ.copy()
    env.pop("LP_NUM_THREADS", None)
    if threads != "default":
        env["LP_NUM_THREADS"] = threads
    command = [sys.executable, __file__, "--worker"]
    result = subprocess.run(command, env=env, check=True, stdout=subprocess.PIPE, text=True,
                            timeout=max(.01, cut_off - started))
    return {"threads": threads, "observed_seconds": monotonic() - started, "sources": json.loads(result.stdout)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    cfg = settings()
    configure_renderer(cfg)
    if args.worker:
        print(json.dumps([measure_source(cfg["asset_root"], RAW / source) for source in SOURCES]))
        return
    if args.output is None or args.output.exists():
        parser.error("Choose a new timing output")
    with admitted(cfg, "smart-swarm-compact-rasterization-profile") as (grant, heartbeat):
        deadline = monotonic() + 120.
        rows = [one_setting(threads, deadline, heartbeat) for threads in THREADS]
        verified = [row for row in rows if not any(item["changed_rgb_values"] for item in row["sources"])]
        if not verified:
            raise ValueError("No software thread setting preserved the accepted RGB byte-for-byte")
        best = min(verified, key=lambda row: max(source["median_seconds"] for source in row["sources"]))
        atomic(args.output, {"scope": "compact simulation development renderer timing",
                             "results": rows, "fastest_verified_threads": best["threads"],
                             "reservation": grant, "new_training_sources": 0, "optimizer_updates": 0})
    print(args.output)


if __name__ == "__main__":
    main()
