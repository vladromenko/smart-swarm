#!/usr/bin/env python3
"""Verify exact frustum culling on real development decision inputs and measure its wall cost."""

import argparse
from contextlib import closing
from pathlib import Path
from time import monotonic

import numpy as np

from smart_swarm.cli import ROOT, settings
from smart_swarm.episode import atomic, load
from smart_swarm.rendering import configure_renderer
from smart_swarm.research_jobs import admitted

SOURCE = ROOT.parent.parent / "work/active-inspection-comparison-mobile-v2/mobile-rehearsal-visible-control"


def timed(draw, camera):
    started = monotonic()
    result = draw(camera)
    return result, monotonic() - started


def compare_capture(world, output, recipe, variation, pickup):
    from mobile_inspection_case import record_decision
    from smart_swarm.motor_renderer import mesh_bounds, render_candidate

    original = world.image
    bounds = mesh_bounds(world.model)
    samples = []

    def candidate(camera):
        image, faces = render_candidate(world, camera, bounds)
        return image.copy(), faces

    def capture(camera):
        before = world.snapshot().integration
        begin_with_stock = world.renderer is None or len(samples) % 2 == 0
        if begin_with_stock:
            baseline, stock_s = timed(original, camera)
            (accelerated, faces), candidate_s = timed(candidate, camera)
        else:
            (accelerated, faces), candidate_s = timed(candidate, camera)
            baseline, stock_s = timed(original, camera)
        if not np.array_equal(before, world.snapshot().integration):
            raise ValueError("Draw-only culling changed the simulator state")
        samples.append({"camera": camera, "at_ns": world.ns, "changed_rgb_values": int(np.count_nonzero(baseline != accelerated)),
                        "stock_s": stock_s, "candidate_s": candidate_s, "excluded_triangles": faces})
        return accelerated

    world.image = capture
    try:
        record_decision(world, variation, pickup, recipe, output / "pre-inspection")
    finally:
        world.image = original
    return samples


def run(config, output):
    from probe_rosmaster_inspection_resolution import check_pre_inspection
    from smart_swarm.evidence_rollouts import prepare_world

    recipe = load(SOURCE / "receipt.json")
    start = monotonic()
    world, variation, pickup = prepare_world(config["asset_root"], recipe["seed"], recipe["family"], variant=recipe["variant"])
    with closing(world):
        samples = compare_capture(world, output, recipe, variation, pickup)
    mismatches = [row for row in samples if row["changed_rgb_values"]]
    scope = None if mismatches else check_pre_inspection(SOURCE, output / "pre-inspection")
    return {"scope": "exact renderer performance on a consumed visible development history",
            "status": "verified" if not mismatches else "rgb_mismatch", "wall_seconds": monotonic() - start,
            "source": str(SOURCE), "pre_inspection_scope": scope, "samples": samples,
            "compared_frames": len(samples), "frame_mismatches": len(mismatches),
            "warm_stock_seconds": sum(row["stock_s"] for row in samples[1:]),
            "warm_candidate_seconds": sum(row["candidate_s"] for row in samples[1:]),
            "independent_cases_opened": 0, "optimizer_updates": 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = settings()
    configure_renderer(config)
    with admitted(config, "smart-swarm-mobile-stock-frustum-verification") as (grant, heartbeat):
        args.output.mkdir()
        receipt = args.output / "probe.json"
        started = monotonic()
        try:
            heartbeat()
            result = run(config, args.output)
        except (OSError, KeyError, ValueError, RuntimeError, KeyboardInterrupt) as error:
            atomic(receipt, {"status": "failed", "reason": str(error), "wall_seconds": monotonic() - started})
            raise
        atomic(receipt, {**result, "reservation": grant})
    print(receipt)


if __name__ == "__main__":
    main()
