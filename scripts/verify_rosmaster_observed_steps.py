#!/usr/bin/env python3
"""Verify split-step contact observations against legacy simulation and archived RGB."""

import argparse
from contextlib import closing
from hashlib import sha256
from pathlib import Path
from time import monotonic

import numpy as np

from smart_swarm.cli import ROOT, settings
from smart_swarm.contract import json_bytes
from smart_swarm.episode import atomic, load, sha
from smart_swarm.rendering import configure_renderer
from smart_swarm.research_jobs import admitted

WORK = ROOT.parent.parent / "work"
VARIANTS = ("hidden-contained", "hidden-surface")
REFERENCE = WORK / "rosmaster-mobile-optical-v2/hidden-surface/rehearsal.json"
WAYPOINT = ROOT / "evidence/learning-pilot-v3/rosmaster-inspection-west-waypoint.json"


class ObservationStopped(Exception):
    """A deliberate callback refusal during the parity test."""


def contact_signature(world):
    data = world.data
    fields = (data.qpos, data.qvel, data.contact.geom1, data.contact.geom2, data.contact.dist, data.contact.pos)
    return world.physics_steps, tuple(sha256(field.tobytes()).hexdigest() for field in fields)


def verify_stop(world):
    start = world.snapshot()
    stop = start.physics_steps + 3

    def reject_third_observation():
        if world.physics_steps == stop:
            raise ObservationStopped

    try:
        world.step_observed(20, reject_third_observation)
    except ObservationStopped:
        stopped = world.snapshot()
    else:
        raise ValueError("A rejected observation did not stop the simulation")
    world.restore(start)
    world.step(3)
    if stopped.physics_steps != stop or not np.array_equal(stopped.integration, world.snapshot().integration):
        raise ValueError("A rejected observation allowed an additional integration tick")
    return True


def verify_ticks(assets, variant):
    from smart_swarm.inspection_mobility import prepare_mobile

    world, _, _ = prepare_mobile(assets, 416031, "F6", variant, seconds=30)
    with closing(world):
        world.step(200)
        world.data.ctrl[world.mobile_actuators] = (-.05, .05)
        world.data.ctrl[world.ros_actuators[0]] += .1
        world.pending[2] += .08
        world.tank_target[:] = (.08, .02)
        reset = world.snapshot()
        legacy = []
        for _ in range(600):
            world.step(1)
            legacy.append(contact_signature(world))
        end = world.snapshot()
        world.restore(reset)
        observed = []
        world.step_observed(len(legacy), lambda: observed.append(contact_signature(world)))
        if observed != legacy or not np.array_equal(end.integration, world.snapshot().integration):
            raise ValueError(f"{variant} changed a measured state, contact or final solver state")
        return {"variant": variant, "exact_tick_contacts_and_states": len(observed),
                "final_integration_exact": True, "refusal_stopped_before_next_tick": verify_stop(world)}


def verify_route(assets, output, heartbeat):
    from rehearse_rosmaster_mobility import compare, decision_snapshot, mobile_scan, read_waypoint, stationary
    from smart_swarm.inspection_mobility import prepare_mobile

    old = load(REFERENCE)
    variant, family, station = (old[key] for key in ("variant", "family", "inspection_station"))
    waypoint = read_waypoint(WAYPOINT, station)
    started = monotonic()
    world, _, pickup = prepare_mobile(assets, old["seed"], family, variant, observed_steps=True)
    with closing(world):
        world.frame_size = (320, 240)
        snapshot, teacher = decision_snapshot(world, pickup, family, variant)
        heartbeat()
        branch_start = monotonic()
        branch_dirs = {name: output / name for name in ("active", "stationary")}
        for path in branch_dirs.values():
            path.mkdir()
        active = mobile_scan(world, snapshot, branch_dirs["active"], station, waypoint)
        schedule = [row["at_ns"] for row in active["policy_observations"]]
        idle = stationary(world, snapshot, schedule, branch_dirs["stationary"])
        matched = compare(active, idle)
    sections = {"teacher_contact_gate": teacher, "active": active, "stationary": idle, "comparison": matched}
    differences = [name for name, value in sections.items() if json_bytes(value) != json_bytes(old[name])]
    modules = ("src/smart_swarm/physics.py", "src/smart_swarm/inspection_mobility.py")
    return {"reference": str(REFERENCE), "reference_sha256": sha(REFERENCE), "original_route_exact": not differences,
            "solver_sources_sha256": {name: sha(ROOT / name) for name in modules},
            "changed_sections": differences, "status": "different" if differences else "matched",
            "old_total_seconds": old["elapsed_wall_seconds"], "total_seconds": monotonic() - started,
            "observed_branch_seconds": monotonic() - branch_start, "steps_per_branch": active["checked_physics_steps"],
            **sections}


def admitted_route(cfg, output):
    with admitted(cfg, "smart-swarm-rosmaster-observed-step-parity") as (grant, heartbeat):
        output.mkdir()
        receipt = output / "verification.json"
        started = monotonic()
        try:
            result = verify_route(cfg["asset_root"], output, heartbeat)
        except (OSError, ValueError, RuntimeError, KeyboardInterrupt) as error:
            atomic(receipt, {"status": "failed", "reason": str(error), "elapsed_wall_seconds": monotonic() - started})
            raise
        atomic(receipt, {**result, "reservation": grant})
        if not result["original_route_exact"]:
            raise ValueError(f"The optimized route changed {result['changed_sections']}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--route", action="store_true", help="Render and compare both full recorded branches")
    args = parser.parse_args()
    cfg = settings()
    configure_renderer(cfg)
    if args.route:
        admitted_route(cfg, args.output)
    else:
        if args.output.exists():
            parser.error("Choose a new step verification output")
        result = [verify_ticks(cfg["asset_root"], name) for name in VARIANTS]
        atomic(args.output, {"scope": "archived F6 simulation setups", "results": result})
    print(args.output)


if __name__ == "__main__":
    main()
