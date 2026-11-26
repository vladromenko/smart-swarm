#!/usr/bin/env python3
"""Compare stock and contact-free simulated teaching on the five established ACT worlds."""

import argparse
from hashlib import sha256
from pathlib import Path
from unittest.mock import patch

import mujoco
import numpy as np

from audit_act_wrist_contact import FIT, PILOT, observe
from smart_swarm.arm_executor import ArmExecutor
from smart_swarm.cli import settings
from smart_swarm.episode import atomic, json_bytes, load, sha
from smart_swarm.motor_diagnostics import contact_valid_placement, observe as feedback, read_teacher, reset_world, summarize
from smart_swarm.motor_input import verify_issue_time
from smart_swarm.physics import STEP
from smart_swarm.precontact_teacher import PRECONTACT, STOCK, ContactFreeApproach
from smart_swarm.teacher import run_insertion, truth

MATCHED = PILOT / "act-wrist-contact-replay.json"
EXPECTED_MATCHED = "99dbf3b0ee0f3d5fb032c06d3bbaf693063954c874fa1661d3363a2d404df052"
STEP_NS = round(STEP * 1e9)
GRASP = 120
VIEW = range(GRASP - 5, GRASP + 1)
STOCK_PREFIX = 133
CONTACT_MONITOR_END_NS = 5_200_000_000


class ObservationLimit(Exception):
    """The stock prefix has reached the last matched observation."""


class ContactClock:
    """Observe solver contacts at each integration step without modifying control."""

    def __init__(self, world):
        self.world = world
        model = world.model
        self.jaws = {model.geom(i).id for i in range(model.ngeom)
                     if model.body(model.geom_bodyid[i]).name in ("so/gripper", "so/moving_jaw_so101_v1")}
        self.arm = {model.geom(i).id for i in range(model.ngeom) if model.body(model.geom_bodyid[i]).name.startswith("so/")}
        self.cube = model.geom("cube_geom").id
        self.table = model.geom("station_surface").id
        self.first_cube = self.first_table = None
        self.step = mujoco.mj_step

    def __call__(self, model, data):
        at = round(data.time / STEP) * STEP_NS
        self.step(model, data)
        if at <= CONTACT_MONITOR_END_NS and data is self.world.data and (self.first_cube is None or self.first_table is None):
            for item in data.contact:
                first, second = int(item.geom1), int(item.geom2)
                body = second if first == self.cube else first if second == self.cube else -1
                if body in self.arm and self.first_cube is None:
                    self.first_cube = at
                jaw = second if first == self.table else first if second == self.table else -1
                if jaw in self.jaws and self.first_table is None:
                    self.first_table = at


class Trace:
    def __init__(self, world, clock):
        self.world, self.clock = world, clock
        self.camera = world.cameras["so101-1"]["so101-1/wrist"]
        self.commands, self.times, self.states = [], [], []
        self.views, self.placement = [], []
        self.initial_cube = np.asarray(truth(world)["cube_xyz_m"])

    def sample(self, action):
        issue = len(self.commands)
        self.commands.append([float(value) for value in action])
        self.times.append(self.world.ns)
        state = self.world.state("so101-1")[0]
        self.states.append(state)
        if (issue + 1) % 15 == 0:
            self.placement.append(feedback(self.world, issue + 1))
        if issue in VIEW:
            row = observe(self.world, issue, self.camera)
            drift = np.linalg.norm(np.asarray(row["cube_xyz_m"]) - self.initial_cube)
            self.views.append({**row, "measured_gripper_open_percent": state[-1],
                               "cube_drift_since_reset_mm": float(1000 * drift)})


def original_command_path(world, pickup, pose, issues=None):
    clock = ContactClock(world)
    trace = Trace(world, clock)
    starts = []

    def sample(_phase, _index, start, action, _previous):
        starts.append(start)
        trace.sample(action)
        if len(trace.commands) == issues:
            raise ObservationLimit

    try:
        with patch.object(mujoco, "mj_step", clock):
            run_insertion(ContactFreeApproach(world, pose), sample, pickup=pickup)
    except ObservationLimit:
        pass
    verify_issue_time(np.asarray(trace.times), np.asarray(starts))
    return trace


def actual_executor(world, commands):
    clock = ContactClock(world)
    trace = Trace(world, clock)
    executor = ArmExecutor(world, "aligned-sample-v2")
    with patch.object(mujoco, "mj_step", clock):
        executor.prime()
        for action in commands:
            trace.sample(action)
            executor.submit(action)
    return trace


def qualify(trace, bounds, closure, archived_placement=None):
    views = trace.views
    requested_grip = [row[-1] for row in trace.commands]
    close_issue = next(issue for issue in range(1, len(requested_grip))
                       if requested_grip[issue] < requested_grip[issue - 1] - 1e-4)
    if close_issue != GRASP or [row["issue"] for row in views] != list(VIEW):
        raise ValueError("The teaching transition no longer uses the original close issue")
    early = all(lo <= value <= hi for row in views
                for value, (lo, hi) in zip(row["cube_camera_center_px"], bounds, strict=True))
    gripper_open = min(row["measured_gripper_open_percent"] for row in views) >= 50.
    drift = max(row["cube_drift_since_reset_mm"] for row in views)
    time = trace.times[close_issue]
    cube, table = trace.clock.first_cube, trace.clock.first_table
    window = {"geometric_projection_in_reference_range": early, "gripper_open": gripper_open,
              "cube_still_within_one_mm": drift <= 1., "maximum_preclose_cube_drift_mm": drift,
              "no_cube_contact_before_close_issue": cube is None or cube >= time,
              "no_jaw_table_contact_before_close_issue": table is None or table >= time}
    passed = all(value for name, value in window.items() if name != "maximum_preclose_cube_drift_mm")
    outcome = summarize(trace.placement) if archived_placement is None else None
    valid = contact_valid_placement(outcome) if outcome is not None else archived_placement["released_stable_containment"]
    compact = ({key: value for key, value in outcome.items() if key != "feedback"}
               if outcome is not None else {"original_contact_record": archived_placement})
    return {"precontact": window, "before_close_observations": views, "close_issue": close_issue,
            "close_issue_ns": time, "first_cube_solver_contact_ns": cube, "first_jaw_table_solver_contact_ns": table,
            "precontact_gate": passed, "completed_contact_valid_placement": valid, "placement": compact,
            "matched_actuator_issues": len(trace.commands), "late_archived_cube_projection": closure}


def verify_live_replay(original, replayed):
    if (replayed.times != original.times or replayed.commands != original.commands
            or original.clock.first_cube != replayed.clock.first_cube or original.clock.first_table != replayed.clock.first_table):
        raise ValueError("The aligned deployment replay disagrees with the teacher command or solver contact clock")
    error = float(np.max(np.abs(np.subtract(replayed.states, original.states))))
    if error > 1e-8:
        raise ValueError(f"Native state changed by {error:.8f} in the aligned executor")
    return error


def one_source(asset_root, entry, prior, bounds):
    meta, arrays, _ = read_teacher(Path(entry["raw_path"]))
    if meta["sha256"] != prior["raw_sha256"] or meta["sha256"] != entry["source_sha256"]:
        raise ValueError("The archived ACT teacher source changed")
    pickup = meta["physics"]["pickup_xy_m"]
    late = next(row["cube_camera_center_px"] for row in prior["contact_window"] if row["issue"] == 132)
    world = reset_world(asset_root, meta)
    original = world.snapshot()
    outcomes = {}
    try:
        for name, pose in (("stock", STOCK), ("raised_and_retreated", PRECONTACT)):
            world.restore(original)
            issues = STOCK_PREFIX if name == "stock" else len(arrays["requested"])
            generated = original_command_path(world, pickup, pose, issues=issues)
            if name == "stock" and (not np.array_equal(np.asarray(generated.commands, dtype=np.float32), arrays["requested"][:issues])
                                    or not np.array_equal(generated.times, arrays["timestamps_ns"][:issues])):
                raise ValueError("The stock replay lost its original handover command or sample identity")
            world.restore(original)
            replay = actual_executor(world, generated.commands)
            difference = verify_live_replay(generated, replay)
            archived = meta["physics"] if name == "stock" else None
            outcomes[name] = {**qualify(replay, bounds, late, archived), "aligned_state_max_abs_residual": difference,
                              "request_sha256": sha256(json_bytes(replay.commands)).hexdigest()}
        if world.renderer is not None:
            raise ValueError("This geometry check must use no renderer")
    finally:
        world.close()
    return {"seed": meta["seed"], "source_sha256": meta["sha256"], "implementations": outcomes}


def audit(progress):
    if sha(MATCHED) != EXPECTED_MATCHED:
        raise ValueError("The archived contact provenance changed")
    reference = load(MATCHED)
    cohorts = {row["seed"]: row for row in reference["sources"]}
    late = [row["cube_camera_center_px"] for source in reference["sources"]
            for row in source["contact_window"] if row["issue"] == reference["second_observation_issue"]]
    bounds = [(float(min(column) - 4), float(max(column) + 4)) for column in zip(*late, strict=True)]
    fit = load(FIT)
    lineage = Path(fit["dataset"]) / "train/smart-swarm-lineage.json"
    entries = load(lineage)["episodes"]
    if len(entries) != 5:
        raise ValueError("The original ACT fitting cohort must contain exactly five independent starts")
    assets = settings()["asset_root"]
    sources = []
    for entry in entries:
        seed = int(entry["source_world_episode_id"].rsplit("-", 1)[1])
        sources.append(one_source(assets, entry, cohorts[seed], bounds))
        atomic(progress, {"status": "in_progress", "sources": sources})
        print(f"Checked matched ACT source {seed}", flush=True)
    passed = [row["seed"] for row in sources if row["implementations"]["raised_and_retreated"]["precontact_gate"]
              and row["implementations"]["raised_and_retreated"]["completed_contact_valid_placement"]]
    return {"scope": "Development physics on five existing ACT fitting starts", "sources": sources,
            "qualified_contact_free_starts": passed, "all_five_passed": len(passed) == 5,
            "close_pose_x_offset_and_height_m": {"stock": STOCK, "raised_and_retreated": PRECONTACT},
            "reference_projected_xy_bounds_with_four_pixel_margin": bounds, "historical_contact_sha256": sha(MATCHED),
            "dense_contact_monitor_through_ns": CONTACT_MONITOR_END_NS,
            "training_lineage_sha256": sha(lineage), "new_rgb_frames": 0, "new_optimizer_updates": 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Choose a new precontact teaching result path")
    pending = args.output.with_suffix(".in-progress.json")
    atomic(args.output, audit(pending))
    pending.unlink()
    print(args.output)


if __name__ == "__main__":
    main()
