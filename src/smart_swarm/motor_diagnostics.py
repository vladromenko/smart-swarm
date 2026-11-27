"""Replay preserved successful demonstrations before inspecting learned behavior."""

from hashlib import sha256
from pathlib import Path
import math

import numpy as np

from .arm_executor import ArmExecutor
from .episode import atomic, load
from .evaluate_motor import Worker, issue_action, sha256_stream
from .motor import ARM, IMAGE_SIZE, variation
from .motor_renderer import MotorWorld
from .motor_visual_profile import match_visual, recorded_visual
from .physics import ARM_NAMES, ARM_UNITS
from .teacher import truth

SEEDS = (10_000, 10_001, 10_002, 20_000, 20_001)


def paths(root):
    return [root / ("train" if seed < 20_000 else "validation") / f"so101-{seed}" for seed in SEEDS]


def read_teacher(path):
    manifest = load(path / "manifest.json")
    file = path / "motor.npz"
    with file.open("rb") as stream:
        digest = sha256_stream(stream)
    if (digest != manifest["sha256"] or manifest["status"] != "accepted"
            or not manifest["physics"]["released_stable_containment"]):
        raise ValueError(f"Teacher source failed the contact or integrity admission check: {path}")
    if tuple(manifest["names"]) != ARM_NAMES or tuple(manifest["units"]) != ARM_UNITS:
        raise ValueError("Teacher action schema mismatch")
    with np.load(file, allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in ("timestamps_ns", "applied_interval_start_ns",
                                               "state", "requested", "applied_previous")}
        camera = {key: archive[key][0].copy() for key in ("wrist", "top")}
    return manifest, arrays, camera


def replay(world, arrays, profile, *, contact_issues=range(0)):
    executor = ArmExecutor(world, profile)
    executor.prime()
    feedback, states, sample_times = [], [], []
    for idx, asked in enumerate(arrays["requested"]):
        if profile == "aligned-sample-v2":
            states.append(world.state(ARM)[0])
            sample_times.append(world.ns)
        executor.submit(asked)
        if idx % 15 == 14 or idx in contact_issues:
            feedback.append(observe(world, idx + 1))
    result = summarize(feedback)
    if states:
        delta = np.asarray(states) - arrays["state"]
        result["sampled_state_max_abs_residual"] = np.abs(delta).max(axis=0).tolist()
        result["precontact_state_max_abs_residual"] = float(np.max(np.abs(delta[:60])))
        result["sample_times_exact"] = bool(np.array_equal(sample_times, arrays["timestamps_ns"]))
    return result


def observe(world, tick):
    result = truth(world)
    point = world.data.site("so/gripperframe").xpos.copy()
    result["tick"] = tick
    result["tcp_xyz_m"] = point.tolist()
    result["tcp_to_cube_m"] = float(np.linalg.norm(point - np.asarray(result["cube_xyz_m"])))
    return result


def summarize(feedback):
    held = [item for item in feedback if len(item["gripper_contact_bodies"]) >= 2]
    lift = [item for item in held if item["cube_height_above_table_m"] >= .10]
    final = all(item["inside_carrier"] and item["released"] and item["stationary"]
                for item in feedback[-2:])
    carry = [item for item in feedback if 240 <= item["tick"] <= 330]
    supported = sum(len(item["gripper_contact_bodies"]) >= 2 for item in carry)
    distance = min(item["tcp_to_cube_m"] for item in feedback)
    stage = first_failed_stage(feedback)
    return {"first_failed_stage": stage, "stable_containment": final,
            "bilateral_contact": bool(held), "held_lift_10cm": bool(lift),
            "carry_bilateral_samples": supported, "carry_sample_count": len(carry),
            "minimum_tcp_cube_distance_m": distance,
            "max_cube_height_m": max(item["cube_height_above_table_m"] for item in feedback),
            "feedback": feedback}


def contact_valid_placement(result):
    return bool(result["stable_containment"] and result["held_lift_10cm"] and result["first_failed_stage"] is None)


def first_failed_stage(feedback):
    held = [row for row in feedback if len(row["gripper_contact_bodies"]) >= 2]
    reached = [row for row in held if row["cube_height_above_table_m"] >= .06
               and np.linalg.norm(np.subtract(row["cube_xyz_m"][:2], row["carrier_xyz_m"][:2])) <= .055]
    if not held:
        return "approach" if min(row["tcp_to_cube_m"] for row in feedback) > .05 else "grasp"
    if not reached:
        return "carry"
    contained = [row for row in feedback if row["inside_carrier"] and row["released"]]
    if not contained:
        return "release"
    if not all(row["inside_carrier"] and row["released"] and row["stationary"] for row in feedback[-2:]):
        return "termination"
    return None


def reset_world(assets, manifest):
    marker = manifest.get("target_marker", "cyan")
    binding = recorded_visual(manifest)
    world = MotorWorld(assets, seed=manifest["seed"], frame_size=IMAGE_SIZE, timeout_s=150,
                       legacy_reveal="target_marker" not in manifest and manifest["seed"] in SEEDS,
                       swap_markers=marker == "violet", visual_profile=binding["visual_profile"])
    world.motor_requested_marker = marker
    settings = variation(world)
    if (settings["pickup_xy_m"] != manifest["physics"]["pickup_xy_m"]
            or settings["cube_sliding_friction"] != manifest["physics"]["cube_sliding_friction"]
            or world.manifest["model_sha256"] != manifest["model_sha256"]):
        world.close()
        raise ValueError("Could not reconstruct the exact recorded teacher setup")
    world.step(200)
    return world


def teacher_case(assets, source):
    meta, arrays, frames = read_teacher(source)
    world = reset_world(assets, meta)
    start = world.snapshot()
    try:
        report = {"seed": meta["seed"], "recording": str(source), "source_sha256": meta["sha256"]}
        for profile in ("historical-eval-v1", "aligned-sample-v2"):
            world.restore(start)
            report[profile] = replay(world, arrays, profile)
        world.restore(start)
        engine = ArmExecutor(world, "aligned-sample-v2")
        engine.prime()
        live = list(world.views(ARM).values())
        report["initial_camera_rgb_bit_exact"] = [bool(np.array_equal(a, b))
                                                  for a, b in zip(live, frames.values(), strict=True)]
        return report
    finally:
        world.close()


def teachers(assets, raw, output):
    results = [teacher_case(assets, path) for path in paths(raw)]
    gate = all(row[profile]["first_failed_stage"] is None and
               row[profile]["carry_bilateral_samples"] == row[profile]["carry_sample_count"] == 7 for row in results
               for profile in ("historical-eval-v1", "aligned-sample-v2"))
    alignment = all(row["aligned-sample-v2"]["sample_times_exact"]
                    and row["aligned-sample-v2"]["precontact_state_max_abs_residual"] < 1e-4
                    and all(row["initial_camera_rgb_bit_exact"]) for row in results)
    record = {"teacher_gate": gate and alignment, "contact_gate": gate,
              "observation_alignment_gate": alignment, "cases": results,
              "historical_difference": "Historical inference observed before the solver interval and installed after it; "
                                       "teacher recorded at the command issue boundary. Historical step rounding shifted "
                                       "some solver ticks by 0.5 ms."}
    atomic(output, record)
    return record


def run_trace(assets, worker, raw, native):
    from .evaluate_motor import action_request
    from .motor import SENSORS, TASK

    source = paths(raw)[0]
    meta, arrays, frames = read_teacher(source)
    world = reset_world(assets, meta)
    try:
        engine = ArmExecutor(world, "aligned-sample-v2")
        engine.prime()
        request = action_request(world)
        report = worker.request({**request, "type": "audit_row", "split": "train", "index": 0,
                                 "split_root": str(native / "train")})
        decoded = engine.submit(report["first_decoded_native_action"])
        report.update(recorded_requested=arrays["requested"][0].tolist(), decoded_exec=decoded,
                      recorded_state=arrays["state"][0].tolist(), live_state=request["state"],
                      archive_frame_hashes={key: sha256(value.tobytes()).hexdigest() for key, value in frames.items()},
                      native_camera_order=list(SENSORS), native_joint_order=list(ARM_NAMES),
                      native_units=list(ARM_UNITS), fixed_task=TASK, recording=str(source),
                      control_period_ns_nominal="33333333", sampling_tick_ns=str(arrays["timestamps_ns"][0]),
                      next_observed_tick_ns=str(world.ns), learned_request_is_teacher=False)
        return report
    finally:
        world.close()


def offline_action_report(worker, dataset):
    phases = (("approach", 15), ("descend", 75), ("grasp", 135), ("carry", 220),
              ("transfer", 290), ("release", 405), ("withdraw", 490))
    rows = []
    for split, episodes in (("train", 3), ("validation", 2)):
        indices = [(stage, tick + group*510) for group in range(episodes) for stage, tick in phases]
        request = {"type": "offline_actions", "split": split, "split_root": str(dataset / split), "rows": indices}
        rows.extend({"split": split, **item} for item in worker.request(request)["rows"])
    return rows


def learned_case(assets, worker, source, profile, *, trace=None):
    meta, arrays, frames = read_teacher(source)
    match_visual(meta, worker.visual)
    world = reset_world(assets, meta)
    try:
        worker.request({"type": "reset", "seed": meta["seed"]})
        engine = ArmExecutor(world, profile)
        engine.prime()
        feedback = []
        for tick in range(meta["sample_count"]):
            observed = world.state(ARM)[0]
            at_issue = world.ns
            control = issue_action(world, worker, engine, tick, trace)
            if tick % 15 == 14:
                feedback.append({**observe(world, tick + 1), **control, "previous_observed_state": observed,
                                 "observed_at_ns": at_issue, "next_observed_at_ns": world.ns,
                                 "previous_applied_end_of_interval": world.arm_from_mj(world.filtered).tolist(),
                                 "teacher_applied_end_of_interval": arrays["applied_previous"][min(tick + 1, meta["sample_count"] - 1)].tolist(),
                                 "teacher_requested_at_index": arrays["requested"][tick].tolist()})
        return {"seed": meta["seed"], "profile": profile, **summarize(feedback),
                **({"act_input_trace": trace.report()} if trace is not None else {})}
    finally:
        world.close()


def act_development(assets, raw, native, weights, output):
    if not load(output / "teacher-replays.json")["teacher_gate"]:
        raise ValueError("Five successful teach-and-replay trajectories are required")
    diagnostics = Worker(weights, native / "validation", output / "act-diagnostic-worker.log")
    record = {"checkpoint": str(weights), "historical_evaluations_reused_for_model_choice": False,
              "dataset": str(native), "cases": [], "status": "in_progress"}
    try:
        record["worker"] = diagnostics.recv()
        record["sample_trace"] = run_trace(assets, diagnostics, raw, native)
        record["offline_native_action_errors"] = offline_action_report(diagnostics, native)
        record["status"] = "trace_complete"
        atomic(output / "act-development-diagnostic.json", record)
        for source in paths(raw):
            record["cases"].append(learned_case(assets, diagnostics, source, "aligned-sample-v2"))
            atomic(output / "act-development-diagnostic.json", record)
        record["cases"].append(learned_case(assets, diagnostics, paths(raw)[0], "historical-eval-v1"))
        record["status"] = "complete"
    finally:
        diagnostics.close()
        atomic(output / "act-development-diagnostic.json", record)
    return record


def smolvla_development(assets, raw, native, weights, output):
    if not load(output / "teacher-replays.json")["teacher_gate"]:
        raise ValueError("Five successful teach-and-replay trajectories are required")
    act = load(output / "act-development-diagnostic.json")
    if act["status"] != "complete" or any(act["sample_trace"]["preprocessing_max_abs_residual"].values()):
        raise ValueError("Shared ACT data and executor path has not been verified")
    destination = output / "smolvla-development-comparison.json"
    worker = Worker(weights, native / "validation", output / "smolvla-comparison-worker.log")
    record = {"checkpoint": str(weights), "dataset": str(native), "status": "in_progress", "cases": [],
              "protocol": "Same five matching development seeds and aligned-sample-v2 as ACT",
              "new_optimizer_updates": 0}
    try:
        record["worker"] = worker.recv(timeout=180)
        for source in paths(raw):
            record["cases"].append(learned_case(assets, worker, source, "aligned-sample-v2"))
            atomic(destination, record)
        record["status"] = "complete"
    finally:
        worker.close()
        atomic(destination, record)
    return record
