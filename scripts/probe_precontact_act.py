#!/usr/bin/env python3
"""Reopen the paired geometry with live MuJoCo RGB and the frozen ACT worker."""

import argparse
from base64 import b64decode, b64encode
from hashlib import sha256
from pathlib import Path
from time import monotonic

import numpy as np

from smart_swarm.cli import ROOT, settings
from smart_swarm.episode import atomic, json_bytes, load, sha
from smart_swarm.motor_input import CLOCKED_PROFILE, conditioned_top, mark_issue_clock
from smart_swarm.rendering import configure_renderer
from smart_swarm.research_jobs import admitted

PILOT = ROOT / "evidence/learning-pilot-v3"
FIT = PILOT / "act-clock-fit.json"
PHYSICS = PILOT / "act-precontact-teacher-physics.json"
CHECKPOINT = PILOT / "act-clock-closure-window.json"
WALL_CAP_S = 180
PREVIOUS_CHARGE_S = 40
CLOCK_OFFSETS = (-15, 0, 15)
STOCK_CONTACT_CONTROL = 132
PREVIOUS_FAILURE = ROOT.parent.parent / "work/act-precontact-rgb-sequence.log"
PREVIOUS_WORKER = PILOT / "act-precontact-teacher-rgb.worker.log"
PREVIOUS_FAILURE_SHA = "4127541dca3ff78b34e67002903ab185b5512e1e244d898ca42eebe497273c15"
PREVIOUS_WORKER_SHA = "f63207b58c3fce8475f4b830ad545bb803ff04c3bd40d573b4f06b0e5daddc78"
OPTICAL_FAILURE = PILOT / "act-precontact-teacher-rgb-v2-failed.json"
STOCK_MISMATCH = PILOT / "act-precontact-stock-pixel-mismatch.json"
STATIC_RGB = PILOT / "act-stock-static-raster.json"
STATIC_RGB_BOOT_FAILURE = PILOT / "act-stock-static-raster-startup-failed.json"
ARCHIVED_CLOCK = PILOT / "act-archived-clock-offset.json"
MATCHED_RENDER_CONTROL = PILOT / "act-matched-render-precontact-control.json"
INPUT_PROBE_FAILURE_CHARGE_S = 15


def qualified_sources():
    if not PHYSICS.is_file():
        raise ValueError("The matched five-world contact-free geometry receipt is pending")
    receipt = load(PHYSICS)
    if not receipt["all_five_passed"]:
        raise ValueError("The same five worlds must pass the contact-free and placement geometry checks")
    fit = load(FIT)
    dataset, checkpoint = (Path(fit[name]) for name in ("dataset", "checkpoint"))
    lineage = dataset / "train/smart-swarm-lineage.json"
    if sha(lineage) != receipt["training_lineage_sha256"]:
        raise ValueError("The original teacher fitting cohort has changed")
    saved = load(CHECKPOINT)
    if any(sha(checkpoint / name) != checksum for name, checksum in saved["artifact_sha256"].items()):
        raise ValueError("The frozen ACT checkpoint or processor changed")
    sources = {case["seed"]: case for case in receipt["sources"]}
    return checkpoint, dataset, load(lineage)["episodes"], sources


def stock_input_parity(raw, marker, issue, request):
    with np.load(raw / "motor.npz", allow_pickle=False) as archive:
        expected = [archive["wrist"][issue],
                    conditioned_top(archive["top"][issue], marker, issue_index=issue)]
        if not np.allclose(request["state"], archive["state"][issue], atol=1e-5, rtol=0.):
            raise ValueError("Original ACT sampled joint state changed on replay")
    return [b64decode(text) == pixels.tobytes() for text, pixels in zip(request["images"], expected, strict=True)]


def captured_requests(world, commands, marker, indices):
    from smart_swarm.arm_executor import ArmExecutor
    from smart_swarm.evaluate_motor import action_request

    engine = ArmExecutor(world, "aligned-sample-v2")
    engine.prime()
    result = []
    for issue, action in enumerate(commands[:max(indices) + 1]):
        if issue in indices:
            if issue != engine.observation_index() or issue % 6:
                raise ValueError("The live motor sample is outside an ACT chunk refresh")
            request = action_request(world, CLOCKED_PROFILE, marker, issue_index=issue)
            result.append((issue, world.ns, request))
        engine.submit(action)
    return result


def altered_clock(request, issue, offset):
    if offset == 0:
        return request
    wrist, top = request["images"]
    original = np.frombuffer(b64decode(top, validate=True), dtype=np.uint8).reshape((120, 160, 3))
    changed = original.copy()
    mark_issue_clock(changed, issue + offset)
    if not np.any(changed != original):
        raise ValueError("The time intervention must change the issue clock pixels")
    delta = changed != original
    delta[2:14, 16:60] = False
    if np.any(delta):
        raise ValueError("The time intervention edited image content outside the clock strip")
    return {**request, "images": [wrist, b64encode(changed.tobytes()).decode("ascii")]}


def decode_variations(world, worker, issue, request):
    predictions = []
    current_time = world.ns
    for offset in CLOCK_OFFSETS:
        compared = altered_clock(request, issue, offset)
        worker.request({"type": "reset"})
        prediction = worker.request(compared)["action"]
        limited = world.arm_from_mj(world.arm_to_mj(prediction)).tolist()
        predictions.append({"offset_issues": offset, "visible_issue": issue + offset,
                            "encoded_top_sha256": sha256(b64decode(compared["images"][1])).hexdigest(),
                            "action_native": prediction, "would_apply_after_executor_limits_native": limited})
    if world.ns != current_time:
        raise ValueError("A timing-only intervention advanced the original simulation")
    return predictions


def one_variant(world, snapshot, pickup, marker, raw, name, reference, worker):
    from check_precontact_teacher import original_command_path
    from smart_swarm.precontact_teacher import PRECONTACT, STOCK

    world.restore(snapshot)
    pose = STOCK if name == "stock" else PRECONTACT
    trajectory = original_command_path(world, pickup, pose, issues=reference["matched_actuator_issues"])
    if sha256(json_bytes(trajectory.commands)).hexdigest() != reference["request_sha256"]:
        raise ValueError("The geometry does not reproduce the already checked trajectory")
    world.restore(snapshot)
    close_issue = reference["close_issue"]
    indices = (close_issue, STOCK_CONTACT_CONTROL) if name == "stock" else (close_issue,)
    frames = captured_requests(world, trajectory.commands, marker, indices)
    results = []
    for issue, at, request in frames:
        parity = stock_input_parity(raw, marker, issue, request) if name == "stock" else None
        if parity is not None and not all(parity):
            raise ValueError("Original wrist or top RGB bytes changed on the stock renderer")
        predictions = decode_variations(world, worker, issue, request)
        unchanged = next(row for row in predictions if row["offset_issues"] == 0)
        results.append({"issue": issue, "at_ns": at, "action_native": unchanged["action_native"],
                        "would_apply_after_executor_limits_native": unchanged["would_apply_after_executor_limits_native"],
                        "state_native": request["state"], "clock_variations": predictions,
                        "encoded_camera_sha256": [sha256(b64decode(text)).hexdigest() for text in request["images"]],
                        "original_stock_inputs_match": parity})
    return results


def one_source(assets, entry, geometry, worker):
    from smart_swarm.motor_diagnostics import read_teacher, reset_world

    path = Path(entry["raw_path"])
    manifest, _, _ = read_teacher(path)
    if manifest["sha256"] != geometry["source_sha256"]:
        raise ValueError("The optical rehearsal lost its original matched raw source")
    world = reset_world(assets, manifest)
    world.deadline = min(world.deadline, worker.deadline)
    snapshot = world.snapshot()
    try:
        pickup = manifest["physics"]["pickup_xy_m"]
        marker = manifest.get("target_marker", "cyan")
        traces = {name: one_variant(world, snapshot, pickup, marker, path, name, reference, worker)
                  for name, reference in geometry["implementations"].items()}
    finally:
        world.close()
    return {"seed": manifest["seed"], "source_sha256": manifest["sha256"], "teacher_inputs": traces}


def prior_failure():
    if sha(PREVIOUS_FAILURE) != PREVIOUS_FAILURE_SHA or sha(PREVIOUS_WORKER) != PREVIOUS_WORKER_SHA:
        raise ValueError("The first headless attempt and its budget provenance must remain available")
    return {"failure_log_sha256": PREVIOUS_FAILURE_SHA, "worker_log_sha256": PREVIOUS_WORKER_SHA,
            "charged_seconds": PREVIOUS_CHARGE_S, "observed_worker_file_to_failure_elapsed_s": 27.837,
            "reason": "MuJoCo imported before EGL configuration"}


def recorded_charges():
    charges = {"first_optical_failure": prior_failure()["charged_seconds"]}
    if OPTICAL_FAILURE.is_file():
        charges["second_optical_failure"] = load(OPTICAL_FAILURE)["conservative_wall_charge_seconds"]
    if STOCK_MISMATCH.is_file():
        charges.update(initial_renderer_input_error=INPUT_PROBE_FAILURE_CHARGE_S,
                       completed_stock_pixel_diagnostic=load(STOCK_MISMATCH)["elapsed_seconds"])
    if ARCHIVED_CLOCK.is_file():
        charges["archived_clock_replay"] = load(ARCHIVED_CLOCK)["current_replay_elapsed_s"]
    if STATIC_RGB_BOOT_FAILURE.is_file():
        charges["static_raster_startup_failure"] = load(STATIC_RGB_BOOT_FAILURE)["charged_seconds"]
    if STATIC_RGB.is_file():
        charges["static_raster_replay"] = load(STATIC_RGB)["charged_seconds"]
    if MATCHED_RENDER_CONTROL.is_file():
        charges["matched_render_precontact_control"] = load(MATCHED_RENDER_CONTROL)["charged_seconds"]
    return charges


def run(config, output, qualified):
    checkpoint, dataset, entries, sources = qualified
    log = output.with_suffix(".worker.log")
    cases = []
    original_failure = prior_failure()
    charges = recorded_charges()
    available = WALL_CAP_S - sum(charges.values())
    if available <= 0:
        raise ValueError("The original cumulative diagnostic allowance is exhausted")
    with admitted(config, "smart-swarm-frozen-act-precontact-rgb") as (grant, heartbeat):
        started = monotonic()
        configure_renderer(config)
        from mujoco.rendering.classic.gl_context import GLContext
        from smart_swarm.evaluate_motor import Worker

        if GLContext.__module__ != "mujoco.egl":
            raise RuntimeError("The precontact RGB check did not start with the configured EGL backend")
        worker = Worker(checkpoint, dataset / "validation", log, deadline=started + available)
        try:
            loaded = worker.recv()
            for entry in entries:
                heartbeat()
                seed = int(entry["source_world_episode_id"].rsplit("-", 1)[1])
                cases.append(one_source(config["asset_root"], entry, sources[seed], worker))
        finally:
            worker.close()
        elapsed = monotonic() - started
        if elapsed > available:
            raise TimeoutError("The rendered ACT diagnostic exceeded its remaining wall allowance")
    return {"scope": "Frozen ACT first action on matched simulation teacher observations",
            "geometry_receipt_sha256": sha(PHYSICS), "checkpoint_sha256": sha(checkpoint / "model.safetensors"),
            "reservation": grant, "worker_reload": loaded, "cases": cases,
            "ordered_cameras": ["so101-1/wrist", "so101-1/top"], "input_profile": CLOCKED_PROFILE,
            "optimizer_updates": 0, "learned_actions_executed": 0, "wall_cap_seconds": WALL_CAP_S,
            "previous_failed_attempt": original_failure, "previous_wall_charges_s": charges,
            "corrected_attempt_wall_seconds": elapsed,
            "remaining_wall_seconds": available - elapsed, "clock_offsets_issues": list(CLOCK_OFFSETS),
            "clock_scope": "A 0.5-second cue shift with the same RGB scene, pose, task and no intervening simulation step."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--output", type=Path, default=PILOT / "act-precontact-teacher-rgb-v2.json")
    args = parser.parse_args()
    if (args.output.exists() or args.output.with_suffix(".worker.log").exists()) and not args.check_only:
        parser.error("Choose a new precontact RGB result path")
    try:
        qualified = qualified_sources()
    except (ValueError, KeyError, OSError) as error:
        parser.error(str(error))
    if args.check_only:
        print("Five matched contact-valid geometries and the original frozen checkpoint are available.")
    else:
        atomic(args.output, run(settings(), args.output, qualified))
        print(args.output)


if __name__ == "__main__":
    main()
