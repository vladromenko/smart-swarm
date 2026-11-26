"""Compare a stock and contact-free ACT observation under one renderer."""

import argparse
from base64 import b64decode
from hashlib import sha256
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from time import monotonic

import numpy as np

from probe_precontact_act import (ARCHIVED_CLOCK, MATCHED_RENDER_CONTROL, PHYSICS, STATIC_RGB, WALL_CAP_S,
                                 captured_requests, qualified_sources, recorded_charges)
from smart_swarm.cli import settings
from smart_swarm.episode import atomic, json_bytes, load, sha
from smart_swarm.motor_input import CLOCKED_PROFILE
from smart_swarm.rendering import configure_renderer
from smart_swarm.research_jobs import admitted, execute
from smart_swarm.training import offline

STOCK_ISSUES = (120, 126, 132)
GRIPPER_PARITY_PP = .0002
OPEN_PERCENT = 25.
CLOSED_PERCENT = 5.
LOG = MATCHED_RENDER_CONTROL.with_suffix(".worker.log")


def reference():
    checkpoint, dataset, entries, sources = qualified_sources()
    entry = entries[0]
    raw = Path(entry["raw_path"])
    manifest = load(raw / "manifest.json")
    source = sources[manifest["seed"]]
    old = next(row for row in load(ARCHIVED_CLOCK)["sources"] if row["seed"] == source["seed"])
    if (sha(raw / "motor.npz") != entry["source_sha256"] or old["source_sha256"] != source["source_sha256"]
            or manifest["sha256"] != old["source_sha256"] or not manifest["physics"]["released_stable_containment"]
            or any(branch["close_issue"] != STOCK_ISSUES[0] for branch in source["implementations"].values())):
        raise ValueError("An original contact-qualified source or command boundary changed")
    controls = {row["issue"]: next(item for item in row["clock_conditions"] if item["offset_issues"] == 0)
                for row in old["samples"]}
    return checkpoint, dataset, raw, manifest, entry["target_marker"], source, controls


def captures(world, geometry, marker):
    from check_precontact_teacher import original_command_path
    from smart_swarm.precontact_teacher import PRECONTACT, STOCK

    snapshot = world.snapshot()
    pickup = geometry["pickup"]
    result = {}
    for name, pose, issues in (("stock", STOCK, STOCK_ISSUES), ("contact_free", PRECONTACT, STOCK_ISSUES[:1])):
        source = geometry["implementations"]["stock" if name == "stock" else "raised_and_retreated"]
        world.restore(snapshot)
        trace = original_command_path(world, pickup, pose, issues=source["matched_actuator_issues"])
        if sha256(json_bytes(trace.commands)).hexdigest() != source["request_sha256"]:
            raise ValueError("A matched teacher command trace changed")
        world.restore(snapshot)
        result[name] = {issue: (stamp, payload) for issue, stamp, payload in
                        captured_requests(world, trace.commands, marker, issues)}
    return result


def archive_inputs(raw, manifest, captures):
    from export_lerobot import load_episode, make_frame
    from profile_motor_device import packet

    arrays = load_episode(raw, manifest, aligned=True)
    with np.load(raw / "motor.npz", allow_pickle=False) as recording:
        stamps = recording["timestamps_ns"]
    inputs = {}
    for issue in STOCK_ISSUES:
        stamp, request = captures["stock"][issue]
        packed = packet(issue, make_frame(arrays, manifest, issue, CLOCKED_PROFILE))["payload"]
        residual = float(np.max(np.abs(np.asarray(request["state"]) - packed["state"])))
        if residual > 1e-5 or stamp != int(stamps[issue]) or request["task"] != packed["task"]:
            raise ValueError("Recreated stock state, timing or target differs from the original sample")
        inputs[issue] = ({"type": "action", **packed}, residual)
    return inputs


def candidate(worker, request):
    worker.request({"type": "reset"})
    action = worker.request(request)["action"]
    if len(action) != 6 or not np.isfinite(action).all():
        raise ValueError("The frozen ACT worker did not return six finite native commands")
    grip = action[-1]
    band = "close" if grip <= CLOSED_PERCENT else "open" if grip > OPEN_PERCENT else "intermediate"
    return {"decoded_action_native": action, "gripper_band": band, "gripper_percent": grip,
            "input_rgb_sha256": [sha256(b64decode(value, validate=True)).hexdigest() for value in request["images"]]}


def compare(worker, captured, archived, controls):
    rows = []
    for issue in STOCK_ISSUES:
        stamp, same_renderer_request = captured["stock"][issue]
        archived_request, state_error = archived[issue]
        archived_response = candidate(worker, archived_request)
        rendered_response = candidate(worker, same_renderer_request)
        saved = controls.get(issue)
        delta = abs(archived_response["gripper_percent"] - saved["first_gripper_percent"]) if saved else None
        if delta is not None and delta > GRIPPER_PARITY_PP:
            raise ValueError("The exact archived gripper response changed on reload")
        rows.append({"issue": issue, "time_ns": stamp, "archived_stock": archived_response, "rendered_stock": rendered_response,
                     "stock_state_residual_native": state_error, "archived_model_response_residual_pp": delta,
                     "rgb_byte_match": [old == current for old, current in zip(archived_response["input_rgb_sha256"],
                                                                               rendered_response["input_rgb_sha256"], strict=True)]})
    matched = all(row["archived_stock"]["gripper_band"] == row["rendered_stock"]["gripper_band"] == band
                  for row, band in zip(rows, ("open", "open", "close"), strict=True))
    issue = STOCK_ISSUES[0]
    at, request = captured["contact_free"][issue]
    if at != rows[0]["time_ns"]:
        raise ValueError("The two geometry branches changed the closure issue time")
    adjusted = {"issue": issue, "time_ns": at, **candidate(worker, request)}
    return rows, matched, adjusted


def measure(config, deadline):
    configure_renderer(config)
    from OpenGL import GL
    from smart_swarm.evaluate_motor import Worker
    from smart_swarm.motor_diagnostics import reset_world

    checkpoint, dataset, raw, meta, marker, source, controls = reference()
    world = reset_world(config["asset_root"], meta)
    world.deadline = deadline
    try:
        captured = captures(world, {**source, "pickup": meta["physics"]["pickup_xy_m"]}, marker)
        gl = GL.glGetString(GL.GL_RENDERER).decode()
        if "llvmpipe" not in gl.lower():
            raise ValueError("The two branches must use the verified software renderer")
    finally:
        world.close()
    archived = archive_inputs(raw, meta, captured)
    worker = Worker(checkpoint, dataset / "validation", LOG, deadline=deadline)
    try:
        worker.recv()
        stock, agrees, adjusted = compare(worker, captured, archived, controls)
    finally:
        worker.close()
    return {"status": "measured", "seed": source["seed"], "renderer": gl, "teacher_source_sha256": source["source_sha256"],
            "stock_controls": stock, "stock_gripper_band_controls_passed": agrees, "adjusted_at_close_issue": adjusted,
            "issued_learned_actions": 0, "optimizer_updates": 0, "physical_motion_requested": False}


def bounded(config):
    charges = recorded_charges()
    remaining = WALL_CAP_S - sum(charges.values())
    if not STATIC_RGB.is_file() or remaining <= 1:
        raise ValueError("The original renderer mismatch and remaining diagnostic budget are required")
    with TemporaryDirectory(prefix="smart-swarm-render-control-") as temporary:
        payload = Path(temporary) / "worker.json"
        with admitted(config, "smart-swarm-act-matched-renderer-precontact") as (grant, heartbeat):
            started = monotonic()
            deadline = started + remaining
            command = [sys.executable, str(Path(__file__).resolve()), "--worker", str(payload), "--deadline", str(deadline - 1.)]
            try:
                status = execute(command, offline(), sys.stderr, deadline, heartbeat)
                report = load(payload) if status == 0 else {"status": "worker_failed", "exit_code": status}
            except (OSError, RuntimeError, TimeoutError, ValueError) as error:
                report = {"status": "worker_failed", "error": str(error)}
            charged = monotonic() - started
    return {**report, "scope": "One consumed development source, matched new renderer control with archived grip anchors",
            "original_rgb_gate_replaced": False, "stock_raster_mismatch_sha256": sha(STATIC_RGB),
            "geometry_receipt_sha256": sha(PHYSICS), "prior_charges_s": charges, "charged_seconds": charged,
            "max_run_seconds": remaining, "remaining_original_seconds": remaining - charged, "reservation": grant}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--deadline", type=float, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if MATCHED_RENDER_CONTROL.exists() or (args.worker is None and LOG.exists()):
        parser.error("Preserve the previous matched-render diagnostic")
    config = settings()
    if args.worker is None:
        atomic(MATCHED_RENDER_CONTROL, bounded(config))
        print(MATCHED_RENDER_CONTROL)
    else:
        atomic(args.worker, measure(config, args.deadline))


if __name__ == "__main__":
    main()
