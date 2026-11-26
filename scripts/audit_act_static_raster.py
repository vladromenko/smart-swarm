"""Localize stock ACT RGB drift on fixed station pixels before testing policy responses."""

import argparse
from base64 import b64decode
from hashlib import sha256
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from time import monotonic

import numpy as np

from probe_precontact_act import FIT, PHYSICS, STATIC_RGB, STOCK_MISMATCH, WALL_CAP_S, recorded_charges
from smart_swarm.cli import settings
from smart_swarm.episode import atomic, json_bytes, load, sha
from smart_swarm.motor_input import CLOCKED_PROFILE, conditioned_top
from smart_swarm.rendering import configure_renderer
from smart_swarm.research_jobs import admitted, execute
from smart_swarm.training import offline

ISSUES = (0, 120, 132)
LIMIT_SECONDS = 25.
IMAGE = (120, 160, 3)


def digest(array):
    return sha256(array.tobytes()).hexdigest()


def original_frames():
    dataset = Path(load(FIT)["dataset"])
    geometry = load(PHYSICS)
    diagnostic = load(STOCK_MISMATCH)
    entries = []
    for split, expected in (("train", 5), ("validation", 2)):
        binding = dataset / split / "smart-swarm-lineage.json"
        checked = load(binding)["episodes"]
        if len(checked) != expected or (split == "train" and sha(binding) != geometry["training_lineage_sha256"]):
            raise ValueError("The fixed original ACT development split changed")
        entries.extend(checked)
    first = Path(entries[0]["raw_path"])
    selected, static_samples = {}, []
    for entry in entries:
        source = Path(entry["raw_path"])
        manifest = load(source / "manifest.json")
        file = source / "motor.npz"
        if not (sha(file) == entry["source_sha256"] == manifest["sha256"]
                and manifest["physics"]["released_stable_containment"]):
            raise ValueError("An original teacher or contact-validity receipt changed")
        with np.load(file, allow_pickle=False) as archive:
            top = archive["top"][list(ISSUES)]
            static_samples.extend(top)
            if source == first:
                marker = entry["target_marker"]
                selected = {i: (wrist, conditioned_top(rgb, marker, issue_index=i), at, state)
                            for i, wrist, rgb, at, state in zip(ISSUES, archive["wrist"][list(ISSUES)], top,
                                                                archive["timestamps_ns"][list(ISSUES)],
                                                                archive["state"][list(ISSUES)], strict=True)}
    manifest = load(first / "manifest.json")
    if manifest["sha256"] != diagnostic["teacher_source_sha256"] or manifest["seed"] != diagnostic["seed"]:
        raise ValueError("The prior raster failure used a different archived teacher")
    stable = np.all(np.asarray(static_samples) == static_samples[0], axis=(0, 3))
    return manifest, entries, selected, stable


def capture(world, executor, issue, marker):
    import mujoco
    from OpenGL import GL
    from probe_motor_rendering import segmented
    from smart_swarm.evaluate_motor import action_request

    if executor.observation_index() != issue:
        raise ValueError("The new image is outside its original issue")
    request = action_request(world, CLOCKED_PROFILE, marker, issue_index=issue)
    wrist, top = (np.frombuffer(b64decode(value, validate=True), dtype=np.uint8).reshape(IMAGE) for value in request["images"])
    seg = segmented(world.renderer)
    station = (seg[..., 0] == world.model.geom("station_surface").id) & (seg[..., 1] == mujoco.mjtObj.mjOBJ_GEOM)
    name = world.cameras["so101-1"]["so101-1/top"]
    camera = world.model.camera(name).id
    pose = np.concatenate((world.data.cam_xpos[camera], world.data.cam_xmat[camera]))
    renderer = {key: GL.glGetString(enum).decode() for key, enum in
                (("vendor", GL.GL_VENDOR), ("renderer", GL.GL_RENDERER), ("version", GL.GL_VERSION))}
    return wrist, top, world.ns, np.asarray(request["state"]), station, pose, renderer


def live_frames(config, manifest, marker):
    from check_precontact_teacher import original_command_path
    from smart_swarm.arm_executor import ArmExecutor
    from smart_swarm.motor_diagnostics import reset_world
    from smart_swarm.precontact_teacher import STOCK

    world = reset_world(config["asset_root"], manifest)
    try:
        snapshot = world.snapshot()
        pickup = manifest["physics"]["pickup_xy_m"]
        path = original_command_path(world, pickup, STOCK, issues=max(ISSUES) + 1)
        stock = next(row["implementations"]["stock"] for row in load(PHYSICS)["sources"] if row["seed"] == manifest["seed"])
        if sha256(json_bytes(path.commands)).hexdigest() != stock["request_sha256"]:
            raise ValueError("The stock command recipe differs from the original contact audit")
        world.restore(snapshot)
        executor = ArmExecutor(world, "aligned-sample-v2")
        executor.prime()
        rows = {}
        for issue, command in enumerate(path.commands):
            if issue in ISSUES:
                rows[issue] = capture(world, executor, issue, marker)
            executor.submit(command)
        return rows
    finally:
        world.close()


def residual(archived, current, mask):
    delta = current[mask].astype(np.int16) - archived[mask].astype(np.int16)
    return {"pixels": len(delta), "changed_pixels": int(np.count_nonzero(np.any(delta != 0, axis=1))),
            "mean_absolute_rgb_residual": float(np.abs(delta).mean()), "maximum_rgb_residual": int(np.abs(delta).max()),
            "mean_signed_residual_by_channel": delta.mean(axis=0).tolist(),
            "first_pixel_archived_rgb": archived[mask][0].tolist(), "first_pixel_replay_rgb": current[mask][0].tolist()}


def station_interior(live, stable_reference):
    selected = np.logical_and.reduce([row[4] for row in live.values()])
    padded = np.pad(selected, 2, constant_values=False)
    height, width = selected.shape
    interior = np.logical_and.reduce([padded[y:y + height, x:x + width] for y in range(5) for x in range(5)])
    top = np.asarray([row[1] for row in live.values()])
    steady = np.all(top == top[0], axis=(0, 3))
    interior[2:14, 2:60] = False
    mask = interior & stable_reference & steady
    if mask.sum() < 500 or not all(np.array_equal(row[5], live[0][5]) for row in live.values()):
        raise ValueError("The overhead probe lacks a fixed camera and enough stable station-interior pixels")
    return mask


def sample_report(archived, live, mask):
    wrist, top, time, state, _, _, _ = live
    old_wrist, old_top, original_time, original_state = archived
    state_delta = float(np.max(np.abs(state - original_state)))
    if time != int(original_time) or state_delta > 1e-5:
        raise ValueError("The comparison no longer uses matched state and control time")
    cameras = {}
    for name, old, image in (("wrist", old_wrist, wrist), ("top", old_top, top)):
        cameras[name] = {**residual(old, image, np.ones(IMAGE[:2], dtype=bool)),
                         "archived_sha256": digest(old), "live_sha256": digest(image)}
    return {"issue_time_ns": time, "max_state_error_native": state_delta, "cameras": cameras,
            "overhead_stable_station_interior": residual(old_top, top, mask)}


def report(config):
    configure_renderer(config)
    import mujoco

    manifest, entries, archived, stable = original_frames()
    live = live_frames(config, manifest, entries[0]["target_marker"])
    mask = station_interior(live, stable)
    comparisons = [{"issue": issue, **sample_report(archived[issue], live[issue], mask)} for issue in ISSUES]
    return {"scope": "Fixed background versus full-image RGB under the matched ACT teacher control",
            "renderer": live[0][-1], "mujoco": mujoco.__version__, "source_sha256": manifest["sha256"],
            "seed": manifest["seed"], "archived_sources_for_stability": [row["source_world_episode_id"] for row in entries],
            "archived_sample_issues": list(ISSUES), "static_mask_sha256": digest(mask), "samples": comparisons,
            "stock_previous_receipt_sha256": sha(STOCK_MISMATCH), "probe_implementation_sha256": sha(Path(__file__)),
            "optimizer_updates": 0, "learned_actions_executed": 0, "physical_motion_requested": False}


def bounded(config):
    charges = recorded_charges()
    remaining = WALL_CAP_S - sum(charges.values())
    allowed = min(LIMIT_SECONDS, remaining)
    if allowed <= 0:
        raise ValueError("The original cumulative RGB diagnostic allowance is exhausted")
    with TemporaryDirectory(prefix="smart-swarm-static-raster-") as directory:
        result = Path(directory) / "worker.json"
        with admitted(config, "smart-swarm-act-static-raster") as (grant, heartbeat):
            started = monotonic()
            try:
                status = execute([sys.executable, str(Path(__file__).resolve()), "--worker", str(result)],
                                 offline(), sys.stderr, started + allowed, heartbeat)
                payload = load(result) if status == 0 else {"status": "worker_failed", "exit_code": status}
            except (OSError, RuntimeError, TimeoutError, ValueError) as error:
                payload = {"status": "worker_failed", "error": str(error)}
            elapsed = monotonic() - started
    return {"status": "measured", **payload, "reservation": grant, "prior_charges_s": charges,
            "max_run_seconds": allowed, "charged_seconds": elapsed, "remaining_original_seconds": remaining - elapsed}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    output = args.worker or STATIC_RGB
    if output.exists():
        parser.error("The static-raster diagnostic already has a retained outcome")
    atomic(output, report(settings()) if args.worker else bounded(settings()))
    print(output)


if __name__ == "__main__":
    main()
