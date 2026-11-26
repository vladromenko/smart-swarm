"""Generate bounded, contact-gated SO101 motor data with both allowed cameras."""

import hashlib
from pathlib import Path
import shutil
import time

import mujoco
import numpy as np

from .episode import atomic, load
from .motor_draw_cache import PROFILE as DRAW_CACHE_PROFILE
from .motor_input import FLOAT32_COMMAND_PROFILE
from .motor_renderer import MotorWorld
from .motor_visual_profile import COMPACT, STOCK, match_visual, visual_binding
from .physics import ARM_NAMES, ARM_UNITS
from .teacher import Teacher, run_insertion, truth

TASK = "Pick up the cube, place it in the accessible box, release, and withdraw."
IMAGE_SIZE = (160, 120)
ARM = "so101-1"
SENSORS = ("so101-1/wrist", "so101-1/top")
FPS = 30
SPLIT_OFFSETS = {"train": 10_000, "validation": 20_000, "test": 30_000}


def variation(world):
    offset = world.rng.uniform(-.004, .004, size=2)
    q = int(world.model.joint("cube_pose").qposadr[0])
    world.data.qpos[q:q + 2] += offset
    source = world.model.geom("cube_geom").id
    friction = float(world.rng.uniform(.94, 1.06))
    world.model.geom_friction[source, 0] = friction
    mujoco.mj_forward(world.model, world.data)
    return {"pickup_xy_m": world.data.qpos[q:q + 2].tolist(), "cube_sliding_friction": friction}


class MotorCapture:
    def __init__(self, world):
        self.world = world
        self.records = []
        self.frames = {name: [] for name in SENSORS}
        self.hold_samples = []
        self.peak = 0.

    def sample(self, phase, index, start, requested, applied):
        now = self.world.ns
        views = self.world.views(ARM)
        for name in SENSORS:
            self.frames[name].append(views[name])
        state = self.world.state(ARM)[0]
        self.records.append((now, start, state, requested, applied))
        if index % 10 == 9:
            outcome = truth(self.world)
            self.peak = max(self.peak, outcome["cube_height_above_table_m"])
            if phase in ("lift", "transfer", "lower"):
                bodies = outcome["gripper_contact_bodies"]
                self.hold_samples.append(len(bodies) >= 2)

    def save(self, path, seed, split, parameters):
        now, starts, measured, commanded, delayed = zip(*self.records, strict=True)
        np.savez(path / "motor.npz", wrist=self.frames[SENSORS[0]], top=self.frames[SENSORS[1]],
                 timestamps_ns=np.asarray(now, dtype=np.int64),
                 applied_interval_start_ns=np.asarray(starts, dtype=np.int64),
                 state=np.asarray(measured, dtype=np.float32), requested=np.asarray(commanded, dtype=np.float32),
                 applied_previous=np.asarray(delayed, dtype=np.float32))
        checksum = hashlib.sha256((path / "motor.npz").read_bytes()).hexdigest()
        entry = {"mode": "mujoco", "source": "scripted_simulation", "status": "accepted",
                 "world_episode_id": path.name, "scene_group_id": path.name, "seed": seed,
                 "lineage": {"split": split, "source_world_ids": []}, "fps": FPS,
                 "model_sha256": self.world.manifest["model_sha256"], "physics": parameters,
                 **self.world.motor_visual,
                 "target_marker": "violet" if getattr(self.world, "motor_swap_markers", False) else "cyan",
                 "action_schema_id": "sim-so101-absolute-v1", "names": ARM_NAMES, "units": ARM_UNITS,
                 "sensor_ids": SENSORS, "task": TASK, "sample_count": len(now), "sha256": checksum,
                 "action_label": "future requested setpoints from current observation, including control delay",
                 "previous_applied_values": "feedback from preceding interval, not action labels"}
        if self.world.motor_visual["visual_profile"] == COMPACT:
            entry["command_execution_profile"] = FLOAT32_COMMAND_PROFILE
        if hasattr(self.world, "motor_draw_cache_hit"):
            entry["drawing_context"] = {"profile": DRAW_CACHE_PROFILE, "reused": self.world.motor_draw_cache_hit}
        if hasattr(self.world, "motor_model_cache_hit"):
            from .motor_model_cache import PROFILE

            entry["compiled_model"] = {"profile": PROFILE, "reused": self.world.motor_model_cache_hit}
        atomic(path / "manifest.json", entry)
        (path / "attempt.json").unlink()

    def result(self):
        final = truth(self.world)
        success = (self.peak >= .1 and sum(self.hold_samples) >= 12 and final["inside_carrier"]
                   and final["released"] and final["stationary"])
        return success, {"peak_lift_m": self.peak, "bilateral_samples": sum(self.hold_samples),
                         "released_stable_containment": success, "final": final}


def record_insertion(world, sample, pickup):
    teacher = Teacher(world, native_float32=world.motor_visual["visual_profile"] == COMPACT)
    run_insertion(teacher, sample, pickup=pickup)


def attempt(assets, path, seed, split, remaining_s, *, visual_profile=STOCK, draw_cache=None, model_cache=None):
    path.mkdir(parents=True, exist_ok=False)
    atomic(path / "attempt.json", {"status": "recording", "seed": seed, "split": split})
    swapped = bool(seed % 2)
    world = None
    try:
        timeout = min(100, remaining_s)
        world = (model_cache.world(seed, timeout) if model_cache is not None else
                 MotorWorld(assets, seed=seed, frame_size=IMAGE_SIZE, timeout_s=timeout,
                            swap_markers=swapped, visual_profile=visual_profile))
        world.motor_swap_markers = swapped
        if draw_cache is not None:
            world.motor_draw_cache_hit = draw_cache.attach(world)
        config = variation(world)
        world.step(200)
        capture = MotorCapture(world)
        record_insertion(world, capture.sample, config["pickup_xy_m"])
        world.step(600)
        success, statistics = capture.result()
        if not success:
            atomic(path / "attempt.json", {"status": "rejected", "seed": seed, "physics": statistics})
            return False
        if draw_cache is not None:
            draw_cache.release(world)
        capture.save(path, seed, split, {**config, **statistics})
        return True
    except TimeoutError as error:
        atomic(path / "attempt.json", {"status": "timed_out", "seed": seed, "error": str(error)})
        return False
    except (Exception, KeyboardInterrupt) as error:
        atomic(path / "attempt.json", {"status": "quarantined", "seed": seed,
                                      "error": str(error) or type(error).__name__})
        raise
    finally:
        if world is not None:
            world.close()


def bind_capture_profile(root, visual_profile):
    binding = visual_binding(visual_profile)
    receipt = root / "motor-visual-profile.json"
    if receipt.is_file():
        match_visual(load(receipt), binding)
    for path in root.glob("*/*/manifest.json"):
        manifest = load(path)
        match_visual(manifest, binding)
        if visual_profile == COMPACT and manifest.get("command_execution_profile") != FLOAT32_COMMAND_PROFILE:
            raise ValueError(f"Compact source predates exact command recording: {path}")
    if not receipt.is_file() and visual_profile != STOCK:
        atomic(receipt, binding)
    return binding


def generate(assets: Path, root: Path, split: str, accepted_goal: int, wall_seconds: int, disk_mb: int, heartbeat=None,
             *, visual_profile=STOCK, draw_cache=None, model_cache=None):
    if not 0 < accepted_goal <= 1200 or not 1 <= wall_seconds <= 7200 or not 64 <= disk_mb <= 16_384:
        raise ValueError("Invalid motor generation budget")
    if model_cache is not None and visual_profile != COMPACT:
        raise ValueError("Only the compact visual profile may copy a pristine compiled template")
    deadline = time.monotonic() + wall_seconds
    root.mkdir(exist_ok=True, parents=True)
    binding = bind_capture_profile(root, visual_profile)
    offset = SPLIT_OFFSETS[split]
    completed, attempts = 0, 0
    limit = accepted_goal * 5
    folder = root / split
    while (completed < accepted_goal and attempts < limit and time.monotonic() < deadline
           and has_space(root, disk_mb)):
        seed = offset + attempts
        path = folder / f"so101-{seed}"
        if (path / "manifest.json").is_file():
            completed += load(path / "manifest.json")["status"] == "accepted"
        elif not path.exists():
            if heartbeat is not None:
                heartbeat()
            completed += attempt(assets, path, seed, split, deadline - time.monotonic(),
                                 visual_profile=visual_profile, draw_cache=draw_cache, model_cache=model_cache)
        attempts += 1
    summary = {"split": split, "attempted": attempts, "accepted": completed,
               "unaccepted": attempts - completed, "goal": accepted_goal, **binding,
               "status": "complete" if completed == accepted_goal else "incomplete"}
    atomic(root / f"{split}-generation.json", summary)
    return summary


def interrupted_summary(root, split, goal):
    accepted, other = 0, []
    for path in sorted((root / split).glob("so101-*")):
        manifest, attempt = path / "manifest.json", path / "attempt.json"
        if manifest.is_file() and load(manifest)["status"] == "accepted":
            accepted += 1
        elif attempt.is_file():
            other.append({"source_episode_id": path.name, **load(attempt)})
    return {"split": split, "attempted": accepted + len(other), "accepted": accepted,
            "unaccepted": len(other), "unaccepted_attempts": other, "goal": goal,
            "status": "incomplete", "stop_reason": "interrupted before completing the split"}


def has_space(root, disk_mb):
    bytes_used = sum(path.stat().st_size for path in root.glob("*/*/motor.npz"))
    per_attempt = 64 * 1024**2
    return bytes_used + per_attempt <= disk_mb * 1024**2 and shutil.disk_usage(root).free > per_attempt
