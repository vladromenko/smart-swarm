"""Closed-loop contact trials with a freshly reloaded offline worker."""

from base64 import b64encode
from collections import deque
from hashlib import sha256
import json
import os
from pathlib import Path
import select
import subprocess
import time
import uuid

import numpy as np

from .arm_executor import ArmExecutor
from .episode import atomic, load
from .motor import ARM, IMAGE_SIZE, SENSORS, TASK, variation
from .motor_renderer import MotorWorld
from .motor_visual_profile import COMPACT, match_visual
from .teacher import truth
from .training import ROOT, offline


def bound_input_profile(checkpoint, validation):
    from .motor_input import CLOCKED_PROFILE, PROFILE

    profile_path = checkpoint / "smart-swarm-input.json"
    split_path = validation / "smart-swarm-lineage.json"
    baseline = "historical-rgb"
    profile = load(profile_path)["input_profile"] if profile_path.is_file() else baseline
    split = load(split_path) if split_path.is_file() else {}
    exported = split.get("input_profile", baseline)
    if profile not in (baseline, PROFILE, CLOCKED_PROFILE) or profile != exported:
        raise ValueError(f"Checkpoint input profile {profile!r} requires matching validation inputs. Found {exported!r}.")
    if profile != baseline:
        camera_keys = {"wrist": "observation.images.camera1", "top": "observation.images.camera2"}
        if split.get("control_timing") != "aligned-sample-v2" or split.get("allowed_images") != camera_keys:
            raise ValueError("The conditioned validation split must preserve the exported issue timing and camera roles")
    return profile


def bound_visual_profile(checkpoint, validation):
    receipt = checkpoint / "smart-swarm-input.json"
    lineage = validation / "smart-swarm-lineage.json"
    return match_visual(load(receipt) if receipt.is_file() else {}, load(lineage) if lineage.is_file() else {})


class Worker:
    def __init__(self, checkpoint, validation, logfile, *, deadline=None, full_validation=False, smolvla_chunks=False):
        self.deadline = deadline
        self.pending_actions = deque()
        self.smolvla_chunks = smolvla_chunks
        self.input_profile = bound_input_profile(checkpoint, validation)
        self.visual = bound_visual_profile(checkpoint, validation)
        config = load(ROOT / "configs/training.json")
        receive, send = os.pipe()
        with logfile.open("w") as stream:
            argv = [config["runtime"], str(ROOT / "scripts/policy_worker.py"), "--fd", str(send),
                    "--checkpoint", str(checkpoint), "--validation", str(validation)]
            if full_validation:
                argv.append("--all-validation-sources")
            try:
                self.process = subprocess.Popen(argv, env=offline(), stdin=subprocess.PIPE, text=True,
                                                stdout=stream, stderr=subprocess.STDOUT, pass_fds=(send,))
            except Exception:
                os.close(receive)
                raise
            finally:
                os.close(send)
        self.replies = os.fdopen(receive)

    def recv(self, timeout=90):
        if self.deadline is not None:
            timeout = min(timeout, max(0., self.deadline - time.monotonic()))
        if not select.select([self.replies], [], [], timeout)[0]:
            raise TimeoutError("Local CPU policy worker did not answer before its deadline")
        line = self.replies.readline()
        if not line:
            raise RuntimeError("Checkpoint worker exited; inspect policy-worker.log")
        return json.loads(line)

    def request(self, data):
        if data["type"] == "reset":
            self.pending_actions.clear()
        self.process.stdin.write(json.dumps(data, allow_nan=False) + "\n")
        self.process.stdin.flush()
        return self.recv()

    def close(self):
        self.process.stdin.close()
        try:
            self.process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait(timeout=5)
        finally:
            self.replies.close()


def action_request(world, input_profile="historical-rgb", marker="cyan", *, issue_index=None):
    from .motor_input import CLOCKED_PROFILE, PROFILE, conditioned_top, instruction

    private = world.views(ARM)
    text = TASK
    if input_profile in (PROFILE, CLOCKED_PROFILE):
        if input_profile == CLOCKED_PROFILE and issue_index is None:
            raise ValueError("Clock-conditioned inference requires the local issue index")
        index = issue_index if input_profile == CLOCKED_PROFILE else None
        private[SENSORS[1]] = conditioned_top(private[SENSORS[1]], marker, issue_index=index)
        text = instruction(TASK, marker)
    return {"type": "action", "state": world.state(ARM)[0], "task": text,
            "images": [b64encode(private[sensor].tobytes()).decode("ascii") for sensor in SENSORS]}


def action(world, worker, executor=None):
    from .motor_input import CLOCKED_PROFILE

    if worker.deadline is not None and time.monotonic() >= worker.deadline:
        raise TimeoutError("The closed-loop action budget has expired")
    if worker.pending_actions:
        return worker.pending_actions.popleft()
    marker = getattr(world, "motor_requested_marker", "cyan")
    index = executor.observation_index() if worker.input_profile == CLOCKED_PROFILE and executor else None
    request = action_request(world, worker.input_profile, marker, issue_index=index)
    if worker.smolvla_chunks:
        worker.pending_actions.extend(worker.request({**request, "type": "smolvla_chunk"})["actions"])
        return worker.pending_actions.popleft()
    return worker.request(request)["action"]


def run_trial(assets, worker, seed, bound, wall_seconds, profile):
    from .motor_diagnostics import contact_valid_placement, observe, summarize
    from .motor_input import CLOCKED_PROFILE, PROFILE

    swapped = worker.input_profile in (PROFILE, CLOCKED_PROFILE) and bool(seed % 2)
    world = MotorWorld(assets, seed=seed, frame_size=IMAGE_SIZE, timeout_s=wall_seconds, swap_markers=swapped,
                       visual_profile=worker.visual["visual_profile"])
    world.motor_requested_marker = "violet" if swapped else "cyan"
    score, feedback = 0, []
    try:
        recipe = variation(world)
        worker.request({"type": "reset", "seed": seed})
        world.step(200)
        executor = ArmExecutor(world, profile)
        executor.prime()
        for idx in range(bound):
            executor.submit(action(world, worker, executor))
            if idx % 15 == 14:
                label = observe(world, idx + 1)
                success = label["released"] and label["inside_carrier"] and label["stationary"]
                score += int(success)
                feedback.append(label)
        success = (len(feedback) >= 2 and all(row["released"] and row["inside_carrier"]
                   and row["stationary"] for row in feedback[-2:]))
        summary = summarize(feedback)
        return {"seed": seed, "policy_reset_seed": seed, "recipe": recipe,
                "target_marker": world.motor_requested_marker,
                "steps": bound, "physics_steps": world.physics_steps,
                "contained_and_released": success, "goal_observed_samples": score,
                "contact_valid_placement": contact_valid_placement(summary),
                "first_failed_stage": summary["first_failed_stage"],
                "maximum_sampled_lift_m": max(row["cube_height_above_table_m"] for row in feedback),
                "final_truth": truth(world), "sampled_evaluator_truth": feedback}
    finally:
        world.close()


def evaluation_record(checkpoint, dataset, profile, seed_start, role, count, reservation):
    visual = bound_visual_profile(checkpoint, dataset / "validation")
    if visual["visual_profile"] == COMPACT and (reservation or {}).get("gpu_render"):
        raise ValueError("The compact RGB profile is bound to the verified software renderer")
    with (checkpoint / "model.safetensors").open("rb") as source:
        weight_hash = sha256_stream(source)
    return {"run_mode": "mujoco", "checkpoint": str(checkpoint), "checkpoint_sha256": weight_hash,
            "executor_profile": profile, "seed_start": seed_start, "evaluation_role": role,
            "requested_resets": count, "results": [], "status": "evaluating", "reservation": reservation, **visual}


def evaluate(assets, checkpoint, dataset, base, count, steps, wall_seconds, profile="aligned-sample-v2", seed_start=31_000,
             *, role="development", heartbeat=None, reservation=None, smolvla_chunks=False):
    if not 1 <= count <= 20 or not 30 <= steps <= 600 or not 30 <= wall_seconds <= 7200:
        raise ValueError("Invalid bounded evaluation profile")
    root = base / f"motor-pilot-{uuid.uuid4().hex[:12]}"
    root.mkdir(exist_ok=False, parents=True)
    started = time.monotonic()
    record = evaluation_record(checkpoint, dataset, profile, seed_start, role, count, reservation)
    if smolvla_chunks:
        record["action_transport"] = "verified-smolvla-chunk"
    atomic(root / "evaluation.json", record)
    worker = None
    try:
        worker = Worker(checkpoint, dataset / "validation", root / "policy-worker.log", deadline=started + wall_seconds,
                        full_validation=role == "fresh-test", smolvla_chunks=smolvla_chunks)
        record["input_profile"] = worker.input_profile
        record.update(worker.recv(timeout=min(480, wall_seconds)))
        for idx in range(count):
            if heartbeat is not None:
                heartbeat()
            remaining = wall_seconds - (time.monotonic() - started)
            if remaining <= 1:
                raise TimeoutError("Closed-loop evaluation wall-time cap reached")
            record["results"].append(run_trial(assets, worker, seed_start + idx, steps, remaining, profile))
            atomic(root / "evaluation.json", record)
        record["status"] = "evaluated"
    except (Exception, KeyboardInterrupt) as error:
        record.update(status="partial", reason=str(error) or type(error).__name__)
        raise
    finally:
        try:
            if worker is not None:
                worker.close()
        finally:
            record["successful_resets"] = sum(row["contact_valid_placement"] for row in record["results"])
            atomic(root / "evaluation.json", record)
    return {"evaluation_path": str(root / "evaluation.json"), **record}


def sha256_stream(stream):
    digest = sha256()
    for block in iter(lambda: stream.read(1024**2), b""):
        digest.update(block)
    return digest.hexdigest()
