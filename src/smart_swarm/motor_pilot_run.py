"""One bounded ACT fitting decision before larger motor optimization."""

import json
from pathlib import Path
import re
import time
import uuid

from .episode import atomic, load
from .arm_executor import ArmExecutor
from .contract import ProtocolError
from .evaluate_motor import Worker, action_request
from .motor_diagnostics import contact_valid_placement, learned_case, read_teacher, reset_world
from .motor_input import CLOCKED_PROFILE, PROFILE
from .research_jobs import admitted
from .training import ROOT, run_motor

REPORT = ROOT / "evidence/learning-pilot-v3"


def optimizer_rows(path):
    pattern = r"step:\s*(\S+)\s+smpl:[^\s]+ .*?loss:\s*(\d+(?:\.\d+)?)"
    log = path.read_text() if path.is_file() else ""
    matches = re.findall(pattern, log)
    return [{"update": int(step) if step.isdecimal() else index, "reported_step": step, "loss": float(value)}
            for index, (step, value) in enumerate(matches, start=1)]


def trace_fit(config, worker, dataset, first):
    manifest, arrays, _ = read_teacher(Path(first["raw_path"]))
    world = reset_world(config["asset_root"], manifest)
    marker = manifest.get("target_marker", "cyan")
    try:
        executor = ArmExecutor(world, "aligned-sample-v2")
        executor.prime()
        observed_at = world.ns
        profile = worker.input_profile
        index = executor.observation_index() if profile == CLOCKED_PROFILE else None
        payload = action_request(world, profile, marker, issue_index=index)
        trace = worker.request({**payload, "type": "audit_row", "split": "train", "index": 0,
                                "split_root": str(dataset / "train")})
        targets = [action_request(world, profile, item, issue_index=index) for item in ("cyan", "violet")]
        target_probe = worker.request({"type": "audit_conditioning", "targets": targets})
        if index is not None:
            same_target = [action_request(world, profile, marker, issue_index=tick) for tick in (index, index + 17)]
            trace["counterfactual_issue_index"] = worker.request({"type": "audit_conditioning", "targets": same_target})
        dispatched = executor.submit(trace["first_decoded_native_action"])
        trace.update(counterfactual_target=target_probe, decoded_executor=dispatched,
                     measured_at_ns=observed_at, recorded_at_ns=int(arrays["timestamps_ns"][0]),
                     teacher_requested=arrays["requested"][0].tolist(), pipeline_profile=profile)
        return trace
    finally:
        world.close()


def fitting_rollouts(config, dataset, checkpoint, lineage, root, deadline, heartbeat):
    worker = Worker(checkpoint, dataset / "validation", root / "fitting-worker.log", deadline=deadline)
    results = []
    try:
        worker_status = worker.recv(timeout=90)
        worker_status["matched_sample_trace"] = trace_fit(config, worker, dataset, lineage["episodes"][0])
        samples = [(phase, idx) for phase, idx in (("approach", 15), ("grasp", 135), ("carry", 220),
                                                  ("release", 405), ("withdraw", 490))]
        request = {"type": "offline_actions", "split": "train", "split_root": str(dataset / "train"), "rows": samples}
        worker_status["offline_native_action_error"] = worker.request(request)["rows"]
        for row in lineage["episodes"]:
            if time.monotonic() < deadline:
                heartbeat()
                case = learned_case(config["asset_root"], worker, Path(row["raw_path"]), "aligned-sample-v2")
                results.append(case)
                atomic(root / "fitting-cases.json", results)
        return worker_status, results
    finally:
        worker.close()


def clock_fit_budget(training):
    prior = load(REPORT / "act-fit.json")
    audit = load(REPORT / "issue-clock-v4-parity.json")
    if prior["fit_gate"] or audit["input_profile"] != CLOCKED_PROFILE or not prior.get("checkpoint"):
        raise ValueError("The clock fit requires the preserved failed fit and verified clock parity")
    previous_steps, previous_wall = prior["optimizer_updates"], prior["elapsed_s"]
    wall = 1800 - previous_wall
    steps = min(200, 1000 - previous_steps, int((wall - 135) / 2.05))
    if steps < 100:
        raise ValueError("Too little of the original fitting allocation remains for the diagnosed correction")
    training["profiles"]["act-clock-fit"] = {"type": "act", "steps": steps, "batch_size": 4, "save_freq": 100,
                                              "wall_seconds": int(wall - 135), "initial_checkpoint": prior["checkpoint"]}
    return {"elapsed_s": previous_wall, "optimizer_updates": previous_steps, "checkpoint": prior["checkpoint"]}


def run_fit(config, dataset, *, clocked=False):
    destination = REPORT / ("act-clock-fit.json" if clocked else "act-fit.json")
    if destination.is_file():
        return load(destination)
    dataset = dataset.resolve()
    train = load(dataset / "train/smart-swarm-lineage.json")
    validation = load(dataset / "validation/smart-swarm-lineage.json")
    profile = CLOCKED_PROFILE if clocked else PROFILE
    if len(train["episodes"]) != 5 or any(row["input_profile"] != profile for row in (train, validation)):
        raise ValueError("Fit requires exactly five verified issue-time aligned trajectories")
    recipe = "act-clock-fit" if clocked else "act-five-fit"
    root = config["data_root"] / "training-runs" / (recipe + "-" + uuid.uuid4().hex[:12])
    training = load(ROOT / "configs/training.json")
    prior = clock_fit_budget(training) if clocked else {"elapsed_s": 0, "optimizer_updates": 0}
    REPORT.mkdir(exist_ok=True, parents=True)
    with admitted(config, root.name, archive_only=bool(config.get("native_cpu_admission"))) as (grant, heartbeat):
        if destination.is_file():
            return load(destination)
        return run_fit_job(config, dataset, root, training, train, validation, grant, heartbeat,
                           recipe=recipe, destination=destination, prior=prior)


def run_fit_job(config, dataset, root, training, train, validation, grant, heartbeat, *, recipe, destination, prior):
    started = time.monotonic()
    deadline = started + 1800 - prior["elapsed_s"]
    spec = training["profiles"][recipe]
    record = {"job_directory": str(root), "stage": "fitting", "dataset": str(dataset),
              "optimizer_update_cap": spec["steps"], "combined_fit_wall_cap_s": 1800, "required_successes": 4,
              "input_profile": train["input_profile"], "previous_fit": prior}
    atomic(destination, record)
    try:
        run_motor(training, recipe, dataset, root, train, validation, grant, heartbeat)
    except RuntimeError as error:
        record["optimizer_stop_reason"] = str(error)
    history = optimizer_rows(root / "training.log")
    record.update(optimizer_updates=history[-1]["update"] if history else 0, training_loss=history)
    evaluate_fit(config, dataset, root, train, record, deadline, heartbeat)
    record["elapsed_s"] = time.monotonic() - started
    record["cumulative_fit_wall_s"] = prior["elapsed_s"] + record["elapsed_s"]
    record["cumulative_fit_updates"] = prior["optimizer_updates"] + record["optimizer_updates"]
    record["stage"] = "passed" if record["fit_gate"] else "failed_or_inconclusive"
    atomic(destination, record)
    return record


def evaluate_fit(config, dataset, root, train, record, deadline, heartbeat):
    checkpoint = root / "training/checkpoints/last/pretrained_model"
    record.update(fit_gate=False, matching_starts=0, successful=0)
    try:
        if (checkpoint / "model.safetensors").is_file() and time.monotonic() < deadline:
            atomic(checkpoint / "smart-swarm-input.json", {"input_profile": train["input_profile"], "dataset": str(dataset)})
            record["checkpoint"] = str(checkpoint.resolve())
            worker, samples = fitting_rollouts(config, dataset, checkpoint, train, root, deadline, heartbeat)
            count = sum(contact_valid_placement(row) for row in samples)
            record.update(matching_starts=len(samples), successful=count, validation=worker, fitting_cases=samples,
                          fit_gate=len(samples) == 5 and count >= 4)
    except (OSError, RuntimeError, ValueError, TimeoutError, ProtocolError) as error:
        record["evaluation_failure"] = str(error)
