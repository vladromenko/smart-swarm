"""One validation-selected ACT continuation and one matched SmolVLA comparison."""

from pathlib import Path
import math
import shutil
import time
import uuid

from .episode import atomic, load
from .contract import ProtocolError
from .evaluate_motor import Worker, evaluate
from .motor_diagnostics import contact_valid_placement, learned_case
from .motor_input import CLOCKED_PROFILE, PROFILE
from .motor_pilot_run import optimizer_rows
from .motor_visual_profile import COMPACT, match_visual, recorded_visual
from .training import ROOT, run_motor


def require_fit(report, model, input_profile):
    fit = load(report / ("act-clock-fit.json" if input_profile == CLOCKED_PROFILE else "act-fit.json"))
    worker = fit.get("validation", {})
    trace = worker.get("matched_sample_trace", {})
    timing = trace.get("recorded_at_ns") == trace.get("measured_at_ns")
    pixels = trace.get("preprocessing_max_abs_residual", {})
    conditioning = trace.get("counterfactual_target", {}).get("preprocessed_max_abs_difference", {})
    if not timing or not pixels or any(pixels.values()) or not conditioning.get("observation.images.camera2"):
        raise ValueError("Full comparison requires the verified issue-time target-aware trace")
    clock = trace.get("counterfactual_issue_index", {}).get("preprocessed_max_abs_difference", {})
    if input_profile == CLOCKED_PROFILE and not clock.get("observation.images.camera2"):
        raise ValueError("Full clocked comparison requires the verified clock-conditioned trace")
    if model == "act" and not fit["fit_gate"]:
        raise ValueError("ACT did not reach four placements on five fitting resets")
    return fit


def evaluation(config, model, data, checkpoint, root, deadline, heartbeat, *, role, smolvla_chunks=False):
    remaining = min(7200, int(deadline - time.monotonic()))
    if remaining < 30:
        raise TimeoutError("Allocator cap left insufficient time for this evaluation stage")
    lineage = load(data / "validation/smart-swarm-lineage.json")
    profile, visual = lineage["input_profile"], recorded_visual(lineage)
    receipt = checkpoint / "smart-swarm-input.json"
    if receipt.is_file():
        match_visual(load(receipt), visual)
    else:
        atomic(receipt, {"input_profile": profile, "dataset": str(data), **visual})
    if role == "validation":
        result = validate_checkpoint(config, data, checkpoint, root, deadline, heartbeat, smolvla_chunks=smolvla_chunks)
    else:
        result = evaluate(config["asset_root"], checkpoint, data, root, 20, 510, remaining,
                          seed_start=32_000, role=role, heartbeat=heartbeat, smolvla_chunks=smolvla_chunks)
    return {"checkpoint": str(checkpoint), "model": model, **result}


def validate_checkpoint(config, data, checkpoint, root, deadline, heartbeat, *, smolvla_chunks=False):
    directory = root / checkpoint.parent.name
    directory.mkdir(exist_ok=False, parents=True)
    lineage = load(data / "validation/smart-swarm-lineage.json")
    worker = Worker(checkpoint, data / "validation", directory / "policy.log", deadline=deadline,
                    full_validation=True, smolvla_chunks=smolvla_chunks)
    try:
        record = {**worker.recv(timeout=480), "resets": [], "evaluation_role": "validation"}
        if smolvla_chunks:
            record["action_transport"] = "verified-smolvla-chunk"
        for row in lineage["episodes"]:
            if time.monotonic() >= deadline:
                raise TimeoutError("Full validation reset cap reached")
            heartbeat()
            result = learned_case(config["asset_root"], worker, Path(row["raw_path"]), "aligned-sample-v2")
            record["resets"].append(result)
            atomic(directory / "validation.json", record)
        record["successful_resets"] = sum(contact_valid_placement(row) for row in record["resets"])
        atomic(directory / "validation.json", record)
        return record
    finally:
        worker.close()


def smolvla_schedule(probe, wall, free_bytes):
    load_s = probe["checkpoint_load_wall_s"]
    minibatch_s = probe["real_minibatch_forward_and_backward_wall_s"]
    weights = probe["model_weights_bytes"]
    moments = probe["optimizer_state_bytes_estimate"]
    if not all(math.isfinite(value) and value > 0 for value in (load_s, minibatch_s, weights, moments)):
        raise ValueError("The SmolVLA preflight did not measure complete training and storage costs")
    # Save two stages when space permits. The native pilot took 39 seconds to save its full checkpoint.
    checkpoint_bytes = weights + moments + 256 * 1024**2
    available_slots = min(2, free_bytes // checkpoint_bytes)
    setup_s, save_s = load_s + 45., 50.
    step_s = 1.25 * minibatch_s + 1.
    stages = min(available_slots, math.floor((wall - setup_s) / (save_s + step_s)))
    if stages < 1:
        raise ValueError("The measured runtime cannot save one complete SmolVLA checkpoint in the remaining allocation")
    steps = min(2_000, math.floor((wall - setup_s - stages * save_s) / step_s))
    frequency = math.ceil(steps / stages)
    return {"steps": steps, "save_freq": frequency, "checkpoint_storage_slots": stages,
            "checkpoint_bytes_with_margin": checkpoint_bytes, "available_disk_bytes": free_bytes,
            "estimated_seconds_per_update": step_s, "startup_seconds_reserved": setup_s,
            "seconds_reserved_per_checkpoint": save_s}


def smolvla_evaluation_reserve(probe):
    queue = probe["queued_action_parity"]
    actions = queue["action_steps_per_chunk"]
    if queue["exact_native_rows"] != actions + 1 or actions < 1:
        raise ValueError("The actual SmolVLA action chunk and refresh must match the installed policy")
    refresh_s = max(queue["first_chunk_seconds"], queue["next_chunk_seconds"])
    # One validation candidate and one fresh cohort, each with 20 reset recipes.
    motor_s = max(row["wall_seconds"] for row in probe["motor_path_timing"])
    reset_s = math.ceil(510 / actions) * refresh_s + motor_s + 1.
    worker_s = probe["checkpoint_load_wall_s"] + 20 * probe["minibatch_forward_wall_s"]
    return max(1200., math.ceil(40 * reset_s + 2 * worker_s + 90.))


def full_spec(model, fit, wall, training_probe, data_root):
    updates = 5000 - fit.get("cumulative_fit_updates", fit["optimizer_updates"]) if model == "act" else 2000
    spec = {"type": model, "steps": updates, "wall_seconds": min(3900, int(wall - 1200)),
            "batch_size": 4 if model == "act" else 1, "save_freq": 350 if model == "act" else 100}
    if model == "act":
        spec["initial_checkpoint"] = fit["checkpoint"]
    schedule = None
    if model == "smolvla" and training_probe is not None:
        evaluation_s = smolvla_evaluation_reserve(training_probe)
        spec["wall_seconds"] = min(spec["wall_seconds"], int(wall - evaluation_s))
        schedule = smolvla_schedule(training_probe, spec["wall_seconds"], shutil.disk_usage(data_root).free)
        schedule["evaluation_seconds_reserved"] = evaluation_s
        schedule["motor_path_receipt_sha256"] = training_probe["motor_path_receipt_sha256"]
        spec.update({key: schedule[key] for key in ("steps", "save_freq")})
    return spec, schedule


def run(config, model, data, report, grant, heartbeat, *, setup_wall_s=0., training_probe=None):
    destination = report / f"{model}-full.json"
    if destination.is_file():
        return load(destination)
    train, validation = [load(data / split / "smart-swarm-lineage.json") for split in ("train", "validation")]
    visual = match_visual(train, validation)
    if visual["visual_profile"] == COMPACT and grant.get("gpu_render"):
        raise ValueError("The compact RGB profile is bound to the verified software renderer")
    profile = train["input_profile"]
    if [len(row["episodes"]) for row in (train, validation)] != [80, 20] or not all(
            row["input_profile"] == profile for row in (train, validation)) or profile not in (PROFILE, CLOCKED_PROFILE):
        raise ValueError("Full training requires the same admitted 80/20 split and observation profile")
    fit = require_fit(report, model, profile)
    training = load(ROOT / "configs/training.json")
    recipe = "act-aligned-full" if model == "act" else "smolvla-pipeline"
    wall = 7200 - setup_wall_s - (fit.get("cumulative_fit_wall_s", fit["elapsed_s"]) if model == "act" else 0)
    spec, schedule = full_spec(model, fit, wall, training_probe, config["data_root"])
    training["profiles"][recipe] = spec
    root = config["data_root"] / "training-runs" / (recipe + "-" + uuid.uuid4().hex[:12])
    deadline = time.monotonic() + wall
    result = {"status": "running", "job_directory": str(root), "cap": spec, "dataset": str(data),
              "provenance": grant, "preflight_wall_s_charged": setup_wall_s, **visual}
    if schedule is not None:
        result["smolvla_checkpoint_schedule"] = schedule
    atomic(destination, result)
    try:
        run_motor(training, recipe, data, root, train, validation, grant, heartbeat)
    except RuntimeError as error:
        result["optimizer_stop_reason"] = str(error)
    rows = optimizer_rows(root / "training.log")
    result["optimizer_updates"] = rows[-1]["update"] if rows else 0
    try:
        result.update(select(config, model, data, root, deadline, heartbeat, smolvla_chunks=schedule is not None))
    except (OSError, RuntimeError, ValueError, TimeoutError, ProtocolError) as error:
        result.update(status="inconclusive", evaluation_blocker=str(error))
    finally:
        atomic(destination, result)
    return result


def select(config, model, data, root, deadline, heartbeat, *, smolvla_chunks=False):
    home = root / "training/checkpoints"
    paths = sorted(path / "pretrained_model" for path in home.iterdir()
                   if path.name.isdecimal() and (path / "pretrained_model/model.safetensors").is_file())
    if not paths:
        raise ValueError("No checkpoint was saved inside the training cap")
    compared = []
    for checkpoint in paths[-2:] if model == "act" else paths[-1:]:
        result = evaluation(config, model, data, checkpoint, root / "validation", deadline, heartbeat,
                            role="validation", smolvla_chunks=smolvla_chunks)
        compared.append(result)
        atomic(root / "validation-choices.json", compared)
    score = lambda item: (item["successful_resets"], -item["offline_validation_mean_loss"])
    checkpoint = Path(max(compared, key=score)["checkpoint"])
    fresh = evaluation(config, model, data, checkpoint, root / "fresh", deadline, heartbeat,
                       role="fresh-test", smolvla_chunks=smolvla_chunks)
    return {"status": "evaluated", "checkpoint": str(checkpoint), "validation": compared, "fresh": fresh,
            "selection": "Validation containment count, tie-broken by independent split offline loss"}
