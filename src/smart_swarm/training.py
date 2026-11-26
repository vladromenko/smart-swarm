"""Bounded local LeRobot subprocesses using the independent motor dataset."""

import os
from pathlib import Path
import subprocess
import time
import uuid

from .episode import atomic, load
from .motor_visual_profile import match_visual

ROOT = Path(__file__).resolve().parents[2]


def offline(*, device="cpu"):
    if device not in ("cpu", "cuda"):
        raise ValueError(f"Unknown policy device {device}")
    env = {**os.environ, "HF_HUB_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1",
           "HF_HUB_DISABLE_TELEMETRY": "1", "WANDB_MODE": "disabled", "OMP_NUM_THREADS": "3",
           "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "3", "TOKENIZERS_PARALLELISM": "false",
           "PYTHONDONTWRITEBYTECODE": "1"}
    if device == "cpu":
        env["CUDA_VISIBLE_DEVICES"] = ""
    return env


def export_motor(source, output, *, clocked=False):
    config = load(ROOT / "configs/training.json")
    argv = [config["runtime"], str(ROOT / "scripts/export_lerobot.py"),
            "--source", str(source.resolve()), "--output", str(output.resolve())]
    if clocked:
        argv.append("--clocked")
    subprocess.run(argv, check=True, env=offline(), timeout=1200)
    return {"dataset": str(output), "source": str(source), "runtime": config["lerobot_version"]}


def training_command(config, recipe, dataset, output, *, device="cpu"):
    spec = config["profiles"][recipe]
    common = [config["runtime"], "-m", "lerobot.scripts.lerobot_train",
              "--dataset.repo_id=smart-swarm/train", f"--dataset.root={dataset / 'train'}",
              "--dataset.use_imagenet_stats=false", "--wandb.enable=false", f"--policy.device={device}",
              "--policy.push_to_hub=false", "--env_eval_freq=0", "--seed=3",
              "--num_workers=0", "--persistent_workers=false", "--log_freq=1",
              f"--steps={spec['steps']}", f"--batch_size={spec['batch_size']}",
              f"--save_freq={min(spec['steps'], spec.get('save_freq', 150 if recipe == 'act-five-fit' else 1000))}",
              f"--output_dir={output}"]
    if spec["type"] == "smolvla" or device == "cuda":
        common = [config["runtime"], str(ROOT / "scripts/train_cpu.py"), *common[3:]]
    if spec["type"] == "smolvla":
        return common + smol_args(config, pilot=recipe == "smolvla-pilot")
    if spec.get("initial_checkpoint"):
        return common + [f"--policy.path={spec['initial_checkpoint']}"]
    return common + act_args(recipe)


def act_args(recipe):
    if recipe in ("act-five-fit", "act-aligned-full"):
        return ["--policy.type=act", "--policy.chunk_size=24", "--policy.n_action_steps=6",
                "--policy.dim_model=128", "--policy.dim_feedforward=512", "--policy.n_encoder_layers=2",
                "--policy.n_vae_encoder_layers=2", "--policy.optimizer_lr=0.00015",
                "--policy.optimizer_lr_backbone=0.00003"]
    base = ["--policy.type=act", "--policy.chunk_size=50", "--policy.n_action_steps=50"]
    if recipe == "act-pilot":
        return base + ["--policy.pretrained_backbone_weights=null", "--policy.dim_model=128",
                       "--policy.dim_feedforward=512", "--policy.n_encoder_layers=2",
                       "--policy.n_vae_encoder_layers=2"]
    return base


def smol_args(config, *, pilot):
    import json

    weights = Path(config["weights"]["smolvla"])
    vlm = Path(config["weights"]["vision_language_backbone"])
    if not (weights / "model.safetensors").is_file() or not (vlm / "config.json").is_file():
        raise FileNotFoundError("Pinned SmolVLA model or cached tokenizer/backbone is missing")
    inputs = {"observation.state": {"type": "STATE", "shape": [6]},
              **{f"observation.images.camera{i}": {"type": "VISUAL", "shape": [3, 120, 160]}
                 for i in (1, 2)}}
    kwargs = [f"--policy.path={weights}", f"--policy.vlm_model_name={vlm}",
              "--policy.input_features=" + json.dumps(inputs),
              "--policy.output_features=" + json.dumps({"action": {"type": "ACTION", "shape": [6]}})]
    if pilot:
        kwargs.extend(["--policy.resize_imgs_with_padding=[256,256]", "--policy.num_steps=2"])
    return kwargs


def train(recipe, dataset, location, platform_config, *, device="cpu"):
    config = load(ROOT / "configs/training.json")
    if recipe not in config["profiles"]:
        raise ValueError(f"Unknown training recipe: {recipe}")
    if config["profiles"][recipe]["type"] == "smolvla":
        verified_shared_path()
    dataset = dataset.resolve()
    data = load(dataset / "train/smart-swarm-lineage.json")
    validation = load(dataset / "validation/smart-swarm-lineage.json")
    if recipe.endswith("-pipeline") and (len(data["episodes"]) < 80 or len(validation["episodes"]) < 20):
        raise ValueError("Full pipeline recipe requires at least 80 train and 20 validation worlds")
    from .research_jobs import admitted

    root = location / f"{recipe}-{uuid.uuid4().hex[:12]}"
    archive_only = device == "cpu" and bool(platform_config.get("native_cpu_admission"))
    with admitted(platform_config, root.name, device=device, archive_only=archive_only) as (grant, heartbeat):
        return run_motor(config, recipe, dataset, root, data, validation, grant, heartbeat, device=device)


def verified_shared_path():
    root = ROOT / "evidence/resume"
    teachers = load(root / "teacher-replays.json")
    act = load(root / "act-development-diagnostic.json")
    clean = not any(act["sample_trace"]["preprocessing_max_abs_residual"].values())
    if not teachers["teacher_gate"] or act["status"] != "complete" or not clean:
        raise ValueError("Shared path must pass the teacher and ACT trace checks before SmolVLA optimization")


def run_motor(config, recipe, dataset, root, data, validation, grant, heartbeat, *, device="cpu"):
    from .research_jobs import verify_compute_scope

    verify_compute_scope(device, grant)
    visual = match_visual(data, validation)
    input_profile = data.get("input_profile", "historical-rgb")
    if validation.get("input_profile", "historical-rgb") != input_profile:
        raise ValueError("Motor train and validation input conditioning must agree")
    root.mkdir(exist_ok=False, parents=True)
    spec = config["profiles"][recipe]
    args = training_command(config, recipe, dataset, root / "training", device=device)
    record = {"recipe": recipe, "status": "training", "run_mode": "mujoco", "device": device,
              "requested_updates": spec["steps"], "argv": args, "train_worlds": len(data["episodes"]),
              "validation_worlds": len(validation["episodes"]), "dataset": str(dataset),
              "run_directory": str(root), "compute_reservation": grant, "input_profile": input_profile, **visual}
    atomic(root / "run.json", record)
    started = time.monotonic()
    record.update(run_once(args, root / "training.log", root / "training", spec["wall_seconds"], heartbeat, device=device))
    bind_checkpoints(root / "training/checkpoints", {"dataset": str(dataset), "input_profile": input_profile, **visual})
    record["elapsed_s"] = time.monotonic() - started
    atomic(root / "run.json", record)
    if record["status"] != "trained":
        raise RuntimeError(f"Training {record['status']}; see {root / 'training.log'}")
    return record


def bind_checkpoints(home, binding):
    checkpoints = {path.resolve() for path in home.glob("*/pretrained_model")}
    for path in sorted(checkpoints):
        if path.is_relative_to(home.resolve()) and (path / "model.safetensors").is_file():
            atomic(path / "smart-swarm-input.json", binding)


def run_once(args, logfile, output, timeout, heartbeat, *, device="cpu"):
    from .contract import ProtocolError
    from .research_jobs import execute

    env = {**offline(device=device), "SMART_SWARM_TRAINING_DEVICE": device}
    with logfile.open("w") as stream:
        try:
            code = execute(args, env, stream, time.monotonic() + timeout, heartbeat)
        except TimeoutError:
            return {"status": "wall_time_capped"}
        except KeyboardInterrupt:
            return {"status": "cancelled"}
        except ProtocolError as error:
            return {"status": "compute_reservation_lost", "reason": str(error)}
    checkpoint = output / "checkpoints/last/pretrained_model"
    if code == -2:
        return {"status": "cancelled", "returncode": code}
    if code or not (checkpoint / "model.safetensors").is_file():
        return {"status": "failed", "returncode": code}
    return {"status": "trained", "returncode": code, "checkpoint": str(checkpoint)}
