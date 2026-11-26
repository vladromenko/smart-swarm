"""Fresh process for checkpoint reload and scoped LeRobot inference."""

import argparse
from base64 import b64decode
import json
import os
from pathlib import Path
from resource import RUSAGE_SELF, getrusage
import sys
from time import monotonic_ns

import numpy as np
import torch

from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import get_policy_class, make_pre_post_processors
from lerobot.utils.constants import ACTION
from policy_runtime_device import verify_runtime


def processor_devices(checkpoint, device):
    settings = {}
    for role, target in (("pre", device), ("post", "cpu")):
        path = Path(checkpoint) / f"policy_{role}processor.json"
        if path.is_file():
            saved = json.loads(path.read_text())
            if any(step.get("registry_name") == "device_processor" for step in saved["steps"]):
                settings[f"{role}processor_overrides"] = {"device_processor": {"device": target}}
    if device == "cuda" and "preprocessor_overrides" not in settings:
        raise ValueError("The checkpoint has no explicit input device processor for CUDA evaluation")
    return settings


def load_policy(checkpoint, *, device="cpu"):
    config = PreTrainedConfig.from_pretrained(str(checkpoint))
    config.device = device
    policy_type = get_policy_class(config.type)
    policy = policy_type.from_pretrained(str(checkpoint), config=config, strict=True).to(device).float().eval()
    options = processor_devices(checkpoint, device)
    processors = make_pre_post_processors(config, pretrained_path=str(checkpoint), **options)
    return config, policy, *processors


def validation_points(dataset, all_sources):
    if not all_sources:
        return np.linspace(37, len(dataset) - 1, num=8, dtype=int)
    issues = (37, 135, 220, 405)
    points = []
    for index, row in enumerate(dataset.meta.episodes):
        first, stop = row["dataset_from_index"], row["dataset_to_index"]
        points.append(first + min(issues[index % len(issues)], stop - first - 1))
    return points


def heldout_loss(root, config, policy, preprocessor, all_sources=False):
    dataset = LeRobotDataset("smart-swarm/validation", root=root,
                            delta_timestamps={"action": [i / 30 for i in range(config.chunk_size)]})
    points = validation_points(dataset, all_sources)
    scores, sources = [], set()
    if all_sources:
        torch.manual_seed(3)
    with torch.inference_mode():
        for point in points:
            sample = dataset[int(point)]
            sources.add(int(sample["episode_index"]))
            batch = {key: value.unsqueeze(0) if isinstance(value, torch.Tensor) else [value]
                     for key, value in sample.items()}
            loss, _ = policy.forward(preprocessor(batch))
            scores.append(float(loss.item()))
    if not np.isfinite(scores).all():
        raise ValueError("Non-finite validation loss")
    if all_sources and len(sources) != dataset.num_episodes:
        raise ValueError("Offline comparison input did not cover every declared validation source")
    return {"policy": config.type, "offline_validation_source_worlds": dataset.num_episodes,
            "offline_validation_sampled_sources": sorted(sources), "offline_validation_samples": len(scores),
            "offline_validation_mean_loss": float(np.mean(scores))}


def example_batch(request):
    state = torch.as_tensor(request["state"], dtype=torch.float32)
    batch = {"observation.state": state, "task": request["task"]}
    for index, image in enumerate(request["images"], 1):
        pixels = np.frombuffer(b64decode(image, validate=True), dtype=np.uint8).copy().reshape(120, 160, 3)
        batch[f"observation.images.camera{index}"] = torch.from_numpy(pixels).permute(2, 0, 1).float() / 255.
    return batch


def step(request, model, pre, post):
    batch = example_batch(request)
    with torch.inference_mode():
        action = post(model.select_action(pre(batch)))
    output = action.detach().cpu().reshape(-1).tolist()
    if len(output) != 6 or not np.isfinite(output).all():
        raise ValueError("Invalid native SO101 action after unnormalization")
    return output


def smolvla_chunk(batch, model, pre, post):
    if (model.config.type != "smolvla" or set(model._queues) != {ACTION} or model._queues[ACTION]
            or model.config.rtc_config is not None and model.config.rtc_config.enabled):
        raise ValueError("This checkpoint does not implement the verified SmolVLA action-only queue")
    with torch.inference_mode():
        actions = model.predict_action_chunk(pre(batch))[0, :model.config.n_action_steps]
        native = [post(row.unsqueeze(0)).detach().cpu().reshape(-1).tolist() for row in actions]
    if np.shape(native) != (model.config.n_action_steps, 6) or not np.isfinite(native).all():
        raise ValueError("Invalid native action sequence after unnormalization")
    return native


def audit_row(request, model, pre, post):
    dataset = LeRobotDataset("smart-swarm/" + request["split"], root=request["split_root"])
    raw = dataset[request["index"]]
    payload = example_batch(request)
    keys = [key for key in payload if key.startswith("observation.")]
    native = {key: raw[key] for key in keys}
    native["task"] = raw["task"]
    source = pre({**payload, "action": raw["action"]})
    stored = pre({**native, "action": raw["action"]})
    residual = {key: float((source[key] - stored[key]).abs().max()) for key in keys}
    roundtrip = post(source["action"]).detach().cpu().flatten()
    with torch.inference_mode():
        model.reset()
        chunk = model.predict_action_chunk(source)
        first = post(chunk[:, 0]).detach().cpu().flatten().tolist()
        emitted = [step(request, model, pre, post) for _ in range(3)]
        expected = [post(chunk[:, i]).detach().cpu().flatten().tolist() for i in range(3)]
        model.reset()
        repeated_first = step(request, model, pre, post)
        model.reset()
    return {"preprocessing_max_abs_residual": residual, "task_matches": raw["task"] == request["task"],
            "recorded_native_action": raw["action"].tolist(), "normalized_target": source["action"].flatten().tolist(),
            "normalized_proprioception": source["observation.state"].flatten().tolist(),
            "normalization_roundtrip_max_abs": float((roundtrip - raw["action"]).abs().max()),
            "chunk_length": chunk.shape[1], "first_raw_model_action": chunk[0, 0].tolist(),
            "first_decoded_native_action": first, "first_three_emitted": emitted,
            "chunk_order_max_abs_residual": float(np.max(np.abs(np.asarray(expected) - emitted))),
            "reset_residual": float(np.max(np.abs(np.asarray(repeated_first) - first)))}


def action_errors(request, config, model, pre, post):
    split = request["split"]
    offsets = {"action": [i / 30 for i in range(config.chunk_size)]}
    data = LeRobotDataset("smart-swarm/" + split, root=request["split_root"], delta_timestamps=offsets)
    report = []
    with torch.inference_mode():
        for stage, index in request["rows"]:
            sample = data[index]
            batch = {key: val.unsqueeze(0) if isinstance(val, torch.Tensor) else [val] for key, val in sample.items()}
            model.reset()
            output = model.predict_action_chunk(pre(batch))[0]
            actions = torch.stack([post(value.unsqueeze(0)).cpu().flatten() for value in output])
            valid = ~sample["action_is_pad"]
            delta = (actions - sample["action"]).abs()[valid]
            report.append({"stage": stage, "index": index, "valid_chunk_actions": int(valid.sum()),
                           "one_step_mae": delta[0].tolist(), "chunk_mae": delta.mean(dim=0).tolist()})
    model.reset()
    return report


def conditioning(request, model, pre):
    inputs = [pre(example_batch(example)) for example in request["targets"]]
    keys = [key for key in inputs[0] if key.startswith("observation.")]
    changes = {key: float((inputs[0][key] - inputs[1][key]).abs().max()) for key in keys}
    with torch.inference_mode():
        outputs = [model.predict_action_chunk(batch) for batch in inputs]
    model.reset()
    return {"preprocessed_max_abs_difference": changes,
            "first_normalized_action_max_abs_difference": float((outputs[0][:, 0] - outputs[1][:, 0]).abs().max())}


def memory_usage(device):
    profile = {"peak_worker_rss_bytes": getrusage(RUSAGE_SELF).ru_maxrss * 1024}
    if device == "cuda":
        profile.update(cuda_peak_tensor_bytes=torch.cuda.max_memory_allocated(),
                       cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved())
    return profile


def startup_profile(config, runtime, load_ns):
    return {**runtime, "policy": config.type, "model_and_processors_load_ns": load_ns,
            "chunk_size": config.chunk_size, "actions_per_chunk": config.n_action_steps,
            "diffusion_steps": getattr(config, "num_steps", None),
            "resize_with_padding": getattr(config, "resize_imgs_with_padding", None)}


def check_timing_inputs(root, windows):
    dataset = LeRobotDataset("smart-swarm/train", root=root)
    for row in windows:
        sample = dataset[row["dataset_index"]]
        batch = example_batch(row["payload"])
        if (sample["episode_index"].item() != row["episode_index"] or sample["frame_index"].item() != row["issue"]
                or batch["task"] != sample["task"]
                or any(not torch.equal(val, sample[key]) for key, val in batch.items() if isinstance(val, torch.Tensor))):
            raise ValueError(f"The timing input at issue {row['issue']} differs from the official training row")
    return {"verified_training_rows": len(windows)}


def timed_action(request, model, pre, post):
    started = monotonic_ns()
    preprocessor_ns = 0

    def measured(batch):
        nonlocal preprocessor_ns
        before = monotonic_ns()
        result = pre(batch)
        preprocessor_ns += monotonic_ns() - before
        return result

    if request["type"] == "smolvla_chunk":
        response = {"actions": smolvla_chunk(example_batch(request), model, measured, post)}
    else:
        response = {"action": step(request, model, measured, post)}
    return {**response, "worker_host_preprocessor_ns": preprocessor_ns,
            "worker_request_ns": monotonic_ns() - started}


def serve(descriptor, checkpoint, validation, all_sources=False, *, device="cpu", timing_startup=False):
    runtime = verify_runtime(device)
    started = monotonic_ns()
    config, model, pre, post = load_policy(checkpoint, device=device)
    with os.fdopen(descriptor, "w", buffering=1) as outbound:
        initial = (startup_profile(config, runtime, monotonic_ns() - started) if timing_startup else
                   heldout_loss(validation, config, model, pre, all_sources))
        print(json.dumps(initial, allow_nan=False), file=outbound)
        for line in sys.stdin:
            command = json.loads(line)
            if command["type"] == "reset":
                if "seed" in command:
                    torch.manual_seed(command["seed"])
                model.reset()
                print('{"reset": true}', file=outbound)
            elif timing_startup and command["type"] in ("action", "smolvla_chunk"):
                print(json.dumps(timed_action(command, model, pre, post), allow_nan=False), file=outbound)
            elif command["type"] == "action":
                print(json.dumps({"action": step(command, model, pre, post)}, allow_nan=False), file=outbound)
            elif command["type"] == "smolvla_chunk":
                result = smolvla_chunk(example_batch(command), model, pre, post)
                print(json.dumps({"actions": result}, allow_nan=False), file=outbound)
            elif command["type"] == "audit_row":
                print(json.dumps(audit_row(command, model, pre, post), allow_nan=False), file=outbound)
            elif command["type"] == "offline_actions":
                result = action_errors(command, config, model, pre, post)
                print(json.dumps({"rows": result}, allow_nan=False), file=outbound)
            elif command["type"] == "audit_conditioning":
                print(json.dumps(conditioning(command, model, pre), allow_nan=False), file=outbound)
            elif command["type"] == "timing_stats" and timing_startup:
                print(json.dumps(memory_usage(device), allow_nan=False), file=outbound)
            elif command["type"] == "timing_inputs" and timing_startup:
                print(json.dumps(check_timing_inputs(validation, command["windows"]), allow_nan=False), file=outbound)
            else:
                raise ValueError("Invalid local worker command")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fd", type=int, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--all-validation-sources", action="store_true")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--timing-startup", action="store_true")
    args = parser.parse_args()
    serve(args.fd, args.checkpoint, args.validation, args.all_validation_sources,
          device=args.device, timing_startup=args.timing_startup)
