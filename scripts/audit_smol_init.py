"""Check the installed SmolVLA CLI and real archived processors without fitting."""

import argparse
from base64 import b64encode
from importlib.metadata import version
from pathlib import Path
from resource import RUSAGE_SELF, getrusage
import sys
import time

import numpy as np
import torch
from lerobot.configs import parser
from lerobot.configs.train import TrainPipelineConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.scripts import lerobot_train

from policy_worker import example_batch, smolvla_chunk
from smart_swarm.episode import atomic, load
from smart_swarm.motor_input import CLOCKED_PROFILE, PROFILE, conditioned_top, instruction
from smart_swarm.training import ROOT, training_command
from train_cpu import bind_smol_inputs, configure


@parser.wrap()
def native_config(config: TrainPipelineConfig):
    config.validate()
    return config


def read_config(dataset, output):
    settings = load(ROOT / "configs/training.json")
    command = training_command(settings, "smolvla-pipeline", dataset, output.with_suffix(".unused"))
    saved = sys.argv
    try:
        sys.argv = command[1:]
        pipeline = native_config()
    finally:
        sys.argv = saved
    policy = pipeline.policy
    expected = settings["weights"]
    if (version("lerobot") != "0.6.1" or pipeline.resume or policy.device != "cpu" or policy.type != "smolvla"
            or str(policy.pretrained_path) != expected["smolvla"] or not policy.load_vlm_weights
            or policy.vlm_model_name != expected["vision_language_backbone"]):
        raise ValueError(f"Native CLI did not preserve the pinned profile: {policy}")
    return pipeline


def processors(config, stats):
    configure()
    mapping = config.normalization_mapping
    features = {**config.input_features, **config.output_features}
    return lerobot_train.make_pre_post_processors(config, pretrained_path=config.pretrained_path, dataset_stats=stats,
        preprocessor_overrides={"device_processor": {"device": "cpu"}, "rename_observations_processor": {"rename_map": {}},
                                "normalizer_processor": {"features": features, "norm_map": mapping, "stats": stats}},
        postprocessor_overrides={"unnormalizer_processor": {"features": config.output_features, "norm_map": mapping,
                                                             "stats": stats}})


def sample_check(dataset, entry, profile, pre, post):
    raw_path = Path(entry["raw_path"])
    manifest = load(raw_path / "manifest.json")
    index = 41
    marker = entry["target_marker"]
    issue = index if profile == CLOCKED_PROFILE else None
    with np.load(raw_path / "motor.npz", allow_pickle=False) as raw:
        views = (raw["wrist"][index], conditioned_top(raw["top"][index], marker, issue_index=issue))
        state = raw["state"][index]
        payload = {"state": state.tolist(), "task": instruction(manifest["task"], marker),
                   "images": [b64encode(frame.tobytes()).decode("ascii") for frame in views]}
    issued = example_batch(payload)
    observed = dataset[index]
    live = pre({**issued, "action": observed["action"]})
    stored = pre({key: observed[key] for key in (*issued, "action")})
    tokens = "observation.language.tokens"
    keys = (*[key for key in issued if key.startswith("observation.")], tokens, "observation.language.attention_mask")
    parity = {key: torch.equal(live[key], stored[key]) for key in keys}
    other = "violet" if marker == "cyan" else "cyan"
    different = pre({**issued, "task": instruction(manifest["task"], other)})
    state_stats = dataset.meta.stats["observation.state"]
    expected_state = (state - state_stats["mean"]) / (state_stats["std"] + 1e-8)
    state_error = float(np.max(np.abs(live["observation.state"].numpy().reshape(-1) - expected_state)))
    action_error = float((post(live["action"]).reshape(-1) - observed["action"]).abs().max())
    change = int(torch.count_nonzero(live[tokens] != different[tokens]))
    if not all(parity.values()) or not change or max(action_error, state_error) >= 1e-4:
        raise ValueError(f"Archived native processor parity failed: {parity}, {change}, {action_error}, {state_error}")
    return {"parity": parity, "target_only_changed_tokens": change, "state_mean_std_max_abs_error": state_error,
            "native_action_roundtrip_max_abs": action_error, "issue_index": index}


def checkpoint_storage(model):
    weights = sum(value.numel() * value.element_size() for value in model.state_dict().values())
    parameters = [value for value in model.get_optim_params() if value.requires_grad]
    trainable_bytes = sum(value.numel() * value.element_size() for value in parameters)
    # Installed AdamW uses two moments and one scalar step per parameter tensor.
    return {"model_weights_bytes": weights, "optimizer_state_bytes_estimate": 2 * trainable_bytes + 4 * len(parameters),
            "peak_process_rss_bytes": getrusage(RUSAGE_SELF).ru_maxrss * 1024}


def archive_inputs(native, config, index):
    keys = (*config.input_features, "task")
    for issue in range(index, index + config.n_action_steps + 1):
        sample = native[issue]
        yield {key: sample[key].unsqueeze(0) if isinstance(sample[key], torch.Tensor) else [sample[key]] for key in keys}


def queue_profile(native, index, config, model, pre, post):
    inputs = list(archive_inputs(native, config, index))
    torch.manual_seed(3)
    model.reset()
    with torch.inference_mode():
        reference = [post(model.select_action(pre(sample.copy()))).detach().cpu().reshape(-1).tolist() for sample in inputs]
    torch.manual_seed(3)
    model.reset()
    started = time.monotonic()
    first = smolvla_chunk(inputs[0].copy(), model, pre, post)
    first_s = time.monotonic() - started
    last = smolvla_chunk(inputs[-1].copy(), model, pre, post)
    next_s = time.monotonic() - started - first_s
    if reference != first + last[:1]:
        raise ValueError("Transported SmolVLA actions differ from the installed single-step queue")
    return {"exact_native_rows": len(reference), "first_chunk_seconds": first_s, "next_chunk_seconds": next_s,
            "action_steps_per_chunk": config.n_action_steps, "diffusion_steps": config.num_steps}


def time_minibatch(dataset, metadata, config, pre, post, index):
    offsets = {"action": [step / 30 for step in range(config.chunk_size)]}
    native = LeRobotDataset("smart-swarm/train", root=dataset / "train", delta_timestamps=offsets)
    sample = native[index]
    batch = {key: value.unsqueeze(0) if isinstance(value, torch.Tensor) else [value]
             for key, value in sample.items()}
    torch.manual_seed(3)
    started = time.monotonic()
    model = lerobot_train.make_policy(cfg=config, ds_meta=metadata)
    loaded_s = time.monotonic() - started
    model.train()
    started = time.monotonic()
    loss, _ = model.forward(pre(batch))
    forward_s = time.monotonic() - started
    if not torch.isfinite(loss).item():
        raise ValueError("The actual training minibatch returned a non-finite loss")
    loss.backward()
    training_s = time.monotonic() - started
    model.zero_grad(set_to_none=True)
    model.eval()
    model.reset()
    allowed = {key: batch[key] for key in (*config.input_features, "task")}
    with torch.inference_mode():
        started = time.monotonic()
        action = post(model.select_action(pre(allowed))).detach().cpu().reshape(-1)
        inference_s = time.monotonic() - started
    if action.numel() != 6 or not torch.isfinite(action).all().item():
        raise ValueError("The actual model returned an invalid native actuator target")
    transport = queue_profile(native, index, config, model, pre, post)
    return {"checkpoint_load_wall_s": loaded_s, "real_minibatch_forward_and_backward_wall_s": training_s,
            "first_native_chunk_issue_wall_s": inference_s, "minibatch_forward_wall_s": forward_s,
            "single_minibatch_loss": float(loss.detach().item()), "queued_action_parity": transport,
            "unpadded_actions": int((~sample["action_is_pad"]).sum().item()), "predicted_native_action": action.tolist(),
            "optimizer_created": False, **checkpoint_storage(model)}


def audit(dataset, output, *, timed_minibatch=False):
    pipeline = read_config(dataset, output)
    policy = pipeline.policy
    native = LeRobotDataset("smart-swarm/train", root=dataset / "train")
    cli_cameras = list(policy.image_features)
    bind_smol_inputs(policy, native.meta)
    pre, post = processors(policy, native.meta.stats)
    tokenizer = next(step.tokenizer_name for step in pre.steps if hasattr(step, "tokenizer_name"))
    if tokenizer != policy.vlm_model_name:
        raise ValueError("Training tokens did not use the installed pinned backbone snapshot")
    lineage = load(dataset / "train/smart-swarm-lineage.json")
    profile, entry = lineage["input_profile"], lineage["episodes"][0]
    if profile not in (PROFILE, CLOCKED_PROFILE):
        raise ValueError("An aligned and target-conditioned motor dataset is required")
    weights = Path(policy.pretrained_path) / "model.safetensors"
    report = {"scope": "Archived development sample and native processors without policy loading or fitting",
              "lerobot": version("lerobot"), "device": policy.device, "optimizer_updates": 0, "closed_loop_trials": 0,
              "policy_checkpoint": str(policy.pretrained_path), "weights_cache_object": weights.resolve().name,
              "weights_present_bytes": weights.stat().st_size, "backbone": policy.vlm_model_name, "tokenizer": tokenizer,
              "resume_optimizer": pipeline.resume, "update_cap": pipeline.steps, "dataset": str(dataset),
              "input_profile": profile,
              "source_world": entry["source_world_episode_id"], "camera_order": ["wrist", "top"],
              "native_cli_merged_camera_keys": cli_cameras, "bound_camera_keys": list(policy.image_features),
              "chunk_size": policy.chunk_size, "action_steps": policy.n_action_steps,
              "resize": policy.resize_imgs_with_padding, "diffusion_steps": policy.num_steps}
    report.update(sample_check(native, entry, profile, pre, post))
    if timed_minibatch:
        if pipeline.optimizer.type != "adamw":
            raise ValueError("The checkpoint estimate requires the declared installed AdamW optimizer")
        report["scope"] = "Official processor and genuine minibatch and inference timing without an optimizer"
        report["minibatch"] = time_minibatch(dataset, native.meta, policy, pre, post, report["issue_index"])
    atomic(output, report)


if __name__ == "__main__":
    arguments = argparse.ArgumentParser(description=__doc__)
    arguments.add_argument("--dataset", type=Path, required=True)
    arguments.add_argument("--output", type=Path, required=True)
    arguments.add_argument("--timed-minibatch", action="store_true")
    args = arguments.parse_args()
    audit(args.dataset.resolve(), args.output.resolve(), timed_minibatch=args.timed_minibatch)
