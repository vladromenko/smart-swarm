"""Invoke the official trainer with the verified camera inputs and float32 policy weights."""

from functools import wraps
import os
from pathlib import Path
import shutil

from lerobot.scripts import lerobot_train
from lerobot.utils.feature_utils import dataset_to_policy_features


def bind_smol_inputs(config, metadata):
    features = dataset_to_policy_features(metadata.features)
    names = ("observation.state", "observation.images.camera1", "observation.images.camera2")
    config.input_features = {name: features[name] for name in names}
    if config.empty_cameras or tuple(config.action_feature.shape) != (6,):
        raise ValueError("SmolVLA requires two real views, no padded camera, and six native SO101 actions")


def install_checkpoint_guard():
    save_checkpoint = lerobot_train.save_checkpoint

    @wraps(save_checkpoint)
    def save_complete(checkpoint_dir, **kwargs):
        policy = kwargs["policy"]
        weights = sum(value.numel() * value.element_size() for value in policy.state_dict().values())
        optimizer = kwargs["optimizer"]
        moments = sum(value.numel() * value.element_size() for state in optimizer.state.values()
                      for value in state.values() if hasattr(value, "element_size"))
        run_directory = Path(kwargs["cfg"].output_dir).parent
        if shutil.disk_usage(run_directory).free < weights + moments + 256 * 1024**2:
            raise RuntimeError("Insufficient space for one complete CPU checkpoint; prior checkpoints are preserved")
        pending = checkpoint_dir.with_name(f".{checkpoint_dir.name}-unpublished")
        save_checkpoint(checkpoint_dir=pending, **kwargs)
        pending.rename(checkpoint_dir)

    lerobot_train.save_checkpoint = save_complete


def configure():
    # Upstream SmolVLA defaults to loading BF16. Homebrain's ARM CPU has no
    # native BF16 matmul. Keep the official model and optimizer, with FP32 weights.
    make_policy = lerobot_train.make_policy

    @wraps(make_policy)
    def make_float_policy(cfg, **kwargs):
        requested = os.environ.get("SMART_SWARM_TRAINING_DEVICE")
        if requested and str(cfg.device).partition(":")[0] != requested:
            raise RuntimeError("The native trainer changed the explicitly requested policy device")
        if cfg.type == "smolvla":
            bind_smol_inputs(cfg, kwargs["ds_meta"])
        policy = make_policy(cfg=cfg, **kwargs).float()
        if requested and next(policy.parameters()).device.type != requested:
            raise RuntimeError("The native trainer loaded the policy on another device")
        return policy

    lerobot_train.make_policy = make_float_policy
    make_processors = lerobot_train.make_pre_post_processors

    @wraps(make_processors)
    def make_local_processors(policy_cfg, **kwargs):
        if policy_cfg.type == "smolvla" and kwargs.get("pretrained_path"):
            tokenizer = Path(policy_cfg.vlm_model_name)
            if not (tokenizer / "tokenizer.json").is_file():
                raise FileNotFoundError(f"Pinned SmolVLA tokenizer is missing at {tokenizer}")
            overrides = dict(kwargs.get("preprocessor_overrides", {}))
            overrides["tokenizer_processor"] = {**overrides.get("tokenizer_processor", {}),
                                                "tokenizer_name": str(tokenizer)}
            kwargs["preprocessor_overrides"] = overrides
        return make_processors(policy_cfg, **kwargs)

    lerobot_train.make_pre_post_processors = make_local_processors
    install_checkpoint_guard()


if __name__ == "__main__":
    from policy_runtime_device import verify_runtime

    verify_runtime(os.environ.get("SMART_SWARM_TRAINING_DEVICE", "cpu"))
    configure()
    lerobot_train.main()
