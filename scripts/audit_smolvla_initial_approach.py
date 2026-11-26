"""Compare the original SmolVLA initialization and fitted approach on matched saved inputs."""

import argparse
from hashlib import file_digest, sha256
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from time import monotonic

import numpy as np

from smart_swarm.cli import ROOT, settings
from smart_swarm.contract import ProtocolError
from smart_swarm.episode import atomic, json_bytes, load, sha
from smart_swarm.research_jobs import admitted, execute
from smart_swarm.training import offline

PILOT = ROOT / "evidence/learning-pilot-v3"
PAST = PILOT / "smolvla-teacher-chunk-closed-loop-comparison.json"
PAST_SHA = "581124dccd99d89fb8891006eb6b6d376d24c77d0ca21c81eb1042c50cf4653c"
FIT = PILOT / "smolvla-full.json"
FIT_SHA = "77bcac47f07038d6769bebe461d829f663c20ee91291d165c949a58ea2fdf6a7"
OUT = PILOT / "smolvla-initial-approach-comparison.json"
CACHE_FAILURE = PILOT / "smolvla-initial-cache-proof-failed.json"
CACHE_FAILURE_SHA = "98c9d17b19ea594aaa623ec9b78a9e4d0573c471ab8a799aa4c87b56b299d2b9"
CACHE_LOOKUP_CHARGE_S = 1.


def original_run():
    if sha(PAST) != PAST_SHA or sha(FIT) != FIT_SHA:
        raise ValueError("The original fitted and archived comparison receipts changed")
    fitted, previous = load(FIT), load(PAST)
    checkpoint = Path(fitted["checkpoint"])
    root = Path(fitted["job_directory"])
    validation = root / "validation" / checkpoint.parent.name / "validation.json"
    run_path = root / "run.json"
    launch = load(run_path)
    if (fitted["optimizer_updates"] != 85 or launch["requested_updates"] != 85
            or launch["returncode"] != 0 or launch["dataset"] != fitted["dataset"]
            or launch["device"] != "cpu" or Path(launch["checkpoint"]).resolve() != checkpoint
            or sha(validation) != previous["validation_sha256"] or str(checkpoint) != previous["checkpoint"]):
        raise ValueError("The original SmolVLA launch, selected weights and development receipt differ")
    return validation, run_path, launch, previous, fitted


def archived_pipeline(launch, scratch):
    from audit_smol_init import native_config

    before = sys.argv
    expected_output = "--output_dir="
    original = launch["argv"][1:]
    if sum(value.startswith(expected_output) for value in original) != 1:
        raise ValueError("The archived trainer command no longer declares one output directory")
    try:
        sys.argv = [f"{expected_output}{scratch}" if value.startswith(expected_output) else value for value in original]
        result = native_config()
    finally:
        sys.argv = before
    if (result.resume or result.policy.type != "smolvla" or result.policy.device != "cpu"
            or result.seed != 3 or result.dataset.use_imagenet_stats or result.steps != launch["requested_updates"]):
        raise ValueError("The recreated policy configuration is outside its original training admission")
    return result


def archived_weight_identity(policy, dataset):
    artifact = Path(policy.pretrained_path) / "model.safetensors"
    previous = load(dataset.parent / "smolvla-development-processor.json")
    runtime = load(ROOT / "configs/training.json")
    if (str(policy.pretrained_path) != previous["policy_checkpoint"] or runtime["weights"]["smolvla"] != str(policy.pretrained_path)
            or policy.vlm_model_name != previous["backbone"] or not policy.load_vlm_weights
            or artifact.stat().st_size != previous["weights_present_bytes"]):
        raise ValueError("The original cached pretrained snapshot or backbone changed")
    with artifact.open("rb") as stream:
        digest = file_digest(stream, "sha256").hexdigest()
    if (artifact.resolve().name != previous["weights_cache_object"] or not artifact.is_symlink()
            or digest != artifact.readlink().name):
        raise ValueError("The original SmolVLA weight content proof does not match the cached blob")
    return {"pretrained_snapshot": str(policy.pretrained_path), "pretrained_content_sha256": digest,
            "local_cache_object": artifact.resolve().name,
            "resolved_backbone": policy.vlm_model_name, "verified_weight_bytes": artifact.stat().st_size}


def load_initial(launch, checkpoint, scratch):
    from audit_smol_init import processors
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.scripts import lerobot_train
    from policy_worker import make_pre_post_processors, processor_devices
    from train_cpu import bind_smol_inputs

    cfg = archived_pipeline(launch, scratch)
    train = Path(cfg.dataset.root)
    metadata = LeRobotDataset("smart-swarm/train", root=train).meta
    if train != Path(launch["dataset"]) / "train" or metadata.total_episodes != 80:
        raise ValueError("The comparison must use the exact 80-source processor normalization")
    proof = archived_weight_identity(cfg.policy, train.parent)
    bind_smol_inputs(cfg.policy, metadata)
    rebuilt = processors(cfg.policy, metadata.stats)
    fitted = PreTrainedConfig.from_pretrained(str(checkpoint))
    frozen_pipeline = make_pre_post_processors(fitted, pretrained_path=str(checkpoint),
                                              **processor_devices(checkpoint, "cpu"))
    fields = ("input_features", "output_features", "normalization_mapping", "num_steps",
              "n_action_steps", "chunk_size", "resize_imgs_with_padding", "use_cache")
    if any(getattr(cfg.policy, key) != getattr(fitted, key) for key in fields):
        raise ValueError("The original and fitted inference profiles are not comparable")
    lerobot_train.set_seed(cfg.seed)
    model = lerobot_train.make_policy(cfg=cfg.policy, ds_meta=metadata, rename_map=cfg.rename_map).eval()
    return model, rebuilt, frozen_pipeline, proof


def verified_processors(window, target, rebuilt, frozen):
    import torch
    from policy_worker import example_batch

    plain = example_batch(window["payload"])
    value = torch.tensor(target, dtype=torch.float32)
    initial = rebuilt[0]({**plain, "action": value})
    trained = frozen[0]({**plain, "action": value})
    names = [name for name, tensor in initial.items() if isinstance(tensor, torch.Tensor)]
    if set(names) != {name for name, tensor in trained.items() if isinstance(tensor, torch.Tensor)} or any(
            not torch.equal(initial[name], trained[name]) for name in names):
        raise ValueError("Initial and fitted processor inputs differ on the same recorded sample")
    restored = rebuilt[1](initial["action"])
    if not torch.equal(restored, frozen[1](trained["action"])) or (restored.flatten() - value).abs().max() >= 1e-4:
        raise ValueError("The original normalization no longer decodes the same native teacher action")
    return sorted(names)


def compare_start(case, previous, model, rebuilt, saved_pipeline):
    import torch
    from audit_smolvla_teacher_chunk import case_metadata, compare_chunk
    from policy_worker import example_batch, smolvla_chunk

    metadata = case_metadata(case)
    if (any(previous[key] != value for key, value in metadata.items()) or not previous["matched_original_initial_chunk"]
            or previous["chunks"][0]["refresh_issue"] != 0):
        raise ValueError("The paired comparison must retain the original two exact validation starts")
    window = case["windows"][0]
    verified = verified_processors(window, case["requested"][0], rebuilt, saved_pipeline)
    torch.manual_seed(case["seed"])
    model.reset()
    start = monotonic()
    candidates = smolvla_chunk(example_batch(window["payload"]), model, *rebuilt)
    timing = monotonic() - start
    initial = compare_chunk(case, 0, np.asarray(candidates))
    fitted = previous["chunks"][0]
    rows = []
    for now, later in zip(initial["sampled_actions"], fitted["sampled_actions"], strict=True):
        if now["issue"] != later["issue"] or now["teacher_action"] != later["teacher_action"]:
            raise ValueError("Initial and fitted actions must use the same validated issue and teacher target")
        rows.append({"issue": now["issue"], "teacher": now["teacher_action"],
                     "initial": now["policy_with_teacher_input"], "fitted_after_85_updates": later["policy_with_teacher_input"]})
    return {"seed": case["seed"], "marker": case["marker"], "window": metadata["windows"][0],
            "verified_processor_tensors": verified, "paired_native_samples": rows, "initial_chunk_seconds": timing,
            "initial_full_chunk_sha256": sha256(json_bytes(candidates)).hexdigest(),
            "initial_mae": initial["native_mae_to_teacher"], "fitted_mae": fitted["native_mae_to_teacher"]}


def worker(scratch):
    from audit_smolvla_teacher_chunk import prepare_inputs
    from policy_runtime_device import verify_runtime

    verify_runtime("cpu")
    validation, run_path, launch, previous, fitted = original_run()
    checkpoint, cases = prepare_inputs(validation)
    model, rebuilt, saved, weights = load_initial(launch, checkpoint, scratch)
    original = previous["teacher_windows"]
    compared = [compare_start(case, row, model, rebuilt, saved) for case, row in zip(cases, original, strict=True)]
    return {**weights, "archived_training_launch_sha256": sha(run_path), "validation_sha256": sha(validation),
            "fitted_checkpoint_sha256": fitted["fresh"]["checkpoint_sha256"], "cases": compared,
            "initial_profile_reconstructed_with_official_trainer": True, "fitted_actions_from_previous_receipt": True}


def bounded(output):
    _, _, _, previous, _ = original_run()
    if sha(CACHE_FAILURE) != CACHE_FAILURE_SHA:
        raise ValueError("Preserve the original stopped cache check before continuing")
    additional_charges = {"cache_guard_stop": load(CACHE_FAILURE)["charged_seconds"], "cache_identity_read": CACHE_LOOKUP_CHARGE_S}
    available = previous["wall_limit_s"] - previous["elapsed_s"] - sum(additional_charges.values())
    runtime = load(ROOT / "configs/training.json")["runtime"]
    log = output.with_suffix(".log")
    result = {"status": "measured", "scope": "Paired original initialization and fitted actions on two archived validation starts",
              "prior_teacher_receipt_sha256": PAST_SHA, "prior_teacher_seconds": previous["elapsed_s"],
              "previous_attempt_charges_s": additional_charges, "cache_failure_sha256": CACHE_FAILURE_SHA,
              "cumulative_wall_limit_s": previous["wall_limit_s"], "additional_wall_limit_s": available,
              "optimizer_updates": 0, "fresh_resets_read": 0, "closed_loop_trials": 0, "robot_commands_sent": 0}
    started = monotonic()
    try:
        with admitted(settings(), "smart-swarm-smolvla-initial-approach", archive_only=True) as (grant, heartbeat):
            result["reservation"] = grant
            with TemporaryDirectory(prefix="smart-swarm-smolvla-initial-") as directory, log.open("x") as stream:
                work = Path(directory)
                target = work / "worker.json"
                argv = [runtime, str(Path(__file__).resolve()), "--worker", str(work), "--output", str(target)]
                code = execute(argv, {**offline(), "PYTHONPATH": str(ROOT / "src")}, stream, started + available, heartbeat)
                if code != 0:
                    raise ValueError(f"Original initializer replay exited {code}. See {log}")
                result.update(load(target))
    except (OSError, ValueError, RuntimeError, TimeoutError, ProtocolError) as error:
        result.update(status="failed", reason=str(error))
    elapsed = monotonic() - started
    return {**result, "charged_seconds": elapsed, "remaining_original_seconds": available - elapsed}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--output", type=Path, default=OUT, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if (args.worker is None and args.output.resolve() != OUT) or args.output.exists() or args.output.with_suffix(".log").exists():
        parser.error("Use the original cumulative comparison receipt path once")
    result = worker(args.worker / "unused-training") if args.worker else bounded(args.output)
    atomic(args.output, result)
    print(args.output)
    raise SystemExit(2 if result.get("status") == "failed" else 0)


if __name__ == "__main__":
    main()
