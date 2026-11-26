"""Reproduce the compact SmolVLA run's committed training sample order."""

import argparse
from collections import Counter
from importlib.metadata import version
from inspect import getfile
from itertools import islice
from pathlib import Path
import re
from statistics import median

from smart_swarm.cli import ROOT, settings
from smart_swarm.episode import atomic, load, sha
from smart_swarm.research_jobs import admitted


JOB = "smolvla-pipeline-d9510148a574"
INVENTORY = ROOT / "evidence/learning-pilot-v3/training-inventory-final.json"


def training_log(path, updates):
    log = path.read_text()
    pattern = r"step:(\d+) smpl:(\d+) [^\r\n]*?updt_s:(\d+\.\d+) data_s:(\d+\.\d+)"
    records = [(int(step), int(samples), float(update), float(data))
               for step, samples, update, data in re.findall(pattern, log)]
    expected = list(range(1, updates + 1))
    if [(step, samples) for step, samples, _, _ in records] != [(n, n) for n in expected]:
        raise ValueError("The training log must record one sample for each saved update")
    parameters, = re.findall(r"num_learnable_params=(\d+)", log)
    if "Effective batch size: 1 x 1 = 1" not in log:
        raise ValueError("The original trainer did not report single-process, single-sample batches")
    return records, int(parameters)


def archived_run():
    job, = (item for item in load(INVENTORY)["jobs"] if item["job"] == JOB)
    step = job["step_receipt"]
    training = job["training_receipt"]
    config = job["configuration_and_processors"]["train_config.json"]
    lineage = job["dataset"]["splits"]["train"]["lineage"]
    inputs = {"step": step, "training": training, "config": config, "lineage": lineage}
    if any(sha(Path(entry["path"])) != entry["sha256"] for entry in inputs.values()):
        raise ValueError("A saved checkpoint configuration, receipt or dataset lineage has changed")
    checkpoint_step = load(Path(step["path"]))
    run = load(Path(training["path"]))
    updates = job["optimizer_updates"]
    if (checkpoint_step != {"step": updates, "num_processes": 1, "batch_size": 1}
            or run["requested_updates"] != updates or run["status"] != "trained" or run["device"] != "cpu"):
        raise ValueError("Expected the completed single-process compact CPU run")
    return job, inputs, run


def original_dataset(config_path, root, updates):
    from lerobot.configs.train import TrainPipelineConfig
    from lerobot.datasets.factory import make_train_eval_datasets
    from lerobot.scripts import lerobot_train

    cfg = TrainPipelineConfig.from_pretrained(config_path, local_files_only=True)
    if (version("lerobot") != "0.6.1" or cfg.resume or cfg.steps != updates or cfg.batch_size != 1
            or cfg.num_workers != 0 or cfg.seed != 3 or cfg.dataset.streaming or cfg.dataset.episodes is not None
            or cfg.dataset.eval_split != 0. or cfg.dataset.image_transforms.enable
            or Path(cfg.dataset.root) != root or cfg.policy.type != "smolvla" or cfg.policy.device != "cpu"):
        raise ValueError("Saved configuration is outside the verified deterministic CPU run")
    dataset, evaluation = make_train_eval_datasets(cfg)
    if evaluation is not None or dataset.num_episodes != 80 or dataset.num_frames != 40800:
        raise ValueError("The original compact training split must contain 80 episodes and 40,800 rows")
    return cfg, dataset, Path(getfile(lerobot_train))


def data_order(cfg, dataset):
    from accelerate import Accelerator
    from lerobot.datasets import EpisodeAwareSampler
    from torch.utils.data import DataLoader

    active = cfg.trainable_config
    episodes = dataset.meta.episodes
    sampler = EpisodeAwareSampler(episodes["dataset_from_index"], episodes["dataset_to_index"],
                                  episode_indices_to_use=dataset.episodes,
                                  drop_n_last_frames=getattr(active, "drop_n_last_frames", 0), shuffle=True,
                                  seed=cfg.seed, absolute_to_relative_idx=dataset.absolute_to_relative_idx)
    if cfg.steps >= len(sampler) or dataset.meta.has_language_columns:
        raise ValueError("The audit requires the original first-epoch default collation route")
    accelerator = Accelerator(cpu=True, step_scheduler_with_optimizer=False)
    if accelerator.num_processes != 1 or accelerator.device.type != "cpu":
        raise ValueError("The replay needs the original single-process CPU loader")
    loader = DataLoader(dataset, num_workers=cfg.num_workers, batch_size=cfg.batch_size,
                        shuffle=False, sampler=sampler, pin_memory=False, drop_last=False)
    return accelerator.prepare(loader), len(sampler), Path(getfile(EpisodeAwareSampler))


def exposure(step, batch, ancestry, episode_ranges, chunk):
    episode = batch["episode_index"].item()
    row_index = batch["index"].item()
    issue = batch["frame_index"].item()
    start, end = [int(values[episode]) for values in episode_ranges]
    entry = ancestry[episode]
    usable = min(chunk, end - row_index)
    if (entry["frame_count"] != end - start or row_index != start + issue or not start <= row_index < end
            or batch["action_is_pad"].shape != (1, chunk)
            or batch["action_is_pad"][0].tolist() != [False] * usable + [True] * (chunk - usable)):
        raise ValueError(f"Incorrect source row or action padding at committed step {step}")
    return {"step": step, "dataset_row": row_index, "issue_index": issue, "dataset_episode_index": episode,
            "source_world": entry["source_world_episode_id"], "source_group": entry["scene_group_id"],
            "source_sha256": entry["source_sha256"], "target_marker": entry["target_marker"],
            "unpadded_actions": usable, "timestamp_s": round(batch["timestamp"].item(), 6)}


def summarize(rows, timing, groups, horizon):
    distinct = Counter(row["source_group"] for row in rows)
    update = median(row[2] for row in timing)
    data = median(row[3] for row in timing)
    update_and_data = median(row[2] + row[3] for row in timing)
    first = [row for row in rows if row["issue_index"] < horizon]
    return {"committed_updates": len(rows), "unique_sample_starts": len({row["dataset_row"] for row in rows}),
            "source_groups_exposed": len(distinct), "source_groups_available": groups,
            "sample_starts_per_source_group": dict(sorted(distinct.items())),
            "target_marker_counts": dict(Counter(row["target_marker"] for row in rows)),
            "startup_horizon_updates": len(first), "startup_horizon_groups": len({row["source_group"] for row in first}),
            "startup_horizon_by_target": dict(Counter(row["target_marker"] for row in first)),
            "exact_reset_start_updates": sum(row["issue_index"] == 0 for row in rows),
            "updates_with_padded_actions": sum(row["unpadded_actions"] < horizon for row in rows),
            "cpu_update_median_s": update, "cpu_data_loading_median_s": data,
            "cpu_update_plus_data_median_s": update_and_data,
            "estimated_cpu_seconds_for_2000_updates_excluding_setup_saves_and_evaluation": 2000 * update_and_data}


def audit(heartbeat):
    job, inputs, run = archived_run()
    log = Path(run["run_directory"]) / "training.log"
    timing, parameters = training_log(log, job["optimizer_updates"])
    lineage = Path(inputs["lineage"]["path"])
    cfg, dataset, trainer = original_dataset(inputs["config"]["path"], lineage.parent, len(timing))
    loader, sample_space, sampler = data_order(cfg, dataset)
    ancestry = {row["dataset_episode_index"]: row for row in load(lineage)["episodes"]}
    episode_ranges = [dataset.meta.episodes[key] for key in ("dataset_from_index", "dataset_to_index")]
    chunk = cfg.policy.chunk_size
    rows = []
    for step, batch in enumerate(islice(loader, len(timing)), 1):
        if step % 10 == 1:
            heartbeat()
        rows.append(exposure(step, batch, ancestry, episode_ranges, chunk))
    if len(rows) != len(timing):
        raise ValueError("The replay did not yield all committed batches")
    summary = summarize(rows, timing, len(ancestry), chunk)
    if summary["unique_sample_starts"] != len(timing):
        raise ValueError("The original first-epoch samples must be drawn without replacement")
    return {"scope": "Deterministic replay of the completed run's map-style loader. Original logs confirm the update count.",
            "status": "verified_deterministic_reconstruction", "job": JOB, "checkpoint": job["weights"],
            "inputs": inputs, "training_log_sha256": sha(log), "original_run_wall_s": run["elapsed_s"],
            "runtime_versions": {name: version(name) for name in ("lerobot", "torch", "accelerate", "numpy")},
            "source_sha256": {"trainer": sha(trainer), "sampler": sha(sampler),
                              "audit": sha(Path(__file__))},
            "sample_space": sample_space, "action_horizon": chunk, "trainable_parameters": parameters,
            "optimizer_updates_run_by_audit": 0, "optimizer": load(Path(inputs["config"]["path"]))["optimizer"],
            "summary": summary, "sample_order": rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        parser.error("Choose a new immutable audit output")
    with admitted(settings(), "smart-swarm-smolvla-training-exposure", archive_only=True) as (_, heartbeat):
        report = audit(heartbeat)
    atomic(args.out, report)
    print(args.out)


if __name__ == "__main__":
    main()
