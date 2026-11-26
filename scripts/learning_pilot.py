"""Smart Swarm learning pilot entry points using native research admission."""

import argparse
import json
import os
from pathlib import Path
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from smart_swarm.cli import settings
from smart_swarm.episode import atomic, load
from smart_swarm.rendering import configure_renderer
from smart_swarm.research_jobs import admitted, execute
from smart_swarm.training import offline

WORK = ROOT.parent.parent / "work"
EVIDENCE = ROOT / "evidence/learning-pilot-v3"
MOTOR_RAW = WORK / "motor-aligned-v3-raw"
MOTOR_FIT = WORK / "motor-aligned-v3-fit"
MOTOR_FULL = WORK / "motor-aligned-v3-full"
MOTOR_CLOCK_FIT = WORK / "motor-clock-v4-fit"
MOTOR_CLOCK_FULL = WORK / "motor-clock-v4-full"
MOTOR_CLOCK_SOURCE = WORK / "motor-clock-v4-fit-source"
STORIES = WORK / "evidence-pilot-v3"
GRAPHICS_STAGES = ("motor-data", "act-full", "smolvla-full")


def fresh_resets():
    import numpy as np

    recipes = []
    for seed in range(32_000, 32_020):
        generator = np.random.default_rng(seed)
        offsets = generator.uniform(-.004, .004, size=2)
        recipes.append({"seed": seed, "target_marker": "violet" if seed % 2 else "cyan",
                        "cube_xy_offset_m": offsets.tolist(), "sliding_friction": float(generator.uniform(.94, 1.06))})
    return {"scope": "untested fresh evaluation reset recipes", "split": "test", "resets": recipes}


def motor_data(config, grant, heartbeat):
    for split in ("train", "validation"):
        folder = MOTOR_RAW / split
        folder.mkdir(parents=True, exist_ok=True)
        for original in (WORK / "motor-pilot" / split).iterdir():
            target = folder / original.name
            if (original / "manifest.json").is_file() and not target.exists():
                target.symlink_to(original.resolve(), target_is_directory=True)
    destination = EVIDENCE / "motor-full-capture.json"
    report = load(destination) if destination.is_file() else {
        "wall_cap_s": 3600, "rollout_timeout_s": 100, "storage_cap_mib": 6800,
        "max_attempts_per_requested": 5, "splits": {}, "generation_elapsed_s": 0.}
    prior_grant = report.get("reservation")
    if "reservation_history" not in report:
        report["reservation_history"] = [prior_grant] if prior_grant else []
    render_grant = {**grant, "egl_vendor_manifest": os.environ["__EGL_VENDOR_LIBRARY_FILENAMES"],
                    "generation_elapsed_before_s": report["generation_elapsed_s"]}
    report["reservation"] = render_grant
    report["reservation_history"].append(render_grant)
    interrupted = report.pop("capture_in_progress", None)
    if interrupted:
        charge = min(interrupted["remaining_s"], max(0., time.time() - interrupted["started_unix"]))
        report["generation_elapsed_s"] += charge
        report.setdefault("interrupted_capture_charges", []).append({**interrupted, "charged_s": charge})
    atomic(destination, report)
    for split, size in (("train", 80), ("validation", 20)):
        capture_motor_split(config, split, size, destination, report, heartbeat)
    data_ready = len(report["splits"]) == 2 and all(row["status"] == "complete" for row in report["splits"].values())
    if data_ready:
        output, option = full_motor_version()
        official_export(MOTOR_RAW, output, option, EVIDENCE / "motor-full-export.log", heartbeat, 1800)
        report["dataset"] = str(output)
        atomic(destination, report)
    return report


def capture_motor_split(config, split, size, destination, report, heartbeat):
    from smart_swarm.motor import generate as generate_motor
    from smart_swarm.motor import interrupted_summary

    remaining = int(report["wall_cap_s"] - report["generation_elapsed_s"])
    if remaining < 1 or report["splits"].get(split, {}).get("status") == "complete":
        return
    started = time.monotonic()
    report["capture_in_progress"] = {"split": split, "started_unix": time.time(), "remaining_s": remaining}
    atomic(destination, report)
    try:
        report["splits"][split] = generate_motor(config["asset_root"], MOTOR_RAW, split, size,
                                                  remaining, report["storage_cap_mib"], heartbeat)
    except KeyboardInterrupt:
        report["splits"][split] = interrupted_summary(MOTOR_RAW, split, size)
        raise
    finally:
        report["generation_elapsed_s"] += time.monotonic() - started
        report.pop("capture_in_progress")
        atomic(destination, report)


def full_motor_version():
    status = EVIDENCE / "act-clock-fit.json"
    if status.is_file() and load(status).get("validation", {}).get("matched_sample_trace", {}).get("counterfactual_issue_index"):
        return MOTOR_CLOCK_FULL, "--clocked"
    return MOTOR_FULL, "--aligned"


def verified_export(source, output, option):
    from smart_swarm.motor_input import CLOCKED_PROFILE, PROFILE
    from smart_swarm.motor_visual_profile import match_visual

    profile = CLOCKED_PROFILE if option == "--clocked" else PROFILE
    splits = ("train", "validation", "test")
    expected = {split: {path.parent.name: (load(path)["sha256"], str(path.parent.resolve()))
                        for path in (source / split).glob("*/manifest.json")} for split in splits}
    manifests = {split: output / split / "smart-swarm-lineage.json" for split in splits}
    if any(expected[split] and not manifests[split].is_file() for split in splits):
        return False
    for split, path in manifests.items():
        if path.is_file():
            lineage = load(path)
            raw = [load(p) for p in (source / split).glob("*/manifest.json")]
            match_visual(lineage, *raw)
            observed = {row["source_world_episode_id"]: (row["source_sha256"], row["raw_path"])
                        for row in lineage["episodes"]}
            if (lineage["input_profile"] != profile or observed != expected[split]
                    or len(observed) != len(lineage["episodes"]) or not lineage["episode_padding_mask_checked"]):
                raise ValueError(f"Native export lineage differs from the admitted {split} recording set at {path}")
    return bool(expected["train"] and expected["validation"])


def official_export(source, output, option, log, heartbeat, wall_seconds):
    if verified_export(source, output, option):
        return
    attempt = uuid.uuid4().hex[:12]
    pending = output.with_name(output.name + "-unpublished-" + attempt)
    runtime = load(ROOT / "configs/training.json")["runtime"]
    command = [runtime, str(ROOT / "scripts/export_lerobot.py"), "--source", str(source),
               "--output", str(pending), option]
    env = {**offline(), "PYTHONPATH": str(ROOT / "src")}
    with log.open("a") as stream:
        print(f"Official export attempt {attempt}: {pending}", file=stream, flush=True)
        code = execute(command, env, stream, time.monotonic() + wall_seconds, heartbeat)
    if code != 0:
        raise RuntimeError(f"Official motor exporter failed. See {log}")
    if not verified_export(source, pending, option):
        raise RuntimeError(f"Official motor exporter did not produce both verified splits. See {log}")
    if output.exists():
        output.rename(output.with_name(output.name + "-unpublished-previous-" + attempt))
    pending.rename(output)


def clocked_motor_data(grant, heartbeat):
    from smart_swarm.motor_input import CLOCKED_PROFILE

    audit = load(EVIDENCE / "issue-clock-v4-parity.json")
    if audit["input_profile"] != CLOCKED_PROFILE or audit["source_counts"] != {"train": 5, "validation": 2}:
        raise ValueError("The clock correction must pass parity on exactly five fitting and two validation sources")
    for split in ("train", "validation"):
        folder = MOTOR_CLOCK_SOURCE / split
        folder.mkdir(parents=True, exist_ok=True)
        rows = load(MOTOR_FIT / split / "smart-swarm-lineage.json")["episodes"]
        for row in rows:
            source = Path(row["raw_path"])
            target = folder / row["source_world_episode_id"]
            if not target.exists():
                target.symlink_to(source, target_is_directory=True)
    official_export(MOTOR_CLOCK_SOURCE, MOTOR_CLOCK_FIT, "--clocked", EVIDENCE / "motor-clock-fit-export.log", heartbeat, 420)
    profiles = {split: load(MOTOR_CLOCK_FIT / split / "smart-swarm-lineage.json")
                for split in ("train", "validation")}
    if any(entry["input_profile"] != CLOCKED_PROFILE for entry in profiles.values()):
        raise ValueError("Official export does not have the requested clocked profile")
    result = {"dataset": str(MOTOR_CLOCK_FIT), "input_profile": CLOCKED_PROFILE, "parent": str(MOTOR_FIT),
              "world_counts": {split: len(entry["episodes"]) for split, entry in profiles.items()},
              "frame_counts": {split: entry["frames"] for split, entry in profiles.items()}, "reservation": grant}
    atomic(EVIDENCE / "motor-clock-fit-export.json", result)
    return result


def smoke_stories(config, heartbeat):
    from smart_swarm.evidence_portfolio import capture_group

    root = WORK / ("evidence-v3-probe-" + uuid.uuid4().hex[:8])
    root.mkdir(exist_ok=False)
    outcomes = []
    for index in range(6):
        heartbeat()
        family = f"F{index + 1}"
        status = {"family": family, "probe_directory": str(root)}
        try:
            status["records"] = capture_group(config["asset_root"], root, ("train", family, 0),
                                              45001 + 2 * index, time.monotonic() + 100)
        except (OSError, RuntimeError, ValueError, TimeoutError) as error:
            status["error"] = str(error)
        outcomes.append(status)
        atomic(EVIDENCE / "story-probes.json", outcomes)
    return outcomes


def story_learn(config, grant, heartbeat):
    portfolio = STORIES / "portfolio.json"
    if not portfolio.is_file() or not load(portfolio)["complete"]:
        raise ValueError("Full ablation awaits the complete grouped story portfolio")
    status = EVIDENCE / "story-learning.json"
    if status.is_file():
        return load(status)
    root = config["data_root"] / "training-runs" / ("temporal-evidence-v3-" + uuid.uuid4().hex[:12])
    root.mkdir(exist_ok=False, parents=True)
    runtime = load(ROOT / "configs/training.json")["runtime"]
    argv = [runtime, str(ROOT / "scripts/learn_evidence_v3.py"), "--config", str(ROOT / "configs/evidence-pilot.json"),
            "--portfolio", str(portfolio), "--output", str(root)]
    env = {**offline(), "PYTHONPATH": str(ROOT / "src"), "SMART_SWARM_VERIFIED_RESEARCH_RESERVATION": grant["job_id"]}
    record = {"job": str(root), "status": "training", "reservation": grant, "dataset": str(portfolio)}
    atomic(status, record)
    try:
        code = evidence_phases(argv, env, root, heartbeat)
        record["status"] = "complete" if code == 0 else "failed"
    except KeyboardInterrupt:
        record["status"] = "interrupted"
        raise
    except Exception as error:
        record.update(status="inconclusive", reason=str(error))
    finally:
        fit = root / "models-fit.json"
        if fit.is_file():
            record["frozen_fit_report"] = str(fit)
        if (root / "models.json").is_file():
            record.update(load(root / "models.json"))
        comparison = root / "calibrated-rgb-particle-filter.json"
        if comparison.is_file():
            record["calibrated_particle_report"] = str(comparison)
        atomic(status, record)
    return record


def evidence_phases(argv, env, root, heartbeat):
    deadline = time.monotonic() + 3900
    with (root / "training.log").open("w") as stream:
        code = execute([*argv, "--phase", "fit"], env, stream, deadline, heartbeat)
    if code != 0:
        return code
    with (root / "evaluation.log").open("w") as stream:
        return execute([*argv, "--phase", "evaluate"], env, stream, deadline, heartbeat)


def invoke(config, stage, *, gpu_render=False):
    if gpu_render and stage not in GRAPHICS_STAGES:
        raise ValueError("GPU rendering requires motor-data, act-full or smolvla-full")
    EVIDENCE.mkdir(exist_ok=True, parents=True)
    atomic(EVIDENCE / "fresh-resets.json", fresh_resets())
    if stage in ("act-fit", "act-clock-fit"):
        configure_renderer(config)
        from smart_swarm.motor_pilot_run import run_fit

        clocked = stage == "act-clock-fit"
        return run_fit(config, MOTOR_CLOCK_FIT if clocked else MOTOR_FIT, clocked=clocked)
    with admitted(config, "smart-swarm-v3-" + stage, gpu_render=gpu_render) as (grant, heartbeat):
        configure_renderer(config, grant=grant)
        if stage == "motor-data":
            return motor_data(config, grant, heartbeat)
        if stage == "motor-clock-data":
            return clocked_motor_data(grant, heartbeat)
        if stage in ("act-full", "smolvla-full"):
            from smart_swarm.motor_full import run as train_full

            dataset, _ = full_motor_version()
            return train_full(config, stage.split("-")[0], dataset, EVIDENCE, grant, heartbeat)
        if stage == "story-probe":
            return smoke_stories(config, heartbeat)
        if stage == "story-data":
            from smart_swarm.evidence_portfolio import generate as generate_stories

            result = generate_stories(config["asset_root"], STORIES, heartbeat)
            digest = {name: result[name] for name in ("limits", "attempted", "accepted_independent", "rejected",
                                                           "recorded_variants", "groups_by_split", "label_masks_by_split")}
            atomic(EVIDENCE / "story-capture.json", {**digest, "reservation": grant, "portfolio": str(STORIES / 'portfolio.json')})
            return digest
        return story_learn(config, grant, heartbeat)


def saved_status():
    paths = {"act_fitting": EVIDENCE / "act-fit.json", "act_clock_fit": EVIDENCE / "act-clock-fit.json",
             "motor_dataset": EVIDENCE / "motor-full-capture.json",
             "act_full": EVIDENCE / "act-full.json", "smolvla_full": EVIDENCE / "smolvla-full.json",
             "story_dataset": STORIES / "portfolio.json", "learned_evidence": EVIDENCE / "story-learning.json"}
    result = {}
    for kind, source in paths.items():
        if source.is_file():
            record = load(source)
            keys = ("stage", "status", "dataset", "checkpoint", "optimizer_updates", "matching_starts", "successful",
                    "accepted_independent", "attempted", "rejected", "groups_by_split", "job_directory", "job")
            result[kind] = {"record": str(source), **{name: record[name] for name in keys if name in record}}
    for kind in ("act_fitting", "act_clock_fit"):
        fitting = result.get(kind, {})
        if fitting.get("stage") == "fitting":
            from smart_swarm.motor_pilot_run import optimizer_rows

            root = Path(fitting["job_directory"])
            updates = optimizer_rows(root / "training.log")
            fitting["latest_logged_update"] = updates[-1] if updates else None
            fitting["saved_checkpoints"] = sorted(path.name for path in (root / "training/checkpoints").glob("[0-9]*"))
    if "learned_evidence" in result:
        evidence = result["learned_evidence"]
        evidence["latest_logged_updates"] = latest_evidence_updates(Path(evidence["job"]) / "training.log")
    return result


def latest_evidence_updates(log):
    updates = {}
    if log.is_file():
        with log.open() as stream:
            for line in stream:
                if line.startswith('{"event": "optimizer_update"') and line.endswith("}\n"):
                    row = json.loads(line)
                    updates[row["model"]] = {"update": row["update"], "loss": row["loss"],
                                             "opportunity": row.get("opportunity", row["update"])}
    return updates


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("act-fit", "act-clock-fit", "motor-data", "motor-clock-data", "act-full", "smolvla-full", "story-probe",
                                          "story-data", "story-learn", "reset-recipes", "status"))
    parser.add_argument("--gpu-render", action="store_true", help="Request the native graphics lease for motor capture or evaluation.")
    args = parser.parse_args()
    if args.gpu_render and args.stage not in GRAPHICS_STAGES:
        parser.error("--gpu-render requires motor-data, act-full or smolvla-full")
    if args.stage == "reset-recipes":
        EVIDENCE.mkdir(exist_ok=True, parents=True)
        atomic(EVIDENCE / "fresh-resets.json", fresh_resets())
    elif args.stage == "status":
        print(json.dumps(saved_status(), indent=2))
    else:
        result = invoke(settings(), args.stage, gpu_render=args.gpu_render)
        print(result if isinstance(result, list) else {k: v for k, v in result.items() if k != "fitting_cases"})


if __name__ == "__main__":
    main()
