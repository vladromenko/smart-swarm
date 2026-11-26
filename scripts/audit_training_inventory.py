#!/usr/bin/env python3
"""Reconcile the archived Smart Swarm optimizer launches and evaluation receipts."""

import argparse
from collections import Counter
from hashlib import file_digest, sha256
import json
from pathlib import Path

from smart_swarm.cli import ROOT, settings
from smart_swarm.episode import atomic, load

EVIDENCE = ROOT / "evidence"


def reference(path):
    with path.open("rb") as stream:
        return {"path": str(path), "sha256": file_digest(stream, "sha256").hexdigest()}


def motor_dataset(run):
    root = Path(run["dataset"])
    result = {}
    seen = set()
    for split in ("train", "validation"):
        path = root / split / "smart-swarm-lineage.json"
        manifest = load(path)
        sources = {row["scene_group_id"] for row in manifest["episodes"]}
        if seen & sources or len(sources) != run[f"{split}_worlds"]:
            raise ValueError(f"{root} has changed independent source membership")
        seen.update(sources)
        hashes = {row["source_world_episode_id"]: row["source_sha256"] for row in manifest["episodes"]}
        result[split] = {"groups": len(sources), "frames": manifest["frames"], "lineage": reference(path),
                         "source_world_sha256": hashes}
    return {"path": str(root), "splits": result}


def motor_evaluation(path):
    report = load(path)
    if "successful_resets" in report:
        cases = report["results"]
        success = sum(row["contained_and_released"] for row in cases)
        scope, labels = "preserved historical MuJoCo evaluation", {}
        expected, count, loss = report["successful_resets"], report["requested_resets"], report
    else:
        cases = report["fitting_cases"]
        success = sum(row["stable_containment"] for row in cases)
        scope = "matched MuJoCo fitting resets"
        labels = {"executor_profiles": sorted({row["profile"] for row in cases}),
                  "first_failure_stages": dict(Counter(row["first_failed_stage"] for row in cases if row["first_failed_stage"]))}
        expected, count, loss = report["successful"], report["matching_starts"], report["validation"]
    if success != expected or len(cases) != count or len({row["seed"] for row in cases}) != count:
        raise ValueError(f"{path} does not match the saved per-reset records")
    return {"scope": scope, "resets": count, "successes": success, **labels,
            "offline_validation_mean_loss": loss["offline_validation_mean_loss"], "receipt": reference(path)}


def checkpoint_processors(folder):
    names = ("config.json", "train_config.json", "policy_preprocessor.json", "policy_postprocessor.json")
    selected = [folder / name for name in names] + sorted(folder.glob("policy_*_processor.safetensors"))
    return {path.name: reference(path) for path in selected}


def motor_training(log):
    run = load(log)
    directory = Path(run.get("run_directory", run.get("job_directory")))
    receipt = directory / "run.json"
    if load(receipt) != run or run["status"] != "trained" or run["returncode"] != 0:
        raise ValueError(f"{directory.name} does not match its completed training receipt")
    checkpoint = Path(run["checkpoint"]).resolve()
    step = checkpoint.parent / "training_state/training_step.json"
    updates = load(step)["step"]
    if updates != run["requested_updates"] or checkpoint.parent.name != f"{updates:06d}":
        raise ValueError(f"{directory.name} does not match its saved optimizer step")
    weight = reference(checkpoint / "model.safetensors")
    return {"job": directory.name, "model": load(checkpoint / "config.json")["type"],
            "optimizer_updates": updates, "checkpoint": str(checkpoint), "weights": weight,
            "configuration_and_processors": checkpoint_processors(checkpoint),
            "device": run["device"], "duration_s": run["elapsed_s"], "dataset": motor_dataset(run),
            "training_receipt": reference(receipt), "step_receipt": reference(step),
            "shared_reservation_recorded": "compute_reservation" in run}


def motor_job(log, evaluated, purpose):
    trained = motor_training(log)
    report = load(evaluated)
    weight = trained["weights"]["sha256"]
    if (Path(report["checkpoint"]).resolve() != Path(trained["checkpoint"])
            or report.get("optimizer_updates", trained["optimizer_updates"]) != trained["optimizer_updates"]
            or report.get("checkpoint_sha256", weight) != weight):
        raise ValueError(f"{trained['job']} original evaluation checkpoint or optimizer step changed")
    return {**trained, "purpose": purpose, "status": "trained_and_evaluated", "evaluation": motor_evaluation(evaluated)}


def snapshot(path):
    raw = path.read_bytes()
    return json.loads(raw), {"path": str(path), "sha256": sha256(raw).hexdigest()}


def compact_validation(root, trained):
    from run_compact_motor_comparison import audited_stage

    choice_file = root / "validation-choices.json"
    if not choice_file.is_file():
        return {"status": "awaiting_selected_validation"}
    choices, selected = snapshot(choice_file)
    choice, = choices
    expected = published_validation_seeds(trained["dataset"])
    if (choice["checkpoint"] != trained["checkpoint"] or choice["evaluation_role"] != "validation"
            or choice["action_transport"] != "verified-smolvla-chunk"
            or [row["seed"] for row in choice["resets"]] != expected):
        raise ValueError("Compact validation differs from the published source order or selected checkpoint")
    stats = audited_stage(choice["resets"], choice["successful_resets"], fresh=False)
    return {"status": "evaluated", **stats, "offline_validation_mean_loss": choice["offline_validation_mean_loss"],
            "receipt": selected}


def published_validation_seeds(dataset):
    lineage = load(Path(dataset["splits"]["validation"]["lineage"]["path"]))
    return [load(Path(row["raw_path"]) / "manifest.json")["seed"] for row in lineage["episodes"]]


def compact_fresh(root, trained, comparison):
    receipts = sorted(root.glob("fresh/*/evaluation.json"))
    if not receipts:
        return {"status": "awaiting_fresh_receipt"}
    path, = receipts
    report, observed = snapshot(path)
    if (report["evaluation_role"] != "fresh-test" or report["checkpoint"] != trained["checkpoint"]
            or report["checkpoint_sha256"] != trained["weights"]["sha256"] or report["requested_resets"] != 20):
        raise ValueError("Fresh comparison uses a different checkpoint or declared reset request")
    if comparison["status"] == "evaluated":
        from run_compact_motor_comparison import audit_completed_result

        result = audit_completed_result("smolvla", comparison)
        if any(comparison["fresh"][key] != report[key] for key in ("status", "results", "successful_resets")):
            raise ValueError("The completed comparison differs from its original fresh receipt")
        return {"status": "evaluated", **result["fresh"], "receipt": observed}
    return {"status": report["status"], "completed_resets": len(report["results"]),
            "requested_resets": report["requested_resets"], "receipt": observed}


def compact_smolvla(path):
    comparison, progress = snapshot(path)
    root = Path(comparison["job_directory"])
    trained = motor_training(root / "run.json")
    if (trained["model"] != "smolvla" or comparison["dataset"] != trained["dataset"]["path"]
            or not trained["shared_reservation_recorded"]):
        raise ValueError("Compact comparison differs from its shared CPU training receipt")
    validation = compact_validation(root, trained)
    fresh = compact_fresh(root, trained, comparison)
    return {**trained, "purpose": "bounded compact SmolVLA comparison on the published motor split",
            "status": "trained_and_evaluated" if comparison["status"] == "evaluated" else "trained",
            "training_cap": comparison["cap"],
            "evaluation": {"scope": "clock-conditioned compact motor comparison", "validation": validation,
                           "fresh": fresh, "comparison_status": comparison["status"], "receipt": progress,
                           "blocker": comparison.get("evaluation_blocker")}}


def interrupted_smolvla():
    path = EVIDENCE / "motor/smolvla-initial-cpu-attempt.json"
    recorded = load(path)
    directory = Path(recorded["run_directory"])
    receipt = directory / "run.json"
    log = directory / "training.log"
    text = log.read_text()
    if (load(receipt) != {key: value for key, value in recorded.items() if key != "observed"}
            or recorded["returncode"] != -2 or "accelerator.backward(loss)" not in text or "KeyboardInterrupt" not in text
            or any((directory / "training/checkpoints").glob("[0-9]*"))):
        raise ValueError("The interrupted SmolVLA attempt no longer matches its saved first-step evidence")
    return {"job": directory.name, "model": "smolvla", "purpose": "initial bounded MuJoCo training attempt",
            "status": "interrupted_before_first_optimizer_update", "optimizer_updates": 0, "checkpoint": None, "evaluation": None,
            "device": recorded["device"], "duration_s": recorded["elapsed_s"], "dataset": motor_dataset(recorded),
            "training_receipt": reference(receipt), "original_attempt_receipt": reference(path), "training_log": reference(log)}


def evidence_dataset(path, digest):
    portfolio = load(path)
    actual = reference(path)
    if actual["sha256"] != digest:
        raise ValueError(f"The evidence portfolio changed at {path}")
    groups = {split: {row["scene_group_id"] for row in portfolio["records"] if row["split"] == split}
              for split in ("train", "validation", "test")}
    count = {split: len(names) for split, names in groups.items()}
    if count != portfolio["groups_by_split"] or len(set.union(*groups.values())) != sum(count.values()):
        raise ValueError("Independent source groups changed or overlap after fitting")
    return {**actual, "independent_groups_by_split": count, "recorded_variants": len(portfolio["records"])}


def learned_model(model, frozen):
    kind = model.get("kind", model.get("model"))
    weight = reference(Path(model["checkpoint"]))
    if (model["updates"], model["checkpoint_sha256"], weight["sha256"]) != (
            frozen["updates"], frozen["checkpoint_sha256"], frozen["checkpoint_sha256"]):
        raise ValueError(f"The archived learned {kind} checkpoint changed")
    if "test" in model:
        counts = model["test"]
        selected = ("independent_groups", "resolvable_count", "resolvable_correct", "f6_independent_pairs",
                    "f6_inspections", "f6_variants", "resolvable_container_count", "resolvable_first_correct_container")
        evaluation = {key: counts[key] for key in selected}
        evaluation["scope"] = "bounded independent F1–F6 pilot"
    else:
        cases = model["test_outcomes"]
        evaluation = {"scope": "initial development source", "independent_groups": model["test_group_count"],
                      "recorded_variants": len(cases), "inspections": sum(row["decision"] == "request_inspection" for row in cases),
                      "resolvable_controls_correct": sum(row["decision"] == row["terminal"] for row in cases
                                                         if row["variant"] in ("F1", "F2", "F6-resolvable")),
                      "indistinguishable_pair_predictions_equal": model["f6_trained_counterfactual_equal"]}
    return {"model": kind, "optimizer_updates": model["updates"], "selected_update": model.get("selected_update", model["updates"]),
            "checkpoint": weight, "evaluation": evaluation}


def evidence_job(path):
    report = load(path)
    directory = Path(report["models"][0]["checkpoint"]).parent
    frozen_path = Path(report.get("frozen_fit_report", directory / "run.json"))
    frozen = load(frozen_path)
    if report["status"] not in ("trained", "complete"):
        raise ValueError(f"Incomplete learned evidence job at {directory}")
    models = [learned_model(a, b) for a, b in zip(report["models"], frozen["models"], strict=True)]
    ds = Path(report.get("dataset", report.get("input_portfolio")))
    digest = report.get("input_portfolio_sha256", report.get("input_sha256"))
    return {"job": directory.name, "purpose": "causal temporal visual evidence supervision comparison",
            "status": "trained_and_evaluated", "device": "cpu", "models": models, "dataset": evidence_dataset(ds, digest),
            "training_receipt": reference(frozen_path), "evaluation_receipt": reference(path),
            "shared_reservation_recorded": "compute_reservation" in report or "reservation" in report}


def inventory(config):
    root = config["data_root"] / "training-runs"
    motor, resume, pilot = (EVIDENCE / name for name in ("motor", "resume", "learning-pilot-v3"))
    jobs = [motor_job(motor / "act-training.json", motor / "act-heldout-evaluation.json", "original ACT MuJoCo milestone"),
            interrupted_smolvla(),
            motor_job(motor / "smolvla-training.json", motor / "smolvla-heldout-evaluation.json", "original SmolVLA MuJoCo milestone"),
            evidence_job(resume / "temporal-evidence-job.json")]
    aligned = []
    for name, purpose in (("act-fit.json", "fit five aligned teacher trajectories"),
                          ("act-clock-fit.json", "warm-start test of causal issue-clock conditioning")):
        result = load(pilot / name)
        aligned.append(motor_job(Path(result["job_directory"]) / "run.json", pilot / name, purpose))
    for split in ("train", "validation"):
        first, second = [job["dataset"]["splits"][split]["source_world_sha256"] for job in aligned]
        if first != second:
            raise ValueError(f"The clocked ACT fit changed the original {split} trajectory ancestry")
    jobs.extend(aligned)
    jobs.append(evidence_job(pilot / "story-learning.json"))
    compact = pilot / "smolvla-full.json"
    if compact.is_file() and load(compact).get("job_directory"):
        jobs.append(compact_smolvla(compact))
    known = {row["job"] for row in jobs}
    directories = {path.name for path in root.iterdir() if path.is_dir()}
    if not known <= directories or len(known) != len(jobs):
        raise ValueError("The archived job inventory differs from the declared Homebrain workspace")
    return {"schema": "smart-swarm-training-inventory-v2", "scope": "archived and completed optimizer launches in Smart Swarm",
            "training_root": str(root), "known_launches": len(jobs), "new_optimizer_updates": 0, "jobs": jobs,
            "unclassified_run_directories": sorted(directories - known)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise ValueError("Choose a new immutable training inventory output")
    atomic(args.out, inventory(settings()))
    print(args.out)


if __name__ == "__main__":
    main()
