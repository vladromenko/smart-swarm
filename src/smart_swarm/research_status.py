"""Read current research receipts without changing native or simulator state."""

from collections import Counter
from pathlib import Path

from .episode import load, sha

ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / "evidence"
PILOT = EVIDENCE / "learning-pilot-v3"
WORK = ROOT.parent.parent / "work"
TEST_ROOT = WORK / "primary-sim-test"
COMPACT_OLD = WORK / "motor-compact-rehearsal-v1"
COMPACT = WORK / "motor-compact-rehearsal-v2"
HEADS = ("events_only", "events_plus_reveal")
STAGES = {"scripts/capture_primary_test.py": "capture", "scripts/score_primary_test.py": "scoring",
          "scripts/replay_primary_compact_controls.py": "matched_compact_controls"}
PYTHON_FLAGS = {b"-u", b"-B", b"-I", b"-s", b"-S", b"-O", b"-OO"}


def saved(path):
    return load(path) if path.is_file() else {}


def process_stamp(path):
    columns = (path / "stat").read_text().rpartition(")")[2].split()
    return columns[0], columns[19]


def current_worker(path, project):
    try:
        if (path / "cwd").resolve(strict=True) != project:
            return None
        before = process_stamp(path)
        if before[0] in ("Z", "X"):
            return None
        argv = (path / "cmdline").read_bytes().split(b"\0")
        if not argv[0].rsplit(b"/", 1)[-1].startswith(b"python"):
            return None
        entry = next((arg for arg in argv[1:] if arg not in PYTHON_FLAGS), b"")
        after = process_stamp(path)
        for script, stage in STAGES.items():
            openings = {script.encode(), str(project / script).encode()}
            if entry in openings and after[0] not in ("Z", "X") and before[1] == after[1]:
                return {"pid": int(path.name), "stage": stage, "start_ticks": int(before[1])}
    except (OSError, ValueError, IndexError):
        return None
    return None


def primary_workers(project, proc=Path("/proc")):
    found = [current_worker(path, project) for path in proc.glob("[0-9]*") if path.name.isdigit()]
    return sorted((entry for entry in found if entry is not None), key=lambda entry: entry["pid"])


def capture_counts(root, expected, release):
    counts = Counter()
    accepted = set()
    for path in root.glob("*/*/capture.json"):
        try:
            row = load(path)
            name, status = row["source_group"], row["status"]
            reservation = row.get("reservation")
            binding = reservation.get("primary_test_release_sha256") if isinstance(reservation, dict) else None
            if not isinstance(name, str) or not isinstance(status, str):
                raise ValueError("An attempt needs a source group and state")
        except (OSError, ValueError, TypeError, KeyError):
            counts["unreadable_receipt"] += 1
        else:
            if name != path.parent.parent.name or name not in expected or not release or binding != release:
                counts["outside_release"] += 1
            else:
                counts[status] += 1
                if status == "accepted":
                    accepted.add(name)
    return dict(counts), len(accepted)


def release_summary():
    allocation_path = PILOT / "primary-sim-source-allocation.json"
    allocation = saved(allocation_path)
    slots = {row["source_group"] for row in allocation.get("slots", []) if row["split"] == "test"}
    release_path = TEST_ROOT / "primary-test-release.json"
    release = saved(release_path)
    expected_profile = PILOT / "primary-profiles-frozen.json"
    valid = (release and expected_profile.is_file() and allocation_path.is_file()
             and release.get("allocation_sha256") == sha(allocation_path)
             and release.get("profile_sha256") == sha(expected_profile))
    digest = sha(release_path) if valid else None
    counts, accepted = capture_counts(TEST_ROOT, slots, digest)
    reports = {kind: (PILOT / f"primary-test-{kind}.json").is_file()
               and (PILOT / f"primary-test-{kind}-predictions.json").is_file() for kind in HEADS}
    return {"planned_independent_sources": len(slots), "accepted_independent_sources": accepted,
            "attempt_receipts": counts, "release_sha256": digest, "live_workers": primary_workers(ROOT),
            "score_pairs_on_disk": reports}


def integration():
    fixture = saved(EVIDENCE / "resume/platform-rehearsal.json")
    native = saved(EVIDENCE / "resume/native-studio-rehearsal.json").get("B_consumption", {})
    return {"fixture_export_to_consumer": fixture.get("A_fixture_export_to_B"),
            "native_technical_export_frames": native.get("frames"), "native_camera_order": native.get("rgb_camera_order"),
            "native_export_technical_only": native.get("technical_only")}


def data_validity():
    teacher = saved(EVIDENCE / "resume/teacher-replays.json")
    cases = teacher.get("cases", [])
    aligned = [row["aligned-sample-v2"] for row in cases]
    previous = saved(COMPACT_OLD / "rehearsal.json")
    rehearsal = saved(COMPACT / "rehearsal.json")
    capture = saved(COMPACT / "full-capture.json")
    split_progress = {name: {key: split[key] for key in ("accepted", "attempted", "unaccepted")}
                      for name, split in capture.get("splits", {}).items()}
    return {"aligned_teacher_attempts": len(aligned),
            "aligned_teacher_successes": sum(row["first_failed_stage"] is None for row in aligned),
            "teacher_gate_passed": teacher.get("teacher_gate"), "compact_rehearsal": rehearsal.get("status", "awaiting_repair_rehearsal"),
            "original_compact_rehearsal": previous.get("status"), "original_compact_rehearsal_reason": previous.get("reason"),
            "compact_capture": capture.get("status", "awaiting_receipt"), "accepted_compact_sources": capture.get("accepted_sources"),
            "compact_split_progress": split_progress, "compact_capture_active_scope": capture.get("active", {}).get("scope"),
            "compact_export": compact_export_progress(capture),
            "compact_capture_reason": capture.get("reason")}


def compact_export_progress(capture):
    if capture.get("status") == "dataset_ready":
        return {"status": "published", "dataset": capture["dataset"], "sources": capture["source_counts"]}
    if capture.get("active", {}).get("stage") != "export":
        return {"status": "awaiting_export"}
    log = COMPACT / "full-export.log"
    if not log.is_file():
        return {"status": "starting"}
    prefix = "Official export attempt "
    line = next((row for row in reversed(log.read_text(errors="replace").splitlines()) if row.startswith(prefix)), "")
    root = Path(line.partition(": ")[2]).resolve()
    if root.parent != COMPACT.resolve() or not root.name.startswith("lerobot-full-unpublished-"):
        return {"status": "starting"}
    progress = {}
    for split in ("train", "validation"):
        try:
            info = saved(root / split / "meta/info.json")
            progress[split] = {"written_sources": info.get("total_episodes", 0), "written_rows": info.get("total_frames", 0)}
        except (OSError, ValueError):
            pass
    return {"status": "writing", "staged_split_progress": progress}


def optimizer_pid(argv, proc=Path("/proc")):
    expected = b"\0".join(arg.encode() for arg in argv) + b"\0"
    for path in proc.glob("[0-9]*"):
        try:
            if (path / "cmdline").read_bytes() == expected and process_stamp(path)[0] not in ("Z", "X"):
                return int(path.name)
        except (OSError, ValueError, IndexError):
            pass
    return None


def motor_evaluation_progress(root, result):
    choices = saved(root / "validation-choices.json") or []
    validation = [{key: row[key] for key in ("checkpoint", "successful_resets", "offline_validation_mean_loss")}
                  | {"resets": len(row["resets"])} for row in choices]
    verified = {row["checkpoint"] for row in validation}
    fresh = {path: saved(path) for path in sorted(root.glob("fresh/*/evaluation.json"))}
    selected = result.get("checkpoint")
    evaluated = {row["checkpoint"] for row in fresh.values() if row.get("evaluation_role") == "fresh-test"}
    if selected is None and len(evaluated) == 1 and evaluated <= verified:
        selected = next(iter(evaluated))
    fresh_results = [attempt for row in fresh.values() for attempt in row["results"]]
    progress = [{key: row[key] for key in ("checkpoint", "status", "requested_resets")}
                | {"receipt": str(path), "completed_resets": len(row["results"])} for path, row in fresh.items()]
    validation_counts = [len(saved(path)["resets"]) for path in sorted(root.glob("validation/*/validation.json"))]
    return {"selected_checkpoint": selected, "completed_validation_candidates": validation,
            "validation_resets_by_candidate": validation_counts, "fresh_evaluations": progress,
            "fresh_resets": len(fresh_results),
            "fresh_successes": sum(attempt["contact_valid_placement"] for attempt in fresh_results)}


def motor_run_progress(result):
    if not result.get("job_directory"):
        return {"optimizer_updates": result.get("optimizer_updates", 0), "optimizer_pid": None}
    from .motor_pilot_run import optimizer_rows

    root = Path(result["job_directory"])
    run = saved(root / "run.json")
    rows = optimizer_rows(root / "training.log")
    checkpoints = root / "training/checkpoints"
    saved_paths = sorted(path / "pretrained_model" for path in checkpoints.iterdir()
                         if path.name.isdecimal() and (path / "pretrained_model/model.safetensors").is_file()) if checkpoints.is_dir() else []
    return {"dataset": result["dataset"], "job_directory": str(root), "cap": result["cap"],
            "optimizer_updates": result.get("optimizer_updates", rows[-1]["update"] if rows else 0),
            "latest_reported_loss": rows[-1]["loss"] if rows else None,
            "optimizer_pid": optimizer_pid(run["argv"]) if run.get("status") == "training" and run.get("argv") else None,
            "complete_checkpoints": [str(path) for path in saved_paths], **motor_evaluation_progress(root, result)}


def motor():
    archived = {}
    for name in ("act", "smolvla"):
        path = EVIDENCE / f"motor/{name}-heldout-evaluation.json"
        row = saved(path)
        if row.get("status") == "evaluated":
            archived[name] = {key: row[key] for key in ("successful_resets", "requested_resets", "checkpoint")}
    current = saved(PILOT / "act-clock-fit.json")
    fitting = current.get("fitting_cases", [])
    comparisons = {name: saved(PILOT / f"{name}-full.json") for name in ("act", "smolvla")}
    development_profile = COMPACT / "smolvla-development-profile.json"
    profile = saved(development_profile)
    native_path = PILOT / "native-policy-timing-and-stop-signal-rehearsal.json"
    native = saved(native_path)
    return {"historical_development_evaluations": archived, "matched_act_attempts": current.get("matching_starts"),
            "matched_act_successes": current.get("successful"), "act_fit_gate": current.get("fit_gate"),
            "act_checkpoint": current.get("checkpoint"),
            "act_precontact_diagnostics": act_precontact_diagnostics(),
            "smolvla_development_profile": {"status": profile.get("status", "awaiting_receipt"),
                                             "charged_wall_s": profile.get("charged_wall_s"),
                                             "receipt": str(development_profile) if profile else None},
            "smolvla_training_exposure": motor_exposure(),
            "device_timing_profiles": motor_device_profiles(),
            "native_execution_qualification": {"hardware_execution_qualified": native.get("hardware_execution_qualified"),
                                               "receipt": str(native_path) if native else None},
            "full_comparison_status": {name: row.get("status", "awaiting_receipt") for name, row in comparisons.items()},
            "full_comparison_progress": {name: motor_run_progress(row) for name, row in comparisons.items()},
            "first_failure_stages": dict(Counter(row["first_failed_stage"] for row in fitting if row["first_failed_stage"]))}


def act_precontact_diagnostics():
    geometry = saved(PILOT / "act-precontact-teacher-physics.json")
    optical = PILOT / "act-precontact-teacher-rgb-v2.json"
    failure = PILOT / "act-precontact-teacher-rgb-v2-failed.json"
    pixel = PILOT / "act-precontact-stock-pixel-mismatch.json"
    raster = PILOT / "act-stock-static-raster.json"
    measured = saved(raster)
    timing = PILOT / "act-archived-clock-offset.json"
    verified = saved(timing)
    return {"contact_valid_teacher_starts": len(geometry.get("qualified_contact_free_starts", [])),
            "optical_status": "evaluated" if optical.is_file() else ("stock_rgb_parity_failed" if failure.is_file() else "pending"),
            "optical_replay_receipt": str(optical) if optical.is_file() else None,
            "optical_failure_receipt": str(failure) if failure.is_file() else None,
            "stock_mismatch_receipt": str(pixel) if pixel.is_file() else None,
            "static_raster_receipt": str(raster) if measured else None,
            "static_raster_status": measured.get("status", "awaiting_receipt"),
            "remaining_original_raster_budget_s": measured.get("remaining_original_seconds"),
            "clock_replay_receipt": str(timing) if verified else None,
            "clock_replay_development_sources": len(verified.get("sources", []))}


def motor_device_profiles():
    profiles = {}
    for path in sorted(PILOT.glob("*-worker-timing.json"), key=lambda item: (item.stat().st_mtime_ns, item.name)):
        report = saved(path)
        if report.get("requested_device") not in ("cpu", "cuda") or report.get("model") not in ("act", "smolvla"):
            raise ValueError(f"Unrecognized motor timing receipt {path}")
        model, device = report["model"], report["requested_device"]
        row = {"status": report["status"], "receipt": str(path), "worker_started": report["worker_started"]}
        if report["status"] == "profiled":
            requests = report["action_requests"]
            row.update(verified_training_rows=report["verified_training_rows"],
                       fresh_chunk_age_ms=[item["available_observation_age_ns"] / 1e6
                                           for item in requests if item["fresh_chunk"]],
                       all_requests_within_coverage=all(item["within_coverage"] for item in requests))
        elif "reason" in report:
            row["reason"] = report["reason"]
        profiles[f"{model}_{device}"] = row
    stage_path = PILOT / "smolvla-cpu-chunk-stage-timing.json"
    if stage_path.is_file() and "smolvla_cpu" in profiles:
        stage = load(stage_path)
        profiles["smolvla_cpu"]["stage_measurement"] = {
            "receipt": str(stage_path), "status": stage["status"],
            "prefix_seconds": [row["prefix_before_first_denoising_step_ns"] / 1e9 for row in stage.get("requests", [])]}
    return profiles


def motor_exposure():
    path = PILOT / "smolvla-compact-training-exposure.json"
    report = saved(path)
    if not report:
        return {"status": "awaiting_receipt"}
    summary = report["summary"]
    initial_path = PILOT / "smolvla-initial-approach-comparison.json"
    initial = saved(initial_path)
    geometry_path = PILOT / "smolvla-validation-approach-geometry.json"
    geometry = saved(geometry_path)
    keys = ("committed_updates", "source_groups_exposed", "source_groups_available",
            "startup_horizon_updates", "exact_reset_start_updates", "cpu_update_plus_data_median_s")
    return {"status": report["status"], "receipt": str(path), "sha256": sha(path),
            "initializer_comparison": {"status": initial.get("status", "awaiting_receipt"),
                                       "matched_validation_starts": len(initial.get("cases", [])),
                                       "receipt": str(initial_path) if initial else None},
            "validation_approach_geometry": {"receipt": str(geometry_path) if geometry else None,
                                             **geometry.get("summary", {})},
            **{name: summary[name] for name in keys}}


def high_resolution_inspection():
    paths = (PILOT / f"rosmaster-high-resolution-matched-development-v{version}.json" for version in (3, 2, 1))
    path = next((candidate for candidate in paths if candidate.is_file()), None)
    if path is None:
        return {"status": "awaiting_receipt"}
    result = load(path)
    comparison_path = WORK / "active-inspection-comparison-mobile-v3/comparison.json"
    comparison = saved(comparison_path)
    archive = PILOT / "rosmaster-bounded-comparison-homebrain.tar.gz"
    decisions = PILOT / "rosmaster-inspection-decisions.json"
    return {"status": "qualified_development" if result["ready"] else "not_qualified",
            "controls_reopened": len(result["results"]), "unqualified_cases": result["refused"],
            "remaining_wall_seconds": comparison.get("remaining_wall_seconds", result["remaining_wall_seconds"]),
            "independent_cases_opened": comparison.get("attempted_groups", result["independent_cases_opened"]),
            "independent_valid_pairs": comparison.get("valid_pairs", 0), "receipt": str(path),
            "comparison": str(comparison_path) if comparison else None,
            "comparison_status": comparison.get("status"),
            "results_archive": str(archive) if archive.is_file() else None,
            "inspection_decision_audit": str(decisions) if decisions.is_file() else None}


def mobile_inspection():
    previous = WORK / "active-inspection-comparison-mobile-v1"
    root = WORK / "active-inspection-comparison-mobile-v2"
    optics = PILOT / "rosmaster-mobile-recovered-optics-triangulation-v2.json"
    prescan = PILOT / "rosmaster-mobile-development-preinspection-scope.json"
    parity = WORK / "rosmaster-observed-step-parity-v2/verification.json"
    development = saved(root / "development-gate.json")
    previous_development = saved(previous / "development-gate.json")
    comparison = saved(root / "comparison.json")
    rows = [saved(path) for path in root.glob("mobile-rehearsal-*/receipt.json")]
    high_resolution = high_resolution_inspection()
    return {"recorded_rgb_development_gate": saved(optics).get("ready"),
            "preinspection_pair_invariant": saved(prescan).get("same_frozen_f6_predictions"),
            "observed_step_route_parity": saved(parity).get("status", "awaiting_receipt"),
            "integrated_development_status": development.get("status", "awaiting_receipt"),
            "original_integrated_status": previous_development.get("status"),
            "integrated_attempt_receipts": dict(Counter(row["status"] for row in rows if row)),
            "integrated_development_reason": development.get("reason"),
            "paired_high_resolution": high_resolution,
            "independent_attempts": comparison.get("attempted_groups", 0) + high_resolution.get("independent_cases_opened", 0)}


def witness_transfer():
    root = WORK / "witness-transfer-v1"
    design = saved(root / "design.json")
    if not design:
        return {"status": "awaiting_allocation"}
    sources = design["sources"]
    reframed = WORK / "witness-transfer-v2-heldout"
    fresh = saved(reframed / "design.json")
    release = PILOT / "witness-transfer-policy-release.json"
    evaluation = PILOT / "witness-transfer-reframed-evaluation.json"
    return {"allocation": str(root / "design.json"), "planned_source_groups": len(sources),
            "capture_receipts_by_split": transfer_capture_counts(root, sources),
            "progress_receipt": str(root / "progress.json") if (root / "progress.json").is_file() else None,
            "frozen_policy": str(release) if release.is_file() else None,
            "fresh_evaluation_allocation": str(reframed / "design.json") if fresh else None,
            "fresh_evaluation_capture_counts": transfer_capture_counts(reframed, fresh["sources"]) if fresh else None,
            "fresh_evaluation_receipt": str(evaluation) if evaluation.is_file() else None}


def transfer_capture_counts(root, sources):
    states = Counter()
    for row in sources:
        receipt = saved(root / row["source_group"] / "receipt.json")
        phase = receipt.get("status", "unopened")
        states[row["split"], "unsealed" if phase == "running" else phase] += 1
    return {split: {state: n for (name, state), n in states.items() if name == split} for split in ("validation", "test")}


def matched_compact_controls(workers):
    root = WORK / "primary-compact-controls-v1"
    output = PILOT / "primary-matched-compact-controls.json"
    seal = root / "predictions.json"
    active = [row for row in workers if row["stage"] == "matched_compact_controls"]
    groups = len(list((root / "groups").glob("*.json")))
    if output.is_file() and seal.is_file():
        status = "evaluated"
    elif seal.is_file():
        status = "forecasts_sealed"
    elif active:
        status = "forecasting"
    else:
        status = "incomplete" if (root / "release.json").is_file() else "awaiting_release"
    return {"status": status, "source_and_team_groups_on_disk": groups,
            "live_workers": active, "predictions": str(seal) if seal.is_file() else None,
            "evaluation": str(output) if status == "evaluated" else None}


def memory():
    path = PILOT / "story-learning.json"
    study = saved(path)
    primary = release_summary()
    compact = matched_compact_controls(primary["live_workers"])
    delay = PILOT / "primary-delivery-delay.json"
    delivery = str(delay) if delay.is_file() else None
    if study.get("status") != "complete":
        return {"pilot": "pending", "primary": primary, "mobile_inspection": mobile_inspection(),
                "witness_transfer": witness_transfer(), "matched_compact_controls": compact, "exploratory_delivery_delay": delivery}
    fields = ("independent_groups", "resolvable_count", "resolvable_correct",
              "f6_independent_pairs", "f6_variants", "f6_inspections")
    models = {row["kind"]: {"optimizer_updates": row["updates"], "checkpoint": row["checkpoint"],
              **{name: row["test"][name] for name in fields}} for row in study["models"]}
    return {"pilot": "evaluated", "pilot_receipt": str(path), "models": models, "primary": primary,
            "mobile_inspection": mobile_inspection(), "witness_transfer": witness_transfer(),
            "matched_compact_controls": compact, "exploratory_delivery_delay": delivery}


def recorded_physical(data):
    solo = saved(data / "so101-pilot-status.json")
    plan = saved(data / "collection-plans/physical-stories-v1.json")
    journal = [saved(data / "physical-stories" / row["story_id"] / "trial.json") for row in plan.get("cards", [])]
    counts = Counter(row["status"] for row in journal if row)
    return {"so101_pilot_phase": solo.get("phase"), "so101_pilot_counts": solo.get("counts"),
            "distributed_cards_planned": plan.get("planned_stories"), "distributed_attempts_by_status": dict(counts),
            "distributed_attempts": sum(counts.values()), "native_readiness_command": "./smart-swarm story-recording readiness"}


def project_summary(config, previous):
    learned = memory()
    milestones = dict(previous.get("milestones", {}))
    if learned["pilot"] == "evaluated":
        milestones["B2"] = "motor_diagnosed_learned_evidence_pilot_evaluated"
    return {**previous, "milestones": milestones,
            "training": "bounded_evidence_pilot_evaluated" if learned["pilot"] == "evaluated" else previous.get("training", "pending"),
            "software_integration": integration(), "data_validity": data_validity(),
            "motor_competence": motor(), "learned_evidence": learned,
            "physical_verification": recorded_physical(config["data_root"])}
