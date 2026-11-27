"""Regression gates for the resumed software and experiment provenance."""

from contextlib import contextmanager
from pathlib import Path
from subprocess import TimeoutExpired

import pytest

from smart_swarm.cli import settings
from smart_swarm.evidence_inputs import audit_b1
from smart_swarm.episode import atomic, load, sha
from smart_swarm.so101_pilot import classify
from smart_swarm.studio import assign_attempts, status_for

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "evidence/resume"


def test_resume_descriptor_has_real_command_and_explicit_external_roots():
    config = settings()
    descriptor = load(ROOT / "handover/experiment.json")
    assert descriptor == load(ROOT / "experiment.json")
    assert descriptor["display_name"] == "Smart Swarm / SO101 pilot"
    assert Path(descriptor["entrypoint"][0]).is_file()
    assert Path(descriptor["project_path"]) == ROOT
    assert (config["platform_handover"] / "HANDOVER_TO_SMART_SWARM.txt").is_file()
    assert (config["studio_project"] / "scripts/export_lerobot.py").is_file()


@pytest.mark.parametrize(("native", "device", "archive_only"), [(True, "cpu", True), (True, "cuda", False), (False, "cpu", False)])
def test_training_only_requests_cpu_lease_for_saved_data(monkeypatch, tmp_path, native, device, archive_only):
    from smart_swarm import research_jobs, training
    from smart_swarm.episode import atomic

    for split in ("train", "validation"):
        ledger = tmp_path / split / "smart-swarm-lineage.json"
        ledger.parent.mkdir()
        atomic(ledger, {"episodes": [{"split": split}]})

    @contextmanager
    def admitted(config, identifier, **options):
        assert options == {"device": device, "archive_only": archive_only}
        assert config["native_cpu_admission"] is native and identifier.startswith("act-pilot-")
        yield {"scope": "fixture"}, lambda: None

    monkeypatch.setattr(research_jobs, "admitted", admitted)
    monkeypatch.setattr(training, "run_motor", lambda *args, **kwargs: args[6])
    result = training.train("act-pilot", tmp_path, tmp_path / "runs", {"native_cpu_admission": native}, device=device)
    assert result == {"scope": "fixture"}


def test_attempt_ids_survive_quarantine_and_late_polling(tmp_path):
    entries = [{"id": "ep_second"}, {"id": "ep_first"}]
    original = assign_attempts(tmp_path, entries)
    extended = assign_attempts(tmp_path, [{"id": "ep_inline"}, *reversed(entries)])
    assert original == {"ep_first": 0, "ep_second": 1}
    assert extended == {**original, "ep_inline": 2}


def write_finalized_test_episode(source, episode):
    raw_episode = source / "episodes" / episode
    raw_episode.mkdir(parents=True)
    (raw_episode / "episode.json").write_text('{"status":"complete","frames":1}')
    (raw_episode / "frames.jsonl").write_text("{}\n")


def test_native_consumer_timeout_resumes_completed_export(monkeypatch, tmp_path):
    from smart_swarm import studio

    source = tmp_path / "native"
    write_finalized_test_episode(source, "native1")
    (source / "dataset.json").write_text("{}")
    export = tmp_path / "export-v1"
    export.mkdir()
    calls = []
    _, episodes = studio.native_export_snapshot(source)
    validation = {"episodes": episodes}

    def generate(*_):
        calls.append("exported")
        return export, {"status": "complete"}

    def consume(config, raw, target):
        assert raw == source and target == export
        calls.append("consumed")
        if calls.count("consumed") == 1:
            raise TimeoutExpired("consume", 160)
        return validation

    monkeypatch.setattr(studio, "export_native", generate)
    monkeypatch.setattr(studio, "consume_native", consume)
    with pytest.raises(TimeoutExpired):
        studio.accepted_export({}, source, tmp_path)
    assert (tmp_path / "completed-native-export.json").is_file()
    assert not (tmp_path / "accepted-native-export.json").exists()
    assert studio.accepted_export({}, source, tmp_path) == (export, validation)
    assert calls == ["exported", "consumed", "consumed"]


def test_native_export_refreshes_when_an_episode_appears_before_catalogue_update(monkeypatch, tmp_path):
    from smart_swarm import studio

    source = tmp_path / "native"
    write_finalized_test_episode(source, "native1")
    (source / "dataset.json").write_text("{}")
    exported = {}

    def generate(_, raw):
        path = tmp_path / f"export-{len(exported) + 1}"
        path.mkdir()
        exported[path] = {"episodes": studio.native_export_snapshot(raw)[1]}
        return path, {"status": "complete"}

    monkeypatch.setattr(studio, "export_native", generate)
    monkeypatch.setattr(studio, "consume_native", lambda config, raw, target: exported[target])
    first, _ = studio.accepted_export({}, source, tmp_path)
    before = load(tmp_path / "accepted-native-export.json")
    write_finalized_test_episode(source, "native2")
    second, validation = studio.accepted_export({}, source, tmp_path)
    after = load(tmp_path / "accepted-native-export.json")
    assert first != second
    assert [row["episode_id"] for row in validation["episodes"]] == ["native1", "native2"]
    assert before["source_manifest_sha256"] == after["source_manifest_sha256"]
    assert before["source_episode_set_sha256"] != after["source_episode_set_sha256"]
    assert studio.accepted_export({}, source, tmp_path) == (second, validation)
    assert len(exported) == 2


def test_rejected_native_cache_resumes_the_completed_replacement_after_timeout(monkeypatch, tmp_path):
    from smart_swarm import studio
    from smart_swarm.episode import atomic

    source = tmp_path / "raw"
    write_finalized_test_episode(source, "native1")
    (source / "dataset.json").write_text("{}")
    generation, episodes = studio.native_export_snapshot(source)
    previous, replacement = tmp_path / "previous", tmp_path / "replacement"
    previous.mkdir()
    replacement.mkdir()
    accepted = tmp_path / "accepted-native-export.json"
    atomic(accepted, {**generation, "export": str(previous)})
    calls = []

    def consume(config, raw, target):
        calls.append(target)
        if target == previous:
            raise studio.NativeExportMismatch("An interior native action changed")
        if calls.count(replacement) == 1:
            raise TimeoutExpired("consume", 160)
        return {"episodes": episodes}

    monkeypatch.setattr(studio, "consume_native", consume)
    monkeypatch.setattr(studio, "export_native", lambda *_: (replacement, {}))
    with pytest.raises(TimeoutExpired):
        studio.accepted_export({}, source, tmp_path)
    evidence = studio.rejection_path(tmp_path, generation, previous)
    assert load(evidence)["reason"] == "An interior native action changed" and previous.is_dir()
    assert load(accepted)["export"] == str(previous)
    monkeypatch.setattr(studio, "export_native", lambda *_: pytest.fail("The completed replacement already exists"))
    assert studio.accepted_export({}, source, tmp_path) == (replacement, {"episodes": episodes})
    assert load(accepted)["export"] == str(replacement) and calls == [previous, replacement, replacement]


def test_native_cache_does_not_replace_a_changing_recording(monkeypatch, tmp_path):
    from smart_swarm import studio
    from smart_swarm.contract import ProtocolError
    from smart_swarm.episode import atomic

    source = tmp_path / "raw"
    write_finalized_test_episode(source, "native1")
    (source / "dataset.json").write_text("{}")
    generation, _ = studio.native_export_snapshot(source)
    previous = tmp_path / "previous"
    previous.mkdir()
    atomic(tmp_path / "accepted-native-export.json", {**generation, "export": str(previous)})

    def consume(*_):
        write_finalized_test_episode(source, "native2")
        raise studio.NativeExportMismatch("Saved native episodes changed")

    monkeypatch.setattr(studio, "consume_native", consume)
    monkeypatch.setattr(studio, "export_native", lambda *_: pytest.fail("The raw generation is still changing"))
    with pytest.raises(ProtocolError, match="Native recordings changed") as error:
        studio.accepted_export({}, source, tmp_path)
    assert error.value.retryable and not studio.rejection_path(tmp_path, generation, previous).exists()


def test_native_consumer_runtime_failure_preserves_a_completed_export(monkeypatch, tmp_path):
    from subprocess import CompletedProcess

    from smart_swarm import studio
    from smart_swarm.contract import ProtocolError

    config = {"studio_project": tmp_path, "studio_python": "studio"}
    monkeypatch.setattr(studio.subprocess, "run", lambda *args, **opts: CompletedProcess([], 1, stderr="ImportError"))
    with pytest.raises(ProtocolError, match="did not run successfully") as error:
        studio.consume_native(config, tmp_path, tmp_path)
    assert error.value.retryable
    monkeypatch.setattr(studio.subprocess, "run", lambda *args, **opts: CompletedProcess([], 0, stdout="{}"))
    with pytest.raises(ProtocolError, match="incomplete receipt") as missing:
        studio.consume_native(config, tmp_path, tmp_path)
    assert missing.value.retryable
    monkeypatch.setattr(studio.subprocess, "run", lambda *args, **opts: CompletedProcess([], 3, stderr="Invalid action"))
    with pytest.raises(studio.NativeExportMismatch, match="Invalid action"):
        studio.consume_native(config, tmp_path, tmp_path)


def test_native_export_deadline_scales_with_saved_visual_work(monkeypatch, tmp_path):
    from smart_swarm import studio
    from smart_swarm.episode import atomic

    ledger = [{"frames": 281}]
    monkeypatch.setattr(studio, "native_export_snapshot", lambda _: ({}, ledger))
    catalogue = tmp_path / "dataset.json"
    atomic(catalogue, {"settings": {"camera_mode": "wrist_only"}})
    assert studio.native_export_timeout(tmp_path) == 180
    ledger[0]["frames"] = 1768
    assert studio.native_export_timeout(tmp_path) == 281
    atomic(catalogue, {"camera_mode": "two_camera"})
    assert studio.native_export_timeout(tmp_path) == 502
    ledger[0]["frames"] = 30_000
    assert studio.native_export_timeout(tmp_path) == 1800


def test_native_export_recovers_only_a_verified_publication_at_timeout(monkeypatch, tmp_path):
    from smart_swarm import studio
    from smart_swarm.episode import atomic, sha

    raw = tmp_path / "raw"
    write_finalized_test_episode(raw, "ep_1")
    catalogue = raw / "dataset.json"
    atomic(catalogue, {"camera_mode": "two_camera"})
    config = {"studio_project": tmp_path, "studio_python": "studio-python"}
    valid = [True]

    def finish_at_timeout(command, **options):
        output = Path(command[command.index("--output") + 1])
        sidecar = output / "provenance/export.json"
        sidecar.parent.mkdir(parents=True)
        atomic(sidecar, {"status": "complete", "source_path": str(raw), "source_hashes": {"dataset.json": sha(catalogue)},
                         "official_verification": {"rgb_pixels_bit_exact": valid[0]}})
        raise TimeoutExpired(command, options["timeout"])

    monkeypatch.setattr(studio.subprocess, "run", finish_at_timeout)
    output, service = studio.export_native(config, raw)
    assert service["recovered_after_native_timeout"] and service["path"] == str(output)
    valid[0] = False
    with pytest.raises(TimeoutExpired):
        studio.export_native(config, raw)


def test_motor_visual_profiles_keep_old_rgb_and_reject_mixing(tmp_path):
    pytest.importorskip("mujoco")
    from smart_swarm.episode import atomic
    from smart_swarm.evaluate_motor import bound_visual_profile
    from smart_swarm.motor import bind_capture_profile
    from smart_swarm.motor_visual_profile import COMPACT, STOCK, visual_binding

    checkpoint, validation, raw = (tmp_path / name for name in ("checkpoint", "validation", "raw"))
    checkpoint.mkdir()
    validation.mkdir()
    raw.mkdir()
    assert bound_visual_profile(checkpoint, validation) == {"visual_profile": STOCK}
    compact = visual_binding(COMPACT)
    assert bind_capture_profile(raw, COMPACT) == compact
    with pytest.raises(ValueError, match="one visual profile"):
        bind_capture_profile(raw, STOCK)
    atomic(checkpoint / "smart-swarm-input.json", compact)
    with pytest.raises(ValueError, match="one visual profile"):
        bound_visual_profile(checkpoint, validation)
    atomic(validation / "smart-swarm-lineage.json", compact)
    assert bound_visual_profile(checkpoint, validation) == compact
    changed = {**compact, "visual_implementation_sha256": {"motor_renderer.py": "unexpected"}}
    atomic(validation / "smart-swarm-lineage.json", changed)
    with pytest.raises(ValueError, match="implementation differs"):
        bound_visual_profile(checkpoint, validation)


def test_full_motor_budget_retains_stock_pilot_and_interrupted_charges(monkeypatch, tmp_path):
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    import complete_compact_motor_dataset as full
    from smart_swarm.episode import atomic

    recipe = {"generation_cap_s": 3600, "stock_elapsed_s": 209.0, "round_trip_elapsed_s": 103.0,
              "original_export_cap_s": 1800, "round_trip_export_ceiling_s": 180}
    state = tmp_path / "ledger.json"
    preserved = {"recipe": recipe, "spent_s": {"generation": 28., "export": 9.}, "splits": {}, "interruptions": [],
                 "active": {"stage": "generation", "available_s": 3260., "wall_started_unix": 100.}}
    atomic(state, preserved)
    monkeypatch.setattr(full.time, "time", lambda: 145.)
    reopened = full.open_record(state, recipe)
    assert full.remaining(reopened, "generation") == 3215.
    assert full.remaining(reopened, "export") == 1611.
    assert reopened["interruptions"][0]["charged_s"] == 45.
    reopened = full.open_record(state, recipe)
    assert full.remaining(reopened, "generation") == 3215.
    with pytest.raises(ValueError, match="has not published a receipt"):
        full.pilot_receipts(tmp_path)


def test_full_motor_inventory_requires_independent_source_ids(monkeypatch, tmp_path):
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    import complete_compact_motor_dataset as full
    from smart_swarm.episode import atomic
    from smart_swarm.motor_input import FLOAT32_COMMAND_PROFILE
    from smart_swarm.motor_visual_profile import COMPACT, visual_binding

    for seed in (10000, 10001):
        path = tmp_path / "train" / f"so101-{seed}"
        path.mkdir(parents=True)
        atomic(path / "manifest.json", {**visual_binding(COMPACT), "world_episode_id": path.name,
                                        "command_execution_profile": FLOAT32_COMMAND_PROFILE,
                                        "scene_group_id": "same-source", "sha256": str(seed), "seed": seed,
                                        "status": "accepted", "lineage": {"split": "train", "source_world_ids": []}})
    with pytest.raises(ValueError, match="independent sources"):
        full.actual_inventory(tmp_path)


def test_verified_motor_export_retains_lineage_after_raw_archive_relocation(monkeypatch, tmp_path):
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    from learning_pilot import verified_export
    from smart_swarm.episode import atomic
    from smart_swarm.motor_input import CLOCKED_PROFILE

    source, backup, export = (tmp_path / name for name in ("raw", "archived-raw", "export"))
    for split in ("train", "validation"):
        raw = source / split / f"episode-{split}"
        raw.mkdir(parents=True)
        atomic(raw / "manifest.json", {"sha256": split})
        (export / split).mkdir(parents=True)
        atomic(export / split / "smart-swarm-lineage.json", {"input_profile": CLOCKED_PROFILE,
               "episode_padding_mask_checked": True, "episodes": [{"source_world_episode_id": raw.name,
               "source_sha256": split, "raw_path": str(raw)}]})
    source.rename(backup)
    source.symlink_to(backup, target_is_directory=True)
    assert verified_export(source, export, "--clocked")
    lineage = export / "train/smart-swarm-lineage.json"
    record = load(lineage)
    record["episodes"][0]["raw_path"] = str(backup / "validation/episode-validation")
    atomic(lineage, record)
    with pytest.raises(ValueError, match="export lineage differs"):
        verified_export(source, export, "--clocked")


def test_motor_resume_checks_admission_before_each_new_attempt(monkeypatch, tmp_path):
    pytest.importorskip("mujoco")
    from smart_swarm import motor
    from smart_swarm.episode import atomic

    previous = tmp_path / "train/so101-10000"
    previous.mkdir(parents=True)
    atomic(previous / "manifest.json", {"status": "accepted"})
    events = []

    def captured(assets, path, seed, split, remaining_s, **options):
        events.append(seed)
        return True

    monkeypatch.setattr(motor, "attempt", captured)
    result = motor.generate(tmp_path, tmp_path, "train", 3, 10, 128, lambda: events.append("lease"))
    assert events == ["lease", 10001, "lease", 10002]
    assert result["status"] == "complete" and result["attempted"] == result["accepted"] == 3


def pilot_outcome_fixture(monkeypatch, root):
    from smart_swarm import studio
    from smart_swarm.episode import atomic, sha

    dataset = "ds_1234567890123456"
    (root / "so101-pilot").mkdir()
    entries = [{"id": "ep_1", "storage": "quarantine", "metadata": {"status": "incomplete"}}]
    for index in range(2, 5):
        episode = f"ep_{index}"
        folder = root / "data" / dataset / "episodes" / episode
        folder.mkdir(parents=True)
        ledger = folder / "frames.jsonl"
        ledger.write_text(f'{{"frame": {index}}}\n')
        metadata = {"status": "complete", "frames": 1, "frames_sha256": sha(ledger),
                    "finished_at": f"2026-11-26T14:15:16.00{index}+00:00"}
        atomic(folder / "episode.json", metadata)
        entries.append({"id": episode, "storage": "episodes", "metadata": metadata})
    recorder = {"active": False, "state": "idle"}
    monkeypatch.setattr(studio, "native_request", lambda cfg, uri: {"recording": recorder} if uri == "/api/status"
                        else list(reversed(entries)))
    monkeypatch.setattr(studio, "prepare", lambda cfg: {"attempt_limit": 4})
    return {"data_root": root, "studio_project": root}, {"dataset_id": dataset}, entries, recorder


def test_pilot_outcome_prompt_keeps_missed_attempts_unknown(monkeypatch, tmp_path):
    from smart_swarm import studio
    from smart_swarm.episode import atomic

    config, binding, entries, _ = pilot_outcome_fixture(monkeypatch, tmp_path)
    labels = tmp_path / "so101-pilot/observed-outcomes.json"
    atomic(labels, {"ep_2": {"outcome": "placement_failed"}})
    pending = studio.pending_outcome(config, binding)
    assert pending["attempt"] == 4 and pending["recording"]["episode_id"] == "ep_4"
    assert pending["recording"]["frames_sha256"] == entries[-1]["metadata"]["frames_sha256"]
    atomic(labels, {name: {"outcome": "unknown"} for name in ("ep_2", "ep_4")})
    assert studio.pending_outcome(config, binding) is None
    entries[-1]["storage"] = "quarantine"
    assert studio.pending_outcome(config, binding) is None


def test_pilot_refuses_changed_metadata_and_live_next_attempt(monkeypatch, tmp_path):
    from smart_swarm import studio
    from smart_swarm.contract import ProtocolError
    from smart_swarm.episode import atomic, load

    config, binding, entries, recorder = pilot_outcome_fixture(monkeypatch, tmp_path)
    recorder.update(active=True, state="ready", dataset_id=binding["dataset_id"], last_episode={"episode_id": "ep_4"})
    initial = studio.pending_outcome(config, binding)
    manifest = tmp_path / "data" / binding["dataset_id"] / "episodes/ep_4/episode.json"
    entries[-1]["metadata"]["operator_labels"] = {"new": True}
    atomic(manifest, entries[-1]["metadata"])
    changed = studio.pending_outcome(config, binding)
    assert initial["recording"]["frames_sha256"] == changed["recording"]["frames_sha256"]
    with pytest.raises(ValueError, match="changed"):
        studio.label_attempt(config, binding, initial, "placement_succeeded", assistance="attended_native_motion")
    recorder.update(state="recording", episode_id="ep_5")
    with pytest.raises(ProtocolError, match="finish saving"):
        studio.label_attempt(config, binding, changed, "placement_failed", assistance="unknown")
    labels = tmp_path / "so101-pilot/observed-outcomes.json"
    assert not labels.exists()
    recorder.update(active=False, state="stopped", episode_id="ep_4")
    result = studio.label_attempt(config, binding, changed, "placement_failed", assistance="unknown")
    assert result["recording"] == changed["recording"] and load(labels) == {"ep_4": result}


def test_native_intake_notices_a_metadata_only_change(monkeypatch, tmp_path):
    from smart_swarm import operator

    config, binding, entries, _ = pilot_outcome_fixture(monkeypatch, tmp_path)
    monkeypatch.setattr(operator, "native_request", lambda *_: entries)
    original = operator.intake_fingerprint(config, binding)
    entries[-1]["metadata"]["operator_labels"] = {"present": True}
    updated = operator.intake_fingerprint(config, binding)
    receipt = {"fingerprint": original, "state": "checked"}
    assert operator.intake_is_due(receipt, updated, 1.)


def test_complete_native_failures_and_quarantine_never_enter_motor_review():
    complete = {"id": "native1", "storage": "episodes", "metadata": {"status": "complete", "frames": 3}}
    incomplete = {**complete, "storage": "quarantine", "metadata": {"status": "incomplete", "reason": "stop"}}
    card = {"attempt_limit": 1, "attempts": [{"source_world_episode_id": "w"}], "scene_group_id": "g"}
    label = {"outcome": "placement_failed"}
    validation = {"source": "hardware", "technical_only": False,
                  "episodes": [{"episode_id": "native1", "frames": 3, "frames_sha256": "ledger"}]}
    assert status_for(complete, label, card, 0, validation)["classification"] == "story_failure"
    interrupted = status_for(incomplete, label, card, 0, None)
    assert interrupted["reasons"] == ["stop"] and interrupted["outcome"] == "unknown"
    assert interrupted["observed_outcome_receipt"] == label
    assert status_for(complete, None, card, 1, None)["classification"] == "rejected"
    manifest = {"run_mode": "fixture", "recording_integrity": "complete",
                "validation_status": "valid", "task_outcome_label": "placement_succeeded"}
    assert classify(manifest, {"valid": True})[0] == "fixture_only"


def test_native_success_requires_this_physical_episode_in_the_export():
    arms = ("leader", "follower")
    meta = {"status": "complete", "frames": 3, "frames_sha256": "b",
            "provenance": {"live_calibration": {arm: {"verified_live": True, "snapshot_sha256": arm} for arm in arms}}}
    complete = {"id": "native2", "storage": "episodes", "metadata": meta}
    card = {"attempt_limit": 1, "attempts": [{"source_world_episode_id": "w"}], "scene_group_id": "g"}
    label = {"outcome": "placement_succeeded"}
    validation = {"source": "hardware", "technical_only": False, "calibration": {"files": {arm: {"sha256": arm} for arm in arms}},
                  "episodes": [{"episode_id": "native1", "frames": 3, "frames_sha256": "a"}]}
    stale = status_for(complete, label, card, 0, validation)
    assert stale["reasons"] == ["native_export_episode_not_verified"]
    assert stale["outcome"] == "placement_succeeded"
    snapshot = {"episode_id": "native2", "frames": 3, "frames_sha256": "a", "episode_manifest_sha256": "raw-meta"}
    validation["episodes"].append(snapshot)
    assert status_for(complete, label, card, 0, validation)["classification"] == "rejected"
    snapshot["frames_sha256"] = "b"
    assert status_for(complete, label, card, 0, validation)["classification"] == "motor_review_excluded"
    label.update(recording=dict(snapshot), assistance="manual_scene_assistance")
    assisted = status_for(complete, label, card, 0, validation)
    assert assisted["classification"] == "motor_review_excluded" and assisted["outcome"] == "placement_succeeded"
    label["assistance"] = "attended_native_motion"
    assert status_for(complete, label, card, 0, validation)["classification"] == "eligible_for_motor_review"
    missing_calibration = status_for(complete, label, card, 0, {**validation, "calibration": {}})
    assert missing_calibration["reasons"] == ["native_calibration_readbacks_unverified"]
    technical = {**validation, "source": "synthetic_test", "technical_only": True}
    assert status_for(complete, label, card, 0, technical)["classification"] == "rejected"
    snapshot["episode_manifest_sha256"] = "changed-native-meta"
    assert status_for(complete, label, card, 0, validation)["outcome"] == "unknown"


def test_preserved_f6_full_binding_and_truth_perturbation(tmp_path):
    result = audit_b1(ROOT / "evidence", tmp_path)
    assert result["rule_prediction_invariant"]
    assert set(result["actual_truth_permutation"]) == {"left", "surface"}
    assert result["resolved_controls"] == {"F1": "left", "F2": "surface"}
    assert result["decision"] == "request_inspection"


def test_actual_native_and_teacher_rehearsals():
    native = load(EVIDENCE / "native-studio-rehearsal.json")
    assert native["export"]["official_verification"]["rgb_pixels_bit_exact"]
    assert native["B_consumption"]["rgb_camera_order"] == ["wrist", "top"]
    assert native["B_consumption"]["technical_only"]
    assert not native["physical_capture_exercised"]
    teacher = load(EVIDENCE / "teacher-replays.json")
    assert len(teacher["cases"]) == 5 and teacher["teacher_gate"]
    act = load(EVIDENCE / "act-development-diagnostic.json")
    assert set(act["sample_trace"]["preprocessing_max_abs_residual"].values()) == {0.0}
    assert all(row["first_failed_stage"] == "approach" for row in act["cases"])


def test_native_intake_waits_for_stop_and_validates_once(monkeypatch, tmp_path):
    from smart_swarm import operator

    (tmp_path / "so101-pilot").mkdir()
    state = {"recording_active": True, "teleoperation_active": False, "policy_active": False}
    synced = []
    binding = {"dataset_id": "ds_test"}
    fingerprint = {"recordings": [["native_saved", "episodes", "complete"]]}
    status = {"counts": {"attempts": 1}, "validation_rejection": None}
    monkeypatch.setattr(operator, "native_status", lambda cfg: state)
    monkeypatch.setattr(operator, "native_binding", lambda cfg: binding)
    monkeypatch.setattr(operator, "intake_fingerprint", lambda cfg, item: fingerprint)
    monkeypatch.setattr(operator, "sync_pilot", lambda cfg, item: synced.append(item) or status)
    config = {"data_root": tmp_path}
    operator.observe_intake(config)
    assert synced == []
    state["recording_active"] = False
    operator.observe_intake(config)
    operator.observe_intake(config)
    assert synced == [binding]


def test_native_intake_retries_blocked_generation_with_bounded_delays(monkeypatch, tmp_path):
    from smart_swarm import operator

    root = tmp_path / "so101-pilot"
    root.mkdir()
    config = {"data_root": tmp_path}
    path = root / "observed-input.json"
    binding = {"dataset_id": "ds_test"}
    clock = [100.]
    transient = {"counts": {"attempts": 1}, "validation_rejection": "Native busy", "validation_retryable": True}
    recovered = {"counts": {"attempts": 1}, "validation_rejection": None}
    reports = iter([transient] * 4 + [recovered])
    monkeypatch.setattr(operator.time, "time", lambda: clock[0])
    monkeypatch.setattr(operator, "intake_fingerprint", lambda cfg, item: {"recordings": [["native_saved"]]})
    monkeypatch.setattr(operator, "sync_pilot", lambda cfg, item: next(reports))
    for due, delay in ((100, 10), (110, 30), (140, 60)):
        clock[0] = due
        assert operator.sync_observed_intake(config, binding)["automatic_retry"] == "retry_pending"
        assert load(path)["next_retry_unix"] == due + delay
        clock[0] = due + delay - 1
        assert operator.sync_observed_intake(config, binding) is None
    clock[0] = 200.
    assert operator.sync_observed_intake(config, binding)["automatic_retry"] == "retry_exhausted"
    clock[0] = 1000.
    assert operator.sync_observed_intake(config, binding) is None
    assert operator.sync_observed_intake(config, binding, force=True)["automatic_retry"] == "checked"
    assert len(load(path)["transient_failures"]) == 4
    assert operator.sync_observed_intake(config, binding) is None


def test_native_intake_does_not_retry_invalid_recordings_until_input_changes(monkeypatch, tmp_path):
    from smart_swarm import operator

    (tmp_path / "so101-pilot").mkdir()
    config = {"data_root": tmp_path}
    binding = {"dataset_id": "ds_test"}
    inputs = ["bad ledger"]
    outcomes = iter([{"validation_rejection": "Changed frame ledger"}, {"validation_rejection": None}])
    monkeypatch.setattr(operator, "intake_fingerprint", lambda cfg, item: {"native_manifest": inputs[-1]})
    monkeypatch.setattr(operator, "sync_pilot", lambda cfg, item: next(outcomes))
    assert operator.sync_observed_intake(config, binding)["automatic_retry"] == "rejected"
    assert operator.sync_observed_intake(config, binding) is None
    inputs.append("new verified recording")
    assert operator.sync_observed_intake(config, binding)["automatic_retry"] == "checked"


def test_native_intake_uses_one_receipt_writer_across_launcher_windows(monkeypatch, tmp_path):
    from fcntl import LOCK_EX, flock

    from smart_swarm import operator
    from smart_swarm.contract import ProtocolError

    root = tmp_path / "so101-pilot"
    root.mkdir()
    config = {"data_root": tmp_path}
    binding = {"dataset_id": "ds_test"}
    monkeypatch.setattr(operator, "intake_fingerprint", lambda cfg, item: {"recordings": [["native_saved"]]})
    checked = []
    monkeypatch.setattr(operator, "sync_pilot", lambda cfg, item: checked.append(item) or {})
    with (root / "intake-receipt.lock").open("a") as owner:
        flock(owner, LOCK_EX)
        assert operator.sync_observed_intake(config, binding) is None
        with pytest.raises(ProtocolError, match="Another Smart Swarm window"):
            operator.sync_observed_intake(config, binding, force=True)
    assert checked == []
    assert operator.sync_observed_intake(config, binding)["automatic_retry"] == "checked"
    assert checked == [binding]


def test_story_preflight_checks_native_rgb_without_general_sensor_dependencies():
    from smart_swarm.fleet_adapter import rgb_readiness
    from smart_swarm.physical_fleet_timing import CLOCK_KIND

    live = {"running": True, "read_only": True, "motion_commands_sent": 0, "robots": {
        role: {"ages_s": {"rgb": 0.1}, "uncertainty_ms": 24.0, "clock_kind": CLOCK_KIND, "boot_id": role,
               "conversion_errors": {"depth": "optional stream unavailable"}} for role in ("tank", "rosmaster")}}
    assert rgb_readiness(live)["ready"]
    live["robots"]["tank"]["boot_id"] = ""
    assert "boot identity" in rgb_readiness(live)["roles"]["tank"]["clock_issue"]
    assert not rgb_readiness(live)["ready"]
    live["robots"]["tank"]["boot_id"] = "tank"
    assert not rgb_readiness({**live, "motion_commands_sent": 1})["ready"]
    live["robots"]["tank"]["conversion_errors"]["rgb"] = "unusable native format"
    report = rgb_readiness(live)
    assert not report["ready"] and not report["roles"]["tank"]["rgb_fresh"]
    assert report["roles"]["tank"]["rgb_conversion_error"] == "unusable native format"
    del live["robots"]["tank"]["conversion_errors"]["rgb"]
    live["robots"]["tank"]["ages_s"]["rgb"] = 1.01
    live["robots"]["rosmaster"].update(uncertainty_ms=None, error="Clock samples expired")
    report = rgb_readiness(live)
    assert not report["ready"]
    assert report["roles"]["tank"]["clock_within_limit"] and not report["roles"]["tank"]["rgb_fresh"]
    assert report["roles"]["rosmaster"]["rgb_fresh"] and not report["roles"]["rosmaster"]["clock_within_limit"]
    assert "rosmaster: Clock samples expired" in report["issues"]


def test_studio_reports_current_stop_and_distinguishes_historical_control(monkeypatch):
    from smart_swarm import studio

    preview = {"active": True, "camera_only": True, "torque_enabled": False, "cameras": {
        "wrist": {"connected": True, "last_frame_age_ms": 30},
        "top": {"connected": True, "last_frame_age_ms": 510}}}
    teleop = {"active": False, "phase": "following", "hardware_open": False,
              "hardware_stop_state": studio.STOP_UNCONFIRMED,
              "provenance": {"live_calibration": {"leader": {"verified_live": True, "timestamp_ns": 10}}}}
    report = {"teleoperation": teleop, "recording": {}, "jobs": [], "hardware_stop_generation": 1,
              "resources": {"device_inventory": {}}, "preview": preview}
    monkeypatch.setattr(studio, "native_request", lambda *_: report)
    status = studio.native_status({"studio_origin": "local"})
    assert status["teleoperation_phase"] is None
    assert status["native_stop_check_required"] and not status["native_arm_hardware_open"]
    assert status["native_calibration_readback"]["leader"] is True
    assert status["native_calibration_readback_monotonic_ns"]["leader"] == 10
    assert status["native_camera_preview"]["wrist"]["fresh"] and not status["native_camera_preview"]["top"]["fresh"]
    assert not status["physical_preflight_verified"]
    teleop.update(active=True, hardware_open=True, hardware_stop_state=None)
    status = studio.native_status({"studio_origin": "local"})
    assert status["teleoperation_phase"] == "following" and status["native_arm_hardware_open"]


def attended_studio_snapshots():
    from copy import deepcopy

    from smart_swarm.attended_so101 import DEVICES, FLAGS

    before = {**dict.fromkeys(FLAGS, False), "inventory_present": dict.fromkeys(DEVICES, True),
              "native_device_identities": {key: f"/dev/{key}" for key in DEVICES},
              "native_camera_preview": {key: {"fresh": True} for key in ("wrist", "top")},
              "native_stop_generation": 1, "native_stop_state": {"teleoperation": "devices_closed"},
              "native_calibration_readback": {"leader": True, "follower": True},
              "native_calibration_readback_monotonic_ns": {"leader": 50, "follower": 51},
              "native_following_readback": {"state": "stopped", "phase": "following", "source": "hardware",
                  "last_sample_ns": 70, "samples": 20, "error": None, "stop_errors": [],
                  "calibration_sha256": {"leader": "a" * 64, "follower": "b" * 64}}}
    after = deepcopy(before)
    after.update(native_stop_generation=2, native_calibration_readback_monotonic_ns={"leader": 120, "follower": 130})
    after["native_following_readback"]["last_sample_ns"] = 180
    return before, after


def test_attended_check_keeps_operator_and_native_stop_evidence(monkeypatch, tmp_path):
    from types import SimpleNamespace

    from smart_swarm import attended_so101, physical_story_source

    before, after = attended_studio_snapshots()
    snapshots = iter((before, after))
    instants = iter((100, 200))
    monkeypatch.setattr(attended_so101, "native_status", lambda _: next(snapshots))
    monkeypatch.setattr(attended_so101, "time", SimpleNamespace(monotonic_ns=lambda: next(instants), time=lambda: 10.))
    monkeypatch.setattr(physical_story_source, "homebrain_boot", lambda: "boot-a")
    monkeypatch.setattr("builtins.input", lambda _: "y")
    config = {"data_root": tmp_path}
    result = attended_so101.record_attended(config)
    assert result["recorded"] and result["native_stop_generation"] == 2
    receipt = load(Path(result["receipt"]))
    assert receipt["operator_confirmed_framing_tracking_and_stop"] and not receipt["motion_requested_by_smart_swarm"]
    current = attended_so101.attended_status(config, {**after, "native_stop_generation": 3})
    assert current["current"] and current["verified_stop_generation"] == 2
    team = {"so101_studio": {**after, "learning_job_active": False}, "active_native_capture": None,
            "story_observation_readiness": {"issues": ["No mobile clock evidence"]}}
    monkeypatch.setattr(physical_story_source, "native_team_status", lambda *_args, **_kwargs: team)
    monkeypatch.setattr(physical_story_source, "fleet_capacity", lambda _: {"free_bytes": 2 << 30,
                        "native_minimum_free_bytes": 1 << 30})
    readiness = physical_story_source.story_readiness(config)
    assert readiness["so101_studio"]["physical_preflight_verified"] and not readiness["ready_for_native_lease_check"]
    assert readiness["pending_attended_checks"] == ["Distributed scene and witness framing"]
    changed = {**after, "native_device_identities": {**after["native_device_identities"], "wrist": "/dev/interchanged"}}
    assert not attended_so101.attended_status(config, changed)["current"]
    monkeypatch.setattr(physical_story_source, "homebrain_boot", lambda: "boot-b")
    assert not attended_so101.attended_status(config, after)["current"]


def test_attended_check_refuses_an_old_follow_or_unacknowledged_stop():
    from smart_swarm.attended_so101 import verify_window

    before, after = attended_studio_snapshots()
    with pytest.raises(ValueError, match="new native STOP"):
        verify_window(before, before, 100, 200)
    with pytest.raises(ValueError, match="new successful native following"):
        verify_window(before, {**before, "native_stop_generation": 2}, 100, 200)
    after["native_calibration_readback"]["follower"] = False
    with pytest.raises(ValueError, match="both saved calibrations"):
        verify_window(before, after, 100, 200)


def test_distributed_start_refuses_unconfirmed_native_stop_before_recorder_check(monkeypatch):
    from smart_swarm import physical_story_source as story
    from smart_swarm.contract import ProtocolError

    state = {"inventory_present": {"wrist": True, "top": True, "leader": True, "follower": True},
             "recording_active": False, "teleoperation_active": False, "policy_active": False,
             "learning_job_active": False, "native_stop_check_required": True, "native_arm_hardware_open": False}
    team = {"so101_studio": state, "story_observation_readiness": {"ready": True, "issues": []},
            "active_native_capture": "existing"}
    monkeypatch.setattr(story, "native_team_status", lambda _, **kw: team)
    monkeypatch.setattr(story, "fleet_capacity", lambda _: {"free_bytes": 2 << 30, "native_minimum_free_bytes": 1 << 30})
    monkeypatch.setattr(story.subprocess, "run", lambda *_, **kw: pytest.fail("Native lease should stay untouched"))
    report = story.story_readiness({})
    assert not report["ready_for_native_lease_check"]
    assert [issue["code"] for issue in report["blocking_issues"]] == ["OFFLINE", "RESOURCE_BUSY"]
    with pytest.raises(ProtocolError, match="hardware STOP"):
        story.story_preflight({})


def healthy_distributed_readback(monkeypatch, story):
    _, studio = attended_studio_snapshots()
    team = {"so101_studio": studio, "active_native_capture": None,
            "story_observation_readiness": {"ready": True, "issues": []}}
    monkeypatch.setattr(story, "native_team_status", lambda *_args, **_kw: team)
    monkeypatch.setattr(story, "fleet_capacity", lambda _: {"free_bytes": 2 << 30,
                        "native_minimum_free_bytes": 1 << 30})


def test_story_preflight_separates_busy_native_reservation_from_helper_failure(monkeypatch):
    from subprocess import CompletedProcess, TimeoutExpired

    from smart_swarm import physical_story_source as story, studio
    from smart_swarm.contract import ProtocolError

    healthy_distributed_readback(monkeypatch, story)
    monkeypatch.setattr(studio, "studio_env", lambda _: {})
    config = {"studio_python": "native-python"}
    for result, code in ((75, "RESOURCE_BUSY"), (1, "OFFLINE")):
        monkeypatch.setattr(story.subprocess, "run", lambda *_, **kw: CompletedProcess([], result, stderr="helper error"))
        with pytest.raises(ProtocolError) as error:
            story.story_preflight(config)
        assert error.value.code == code
        assert not story.story_readiness(config)["recording_reservation_available"]

    def timeout(*args, **kwargs):
        raise TimeoutExpired(args[0], kwargs["timeout"])

    monkeypatch.setattr(story.subprocess, "run", timeout)
    with pytest.raises(ProtocolError, match="reservation did not answer"):
        story.story_preflight(config)


def test_story_start_requires_the_current_attended_receipt(monkeypatch):
    from smart_swarm import physical_story_source as story
    from smart_swarm.contract import ProtocolError

    healthy_distributed_readback(monkeypatch, story)
    monkeypatch.setattr(story, "require_recording_lease", lambda _: None)
    pending = story.story_readiness({})
    assert pending["status"] == "SOFTWARE_READY_AWAITING_ATTENDED_CHECK" and pending["recording_reservation_available"]
    with pytest.raises(ProtocolError, match="attended SO101 setup"):
        story.story_preflight({})
    story.story_preflight({}, require_attended=False)
    monkeypatch.setattr(story, "attended_status", lambda *_: {"current": True, "receipt": "technical"})
    ready = story.story_readiness({})
    assert ready["status"] == "READY_FOR_STORY_COLLECTION"
    assert ready["recording_reservation_available"] and ready["blocking_issues"] == []
    assert ready["pending_attended_checks"] == ["Distributed scene and witness framing"]
    story.story_preflight({})


def test_story_readiness_applies_the_existing_native_storage_floor(monkeypatch, tmp_path):
    from smart_swarm import physical_story_source as story

    native = {"inventory_present": {key: True for key in ("wrist", "top", "leader", "follower")},
              "native_stop_check_required": False, "native_arm_hardware_open": False,
              "recording_active": False, "teleoperation_active": False, "policy_active": False, "learning_job_active": True}
    team = {"so101_studio": native, "story_observation_readiness": {"issues": []}, "active_native_capture": None}
    monkeypatch.setattr(story, "native_team_status", lambda *_args, **_kwargs: team)
    space = {"free_bytes": 1024**3 - 1, "native_minimum_free_bytes": 1024**3}
    monkeypatch.setattr(story, "fleet_capacity", lambda _: dict(space))
    monkeypatch.setattr(story, "require_recording_lease", lambda _: None)
    config = {"data_root": tmp_path}
    blocked = story.story_readiness(config)
    assert [row["code"] for row in blocked["blocking_issues"]] == ["STORAGE_LIMIT"]
    assert blocked["recording_reservation_available"] is None
    assert not blocked["motion_requested"] and not blocked["recording_started"]
    space["free_bytes"] += 1
    assert story.story_readiness(config)["ready_for_native_lease_check"]


def test_story_capture_budget_follows_remaining_native_archives(tmp_path):
    from smart_swarm.episode import atomic
    from smart_swarm.physical_story_plan import prepare
    from smart_swarm.physical_story_source import capture_budget

    config = {"data_root": tmp_path}
    space = {"free_bytes": 5 << 30, "native_minimum_free_bytes": 1 << 30}
    estimates = [300 << 20, 320 << 20]
    reference = {"estimated_60s_payload_bytes": estimates}
    assert capture_budget(config, space, reference) is None
    assert not list(tmp_path.iterdir())
    plan = prepare(config)
    archived = [{"status": "complete", "fleet_capture_id": "saved"}, {"status": "incomplete"},
                {"status": "recording", "fleet_capture_id": "acknowledged"}, {"status": "placement_recording"}]
    for card, state in zip(plan["cards"], archived):
        journal = tmp_path / "physical-stories" / card["story_id"] / "trial.json"
        journal.parent.mkdir(parents=True)
        atomic(journal, state)
    result = capture_budget(config, space, reference)
    assert result["next_block"]["block"] == 1 and result["next_block"]["archives_remaining"] == 3
    deck = result["remaining_deck"]
    assert deck["archives_remaining"] == 21
    expected = [space["native_minimum_free_bytes"] + 21 * size for size in estimates]
    assert deck["estimated_bytes_with_native_reserve"] == expected
    assert deck["free_bytes_beyond_upper_estimate"] == space["free_bytes"] - expected[-1] < 0


def test_native_mobile_probe_preserves_wrapped_socket_error_codes(monkeypatch):
    import errno
    import socket

    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    from native_fleet_story_capture import transport_failure

    for code, expected in ((socket.EAI_NONAME, "name_resolution_failed"),
                           (errno.EHOSTUNREACH, "host_unreachable"), (errno.ECONNREFUSED, "connection_refused")):
        error = ConnectionError(f"[Errno {code}] Native connection failed")
        result = transport_failure(error)
        assert result["errno"] == code and result["status"] == expected


def test_framing_never_reads_images_during_active_native_recording(monkeypatch, tmp_path):
    from smart_swarm import physical_framing as framing
    from smart_swarm.contract import ProtocolError

    state = {"recording": {"active": True}, "teleoperation": {}, "preview": {}, "jobs": []}
    monkeypatch.setattr(framing, "native_request", lambda *_: state)
    monkeypatch.setattr(framing, "read_preview_jpeg", lambda *_: pytest.fail("Preview read during active recording"))
    with pytest.raises(ProtocolError, match="Studio control and recording are idle"):
        framing.capture_framing({"data_root": tmp_path})
    assert not list(tmp_path.iterdir())


def test_framing_discards_images_when_stop_changes(monkeypatch, tmp_path):
    from io import BytesIO

    from PIL import Image

    from smart_swarm import physical_framing as framing
    from smart_swarm.contract import ProtocolError

    stream = {"width": 8, "height": 6}
    cameras = {role: {"stream": stream, "frames_received": 10 + index} for index, role in enumerate(framing.ROLES)}
    reads = iter({"stop_generation": generation, "cameras": cameras} for generation in (1, 2))
    output = BytesIO()
    Image.new("RGB", (8, 6), "white").save(output, format="JPEG")
    monkeypatch.setattr(framing, "snapshot", lambda _: next(reads))
    monkeypatch.setattr(framing, "read_preview_jpeg", lambda *_: output.getvalue())
    with pytest.raises(ProtocolError, match="STOP changed"):
        framing.capture_framing({"data_root": tmp_path})
    assert not list(tmp_path.iterdir())


def test_mobile_transport_probe_keeps_stale_hub_blocked(monkeypatch):
    import json
    from subprocess import CompletedProcess

    from smart_swarm import physical_story_source as story

    native = {"rgb_fresh": False, "clock_within_limit": False}
    routes = [{"role": role, "isolated": isolated, "port": port + int(isolated), "status": result}
              for role, result, port in (("tank", "clock_sample", 8043), ("rosmaster", "host_unreachable", 8053))
              for isolated in (False, True)]
    report = {"ready_for_native_lease_check": False, "blocking_issues": [{"code": "OFFLINE"}],
              "mobile_rgb_and_clocks": {"roles": {"tank": native, "rosmaster": native}}}
    monkeypatch.setattr(story, "story_readiness", lambda _: dict(report))
    monkeypatch.setattr(story, "fleet_invocation", lambda _, *options: (list(options), {}))

    def run(args, **_):
        assert args == ["--probe-mobile"]
        value = {"routes": routes, "motion_requested": False, "recording_started": False}
        return CompletedProcess(args, 0, stdout=json.dumps(value))

    monkeypatch.setattr(story.subprocess, "run", run)
    checked = story.story_readiness_with_transport({"explorer_python": "python", "explorer_project": "explorer"})
    assert checked["ready_for_native_lease_check"] is False and checked["blocking_issues"] == report["blocking_issues"]
    assert checked["mobile_native_transport"]["next_check_by_role"]["tank"].startswith("Check this role's RGB")
    assert checked["mobile_native_transport"]["next_check_by_role"]["rosmaster"].startswith("On rosmaster's console")


def test_story_preflight_refuses_unavailable_explorer_helper_before_reserving_studio(monkeypatch):
    from smart_swarm import physical_story_source as story
    from smart_swarm.contract import ProtocolError

    native = {"inventory_present": {"wrist": True, "top": True, "leader": True, "follower": True},
              "native_stop_check_required": False, "native_arm_hardware_open": False,
              "recording_active": False, "teleoperation_active": False, "policy_active": False,
              "learning_job_active": False}
    team = {"so101_studio": native, "story_observation_readiness": {"issues": []}, "active_native_capture": None}
    monkeypatch.setattr(story, "native_team_status", lambda *_args, **_kwargs: team)
    monkeypatch.setattr(story, "fleet_capacity", lambda _: {"readback_error": "recorder unavailable"})
    monkeypatch.setattr(story.subprocess, "run", lambda *_args, **_kwargs: pytest.fail("Studio lease should stay untouched"))
    with pytest.raises(ProtocolError, match="native Explorer recorder helper") as error:
        story.story_preflight({})
    assert error.value.code == "OFFLINE"


def test_native_clock_replay_keeps_evaluator_frame_outside_policy_scope(tmp_path):
    import numpy as np
    from smart_swarm.contract import interval
    from smart_swarm.episode import Recorder, Replay, TRUTH

    clock = {"host_id": "homebrain", "boot_id": "test-boot", "kind": "monotonic"}
    source = tmp_path / "story"
    agent = "rosmaster-1"
    sensor = agent + "/front"
    recorder = Recorder(source, "native-test", "test-group", {agent: [sensor]},
                        {"run_mode": "hardware", "native_recording_clock": clock}, clock=clock)
    image = np.zeros((4, 4, 3), dtype=np.uint8)
    private = recorder.capture(agent, {sensor: image}, 10, [], [], exposures={sensor: 8})
    reveal = recorder.frame(TRUTH, "evaluator/top", 1, image, 20)
    recorder.append(TRUTH, "reveal", {"camera": reveal, "target_state_time_or_interval": interval(15, 20, clock),
                                      "relations_preserved_until_reveal": True})
    recorder.close(12, 15, 20)
    replay = Replay(source, agent)
    assert replay.packets() == [private]
    assert replay.image(private["cameras"][0]["frame_id"]).size == (4, 4)
    with pytest.raises(KeyError):
        replay.image(reveal["frame_id"])


def test_smolvla_chunk_transport_refreshes_and_resets_at_the_same_issue(monkeypatch):
    from collections import deque
    from io import StringIO
    from types import SimpleNamespace

    from smart_swarm import evaluate_motor
    from smart_swarm.motor_input import CLOCKED_PROFILE

    worker = object.__new__(evaluate_motor.Worker)
    worker.input_profile, worker.smolvla_chunks = CLOCKED_PROFILE, True
    worker.deadline = None
    worker.pending_actions = deque()
    worker.process = SimpleNamespace(stdin=StringIO())
    replies = iter(({"actions": [[1.] * 6, [2.] * 6]}, {"reset": True}, {"actions": [[3.] * 6, [4.] * 6]}))
    worker.recv = lambda: next(replies)
    issues = []
    monkeypatch.setattr(evaluate_motor, "action_request", lambda *_, issue_index: issues.append(issue_index) or {})
    executor = SimpleNamespace(observation_index=lambda: 0)
    assert evaluate_motor.action(object(), worker, executor) == [1.] * 6
    executor.observation_index = lambda: 1
    assert evaluate_motor.action(object(), worker, executor) == [2.] * 6
    worker.pending_actions.append([9.] * 6)
    worker.request({"type": "reset", "seed": 2})
    executor.observation_index = lambda: 0
    assert evaluate_motor.action(object(), worker, executor) == [3.] * 6
    assert issues == [0, 0]


def test_smolvla_schedule_uses_real_slots_inside_the_training_allowance():
    from smart_swarm.motor_full import smolvla_schedule

    probe = {"checkpoint_load_wall_s": 16.4, "real_minibatch_forward_and_backward_wall_s": 9.9,
             "model_weights_bytes": 1_800_184_704, "optimizer_state_bytes_estimate": 799_048_556}
    large = smolvla_schedule(probe, 1_500, 7_000_000_000)
    small = smolvla_schedule(probe, 1_500, 4_000_000_000)
    assert large["checkpoint_storage_slots"] == 2 and 0 < large["save_freq"] < large["steps"]
    assert small["checkpoint_storage_slots"] == 1 and small["save_freq"] == small["steps"] < 2_000
    with pytest.raises(ValueError, match="one complete"):
        smolvla_schedule(probe, 1_500, 2_000_000_000)


@pytest.mark.parametrize("module", ("run_compact_motor_comparison", "probe_precontact_act", "audit_act_static_raster"))
def test_motor_probe_import_allows_egl_configuration_before_mujoco(module):
    import os
    import subprocess
    import sys

    code = '''
import importlib
import sys
runner = importlib.import_module(sys.argv[1])

assert "mujoco" not in sys.modules
runner.configure_renderer(runner.settings())
import smart_swarm.motor_full
from mujoco.rendering.classic.gl_context import GLContext

assert GLContext.__module__ == "mujoco.egl"
'''
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src") + os.pathsep + str(ROOT / "scripts")}
    subprocess.run([sys.executable, "-B", "-c", code, module], cwd=ROOT, env=env, check=True, timeout=20)


def test_motor_evaluation_requires_matching_conditioned_profile_and_camera_roles(tmp_path):
    from smart_swarm.episode import atomic
    from smart_swarm.evaluate_motor import bound_input_profile
    from smart_swarm.motor_input import CLOCKED_PROFILE, PROFILE

    checkpoint = tmp_path / "checkpoint"
    split = tmp_path / "validation"
    checkpoint.mkdir()
    split.mkdir()
    assert bound_input_profile(checkpoint, split) == "historical-rgb"
    binding = checkpoint / "smart-swarm-input.json"
    lineage = split / "smart-swarm-lineage.json"
    declared = {"input_profile": PROFILE, "control_timing": "aligned-sample-v2",
                "allowed_images": {"wrist": "observation.images.camera1", "top": "observation.images.camera2"}}
    atomic(binding, {"input_profile": PROFILE})
    with pytest.raises(ValueError, match="matching validation inputs"):
        bound_input_profile(checkpoint, split)
    atomic(lineage, declared)
    assert bound_input_profile(checkpoint, split) == PROFILE
    atomic(binding, {"input_profile": CLOCKED_PROFILE})
    with pytest.raises(ValueError, match="matching validation inputs"):
        bound_input_profile(checkpoint, split)
    atomic(lineage, {**declared, "input_profile": CLOCKED_PROFILE})
    assert bound_input_profile(checkpoint, split) == CLOCKED_PROFILE
    atomic(lineage, {**declared, "input_profile": CLOCKED_PROFILE,
                     "allowed_images": {"wrist": "observation.images.camera2", "top": "observation.images.camera1"}})
    with pytest.raises(ValueError, match="camera roles"):
        bound_input_profile(checkpoint, split)


def test_motor_evaluation_saves_startup_error_without_launching_a_policy(tmp_path, monkeypatch):
    from smart_swarm import evaluate_motor
    from smart_swarm.episode import atomic
    from smart_swarm.motor_input import PROFILE

    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "model.safetensors").write_bytes(b"fixture")
    atomic(checkpoint / "smart-swarm-input.json", {"input_profile": PROFILE})
    monkeypatch.setattr(evaluate_motor.subprocess, "Popen", lambda *_, **kw: pytest.fail("No worker should launch"))
    output = tmp_path / "evaluation"
    with pytest.raises(ValueError, match="matching validation inputs"):
        evaluate_motor.evaluate(None, checkpoint, tmp_path / "data", output, 1, 30, 30)
    receipt, = output.glob("*/evaluation.json")
    result = load(receipt)
    assert result["status"] == "partial" and result["results"] == []
    assert result["successful_resets"] == 0 and "matching validation inputs" in result["reason"]


def test_motor_evaluation_only_selects_gpu_rendering_inside_native_admission(tmp_path, monkeypatch):
    from contextlib import contextmanager

    from smart_swarm import cli, evaluate_motor
    from smart_swarm.training import offline

    events = []
    grant = {"reserved": True, "gpu_render": True, "locks": ["cpu_compute", "recording", "gpu_compute"]}

    @contextmanager
    def reservation(config, job, *, gpu_render):
        assert gpu_render and job == "smart-swarm-evaluate-motor"
        events.append("admitted")
        yield grant, lambda: None
        events.append("released")

    def configure(config, *, grant):
        assert events == ["admitted"] and grant["gpu_render"]
        events.append("configured")

    def evaluate(*args, reservation, heartbeat):
        assert reservation == grant and offline()["CUDA_VISIBLE_DEVICES"] == ""
        heartbeat()
        events.append("evaluated")
        return "result"

    monkeypatch.setattr(cli, "admitted", reservation)
    monkeypatch.setattr(cli, "configure_renderer", configure)
    monkeypatch.setattr(evaluate_motor, "evaluate", evaluate)
    args = cli.parser().parse_args(["evaluate", "--protocol", "motor-pilot", "--checkpoint", "path", "--gpu-render"])
    assert cli.motor_command({"asset_root": tmp_path, "data_root": tmp_path}, args) == "result"
    assert events == ["admitted", "configured", "evaluated", "released"]


def test_cuda_policy_requires_the_native_compute_grant(monkeypatch):
    from smart_swarm.research_jobs import admitted, studio_lease_mode, verify_compute_scope
    from smart_swarm.training import offline

    grant = {"reserved": True, "backend": "so101_lab.resources.ResourceLease", "device": "cuda",
             "gpu_render": False, "locks": ["cpu_compute", "recording", "gpu_compute"]}
    verify_compute_scope("cuda", grant)
    with pytest.raises(ValueError, match="native Studio compute grant"):
        verify_compute_scope("cuda", {**grant, "gpu_render": True})
    with pytest.raises(ValueError, match="native Studio compute grant"):
        verify_compute_scope("cuda", {**grant, "locks": ["cpu_compute"]})
    with pytest.raises(ValueError, match="Studio reservation"):
        with admitted({}, "test-cuda", device="cuda"):
            pytest.fail("A platform without native Studio admission must deny CUDA")
    assert studio_lease_mode(False, False, "cuda")[0] == ["--gpu-compute"]
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    assert offline()["CUDA_VISIBLE_DEVICES"] == ""
    assert offline(device="cuda")["CUDA_VISIBLE_DEVICES"] == "0"


def test_motor_visual_mesh_merges_duplicate_faces_and_preserves_surviving_winding():
    import numpy as np

    from smart_swarm.motor_visual_mesh import reduced_faces

    vertices = np.array([[0, 0, 0], [.001, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
    faces = np.array([[0, 2, 3], [1, 3, 2], [2, 3, 1], [0, 1, 3], [3, 4, 2]])
    before = faces.copy()
    reduced, retained = reduced_faces(vertices, faces)
    assert retained.tolist() == [0, 1, 4]
    assert reduced.tolist() == [[0, 2, 3], [0, 3, 2], [3, 4, 2]]
    assert np.array_equal(faces, before)


def test_motor_issue_clock_changes_only_its_observation_region():
    import numpy as np
    from smart_swarm.motor_input import conditioned_top

    source = np.full((120, 160, 3), 127, dtype=np.uint8)
    before = conditioned_top(source, "cyan", issue_index=41)
    after = conditioned_top(source, "cyan", issue_index=58)
    changed = np.any(before != after, axis=2)
    assert changed[2:14, 16:60].any()
    changed[2:14, 16:60] = False
    assert not changed.any()
    assert (source == 127).all()
    with pytest.raises(ValueError):
        conditioned_top(source, "cyan", issue_index=1024)


def test_f3_motor_segment_ends_before_removal(tmp_path):
    import json
    from smart_swarm.physical_motor_prefix import segment_choice
    from smart_swarm.physical_story_source import story_phases

    raw = tmp_path / "data/native-set/episodes/native-episode"
    raw.mkdir(parents=True)
    sample_rows = [{"frame_id": i, "observation_timestamp_ns": i, "action_timestamp_ns": i + 1,
                    "state_valid": True, "action_valid": True} for i in range(1, 81)]
    (raw / "frames.jsonl").write_text("\n".join(json.dumps(row) for row in sample_rows))
    state = {"card": {"family": "F3", "story_id": "story-f3", "source_group_id": "group", "split": "development"},
             "start": {"monotonic_ns": 1}, "studio_dataset_id": "native-set", "native_episode_id": "native-episode",
             "markers": {"insertion_closed": {"monotonic_ns": 45, "observed_placement_outcome": "placement_succeeded",
                                              "assistance": "attended_native_motion"},
                         "placement_closed": {"monotonic_ns": 75, "observed_placement_outcome": "unknown"}}}
    assert story_phases(state) == ("insertion_closed", "placement_closed", "relocation_closed")
    selected, reason = segment_choice({"studio_project": tmp_path}, state)
    assert reason is None
    assert selected["end_boundary"] == "insertion_closed"
    assert selected["frame_index_end_inclusive"] == selected["sample_count"] == 44
    assert selected["last_transmitted_action_ns"] == 45
    assert selected["native_episode"] == "native-episode" and selected["scene_group_id"] == "group"
    state["markers"]["insertion_closed"]["observed_placement_outcome"] = "unknown"
    selected, reason = segment_choice({"studio_project": tmp_path}, state)
    assert selected is None and "no observed successful placement" in reason
    state["markers"]["insertion_closed"].update(observed_placement_outcome="placement_succeeded",
                                                assistance="human_hand_assistance")
    selected, reason = segment_choice({"studio_project": tmp_path}, state)
    assert selected is None and "assistance outside native arm motion" in reason


def test_story_interrupt_preserves_index_and_charges_the_attempt(monkeypatch, tmp_path):
    from smart_swarm import evidence_portfolio as portfolio

    plan = [("train", "F1", 0, 100000), ("train", "F1", 0, 110000)]
    monkeypatch.setattr(portfolio, "group_plan", lambda: iter(plan))

    def capture(assets, root, group, seed, deadline):
        if seed == 100000:
            raise KeyboardInterrupt
        return [{"split": "train", "settled_placement": "cyan", "relocation_assignment": "cyan_left", "terminal": "left"}]

    monkeypatch.setattr(portfolio, "capture_group", capture)
    with pytest.raises(KeyboardInterrupt):
        portfolio.generate(None, tmp_path, lambda: None)
    first = load(tmp_path / "portfolio.json")
    assert first["rejected"] == 1 and first["generation_elapsed_s"] > 0
    resumed = portfolio.generate(None, tmp_path, lambda: None)
    assert [row["seed"] for row in resumed["attempts"]] == [100000, 110000]
    assert resumed["accepted_independent"] == 1
    assert resumed["generation_elapsed_s"] > first["generation_elapsed_s"]


def test_physical_motor_corpus_publishes_only_stable_reviewed_sources(monkeypatch, tmp_path):
    from smart_swarm import physical_motor_dataset as corpus
    from smart_swarm.physical_story_plan import prepare

    config = {"data_root": tmp_path, "studio_project": tmp_path / "studio"}
    first, failure = prepare(config)["cards"][:2]
    root = tmp_path / "physical-stories"
    export = config["studio_project"] / "runtime/smart-swarm-pilot-exports/fixture"
    meta = export / "provenance/export.json"
    meta.parent.mkdir(parents=True)
    atomic(meta, {"status": "complete", "dataset_sha256": "export", "repo_id": "native/export",
                  "source": "hardware", "test_only": False})
    parquet = export / "data/rows.parquet"
    parquet.parent.mkdir()
    parquet.write_bytes(b"original numeric fixture")
    for card in (first, failure):
        trial = root / card["story_id"]
        trial.mkdir(parents=True)
        atomic(trial / "trial.json", {"card": card, "studio_dataset_id": "ds_1234567890123456"})
    selected = {"scene_group_id": first["source_group_id"], "split": first["split"], "sample_count": 40,
                "outcome": "placement_succeeded", "native_frame_manifest_sha256": "original"}
    binding = {"path": str(export), "export_metadata_sha256": sha(meta), "dataset_sha256": "export",
               "repo_id": "native/export", "source_frame_manifest_sha256": "original", "source": "hardware",
               "technical_only": False, "camera_keys": list(corpus.CAMERA_KEYS),
               "live_calibration_snapshot_hashes": {"leader": "a", "follower": "b"},
               "export_storage_sha256": corpus.export_storage_digest(export)}
    good = root / first["story_id"]
    atomic(good / "motor-intake.json", {"status": "reviewable_prefix", "segment": "motor-segment.json"})
    atomic(good / "motor-segment.json", {**selected, "native_export": binding})
    atomic(root / failure["story_id"] / "motor-intake.json", {"status": "excluded", "reason": "observed failure"})
    monkeypatch.setattr(corpus, "segment_choice", lambda *_: (selected, None))
    result = corpus.publish(config)
    assert corpus.publish(config) == result
    assert result["prefix_count"] == 1 and result["frames_by_split"] == {"development": 40}
    assert result["corpus_rejections"][0]["reason"] == "observed failure"
    manifest = Path(result["manifest"])
    assert sha(manifest) == manifest.stem
    parquet.write_bytes(b"changed numeric fixture")
    assert corpus.publish(config)["prefix_count"] == 0
    assert sha(manifest) == manifest.stem


def test_motor_export_publishes_only_verified_native_lineages(monkeypatch, tmp_path):
    from scripts import learning_pilot as pilot
    from smart_swarm.episode import atomic
    from smart_swarm.motor_input import PROFILE

    source, output = tmp_path / "raw", tmp_path / "native"
    for split in ("train", "validation"):
        raw = source / split / split
        raw.mkdir(parents=True)
        atomic(raw / "manifest.json", {"sha256": split})

    def exporter(command, env, stream, deadline, heartbeat):
        directory = Path(command[command.index("--output") + 1])
        folder = directory / "train"
        folder.mkdir(parents=True)
        if len(list(tmp_path.glob("native-unpublished-*"))) == 1:
            return 1
        for split in ("train", "validation"):
            folder = directory / split
            folder.mkdir(exist_ok=True)
            row = {"source_world_episode_id": split, "source_sha256": split,
                   "raw_path": str(source / split / split)}
            atomic(folder / "smart-swarm-lineage.json", {"input_profile": PROFILE,
                   "episodes": [row], "episode_padding_mask_checked": True})
        return 0

    monkeypatch.setattr(pilot, "execute", exporter)
    with pytest.raises(RuntimeError, match="exporter failed"):
        pilot.official_export(source, output, "--aligned", tmp_path / "export.log", lambda: None, 2)
    assert not output.exists()
    pilot.official_export(source, output, "--aligned", tmp_path / "export.log", lambda: None, 2)
    assert pilot.verified_export(source, output, "--aligned")
    assert len(list(tmp_path.glob("native-unpublished-*"))) == 1


def test_unverified_physical_reveal_preserves_prediction_without_terminal_label(monkeypatch, tmp_path):
    from smart_swarm.episode import Recorder, atomic, rows
    from smart_swarm import physical_story_replay as story

    config = {"data_root": tmp_path}
    monkeypatch.setattr(story, "card_for", lambda cfg, card: None)
    agent = "so101-1"
    trial = story.story_workspace(config) / "unverified-trial"
    root = trial / "private-replay"
    clock = {"host_id": "homebrain", "boot_id": "unverified-test", "kind": "monotonic"}
    recorder = Recorder(root, "unverified-trial", "test-group", {agent: []},
                        {"run_mode": "hardware", "native_recording_clock": clock}, clock=clock)
    recorder.capture(agent, {}, 10, [], [])
    decision = {"query_all": {"decision": "request_inspection"}}
    recorder.append("decisions", "prediction", decision)
    state = {"card": {"family": "F6"}, "status": "pending_reveal", "decision": decision}
    atomic(trial / "trial.json", state)
    with pytest.raises(ValueError, match="confirmation"):
        story.terminal_reveal(config, "unverified-trial", "left", preserved=False)
    assert load(trial / "trial.json")["status"] == "pending_reveal"
    result = story.terminal_reveal(config, "unverified-trial", "relation_unverified", preserved=False)
    assert result["status"] == "incomplete" and result["observed_terminal_outcome"] == "unknown"
    assert rows(root / "decisions/prediction.jsonl") == [decision]
    assert not (root / "training_labels/settled-event.jsonl").exists()


def test_failed_fleet_capture_retains_admitted_native_arm_prefix(monkeypatch, tmp_path):
    from smart_swarm.episode import atomic
    from smart_swarm import physical_story_replay as story

    config = {"data_root": tmp_path}
    monkeypatch.setattr(story, "card_for", lambda cfg, card: None)
    idle_status = dict(recording_active=False, teleoperation_active=False, policy_active=False, learning_job_active=False)
    monkeypatch.setattr(story, "native_status", lambda cfg: idle_status)
    monkeypatch.setattr(story, "fleet_manifest", lambda cfg, capture: {"phase": "failed", "error": "Native camera timed out"})
    trial = story.story_workspace(config) / "lost-witness"
    trial.mkdir()
    state = {"status": "recording", "card": {"family": "F1"}, "fleet_capture_id": "recorded-fault",
             "markers": {"placement_closed": {}, "relocation_closed": {}}}
    atomic(trial / "trial.json", state)

    def prepare(cfg, path, journal, *, motor_only):
        assert motor_only
        journal.update(motor_segment_admitted=True, native_episode_id="accepted-arm-episode")
        return path, {}

    monkeypatch.setattr(story, "prepare_native_arm", prepare)
    result = story.predict_selected(config, "lost-witness")
    stored = load(trial / "trial.json")
    assert result == {"status": "incomplete", "motor_segment_admitted": True, "reason": "Native camera timed out"}
    assert stored["native_episode_id"] == "accepted-arm-episode"
    assert stored["observed_terminal_outcome"] == "unknown" and "decision" not in stored
