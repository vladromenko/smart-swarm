"""Regression gates for the resumed software and experiment provenance."""

from pathlib import Path
from subprocess import TimeoutExpired

import pytest

from smart_swarm.cli import settings
from smart_swarm.evidence_inputs import audit_b1
from smart_swarm.episode import load
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


def test_attempt_ids_survive_quarantine_and_late_polling(tmp_path):
    entries = [{"id": "ep_second"}, {"id": "ep_first"}]
    original = assign_attempts(tmp_path, entries)
    extended = assign_attempts(tmp_path, [{"id": "ep_inline"}, *reversed(entries)])
    assert original == {"ep_first": 0, "ep_second": 1}
    assert extended == {**original, "ep_inline": 2}


def test_native_consumer_timeout_resumes_completed_export(monkeypatch, tmp_path):
    from smart_swarm import studio

    source = tmp_path / "native"
    source.mkdir()
    (source / "dataset.json").write_text("{}")
    export = tmp_path / "export-v1"
    export.mkdir()
    calls = []

    def generate(*_):
        calls.append("exported")
        return export, {"status": "complete"}

    def consume(config, raw, target):
        assert raw == source and target == export
        calls.append("consumed")
        if calls.count("consumed") == 1:
            raise TimeoutExpired("consume", 160)
        return {"validated": True}

    monkeypatch.setattr(studio, "export_native", generate)
    monkeypatch.setattr(studio, "consume_native", consume)
    with pytest.raises(TimeoutExpired):
        studio.accepted_export({}, source, tmp_path)
    assert (tmp_path / "completed-native-export.json").is_file()
    assert not (tmp_path / "accepted-native-export.json").exists()
    assert studio.accepted_export({}, source, tmp_path) == (export, {"validated": True})
    assert calls == ["exported", "consumed", "consumed"]


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


def test_pilot_outcome_prompt_selects_next_completed_attempt(monkeypatch, tmp_path):
    from smart_swarm import studio
    from smart_swarm.episode import atomic

    root = tmp_path / "so101-pilot"
    root.mkdir()
    entries = [{"id": "ep_1", "storage": "quarantine", "metadata": {"status": "incomplete"}}]
    entries.extend({"id": f"ep_{index}", "storage": "episodes", "metadata": {"status": "complete"}}
                   for index in range(2, 6))
    monkeypatch.setattr(studio, "native_request", lambda cfg, uri: list(reversed(entries)))
    monkeypatch.setattr(studio, "prepare", lambda cfg: {"attempt_limit": 4})
    labels = root / "observed-outcomes.json"
    atomic(labels, {"ep_2": {"outcome": "placement_failed"}})
    config, binding = {"data_root": tmp_path}, {"dataset_id": "ds_example"}
    assert studio.pending_outcome(config, binding) == {"episode_id": "ep_3", "attempt": 3}
    atomic(labels, {name: {"outcome": "unknown"} for name in ("ep_2", "ep_3", "ep_4")})
    assert studio.pending_outcome(config, binding) is None


def test_complete_native_failures_and_quarantine_never_enter_motor_review():
    complete = {"id": "native1", "storage": "episodes", "metadata": {"status": "complete", "frames": 3}}
    incomplete = {**complete, "storage": "quarantine", "metadata": {"status": "incomplete", "reason": "stop"}}
    card = {"attempt_limit": 1, "attempts": [{"source_world_episode_id": "w"}], "scene_group_id": "g"}
    label = {"outcome": "placement_failed"}
    assert status_for(complete, label, card, 0, {"valid": True})["classification"] == "story_failure"
    assert status_for(incomplete, None, card, 0, None)["reasons"] == ["stop"]
    assert status_for(complete, None, card, 1, None)["classification"] == "rejected"
    manifest = {"run_mode": "fixture", "recording_integrity": "complete",
                "validation_status": "valid", "task_outcome_label": "placement_succeeded"}
    assert classify(manifest, {"valid": True})[0] == "fixture_only"


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


def test_story_preflight_checks_native_rgb_without_general_sensor_dependencies():
    from smart_swarm.fleet_adapter import rgb_readiness

    live = {"running": True, "read_only": True, "robots": {
        role: {"ages_s": {"rgb": 0.1}, "uncertainty_ms": 24.0} for role in ("tank", "rosmaster")}}
    assert rgb_readiness(live)["ready"]
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


def test_distributed_start_refuses_unconfirmed_native_stop_before_recorder_check(monkeypatch):
    from smart_swarm import physical_story_source as story
    from smart_swarm.contract import ProtocolError

    state = {"inventory_present": {"wrist": True, "top": True, "leader": True, "follower": True},
             "recording_active": False, "teleoperation_active": False, "policy_active": False,
             "learning_job_active": False, "native_stop_check_required": True, "native_arm_hardware_open": False}
    team = {"so101_studio": state, "story_observation_readiness": {"ready": True, "issues": []},
            "active_native_capture": "existing"}
    monkeypatch.setattr(story, "native_team_status", lambda _, **kw: team)
    monkeypatch.setattr(story.subprocess, "run", lambda *_, **kw: pytest.fail("Native lease should stay untouched"))
    report = story.story_readiness({})
    assert not report["ready_for_native_lease_check"]
    assert [issue["code"] for issue in report["blocking_issues"]] == ["OFFLINE", "RESOURCE_BUSY"]
    with pytest.raises(ProtocolError, match="hardware STOP"):
        story.story_preflight({})


def test_story_preflight_separates_busy_native_reservation_from_helper_failure(monkeypatch):
    from subprocess import CompletedProcess, TimeoutExpired

    from smart_swarm import physical_story_source as story
    from smart_swarm.contract import ProtocolError

    monkeypatch.setattr(story, "story_readiness", lambda _: {"blocking_issues": []})
    monkeypatch.setattr(story, "studio_env", lambda _: {})
    config = {"studio_python": "native-python"}
    for result, code in ((75, "RESOURCE_BUSY"), (1, "OFFLINE")):
        monkeypatch.setattr(story.subprocess, "run", lambda *_, **kw: CompletedProcess([], result, stderr="helper error"))
        with pytest.raises(ProtocolError) as error:
            story.story_preflight(config)
        assert error.value.code == code

    def timeout(*args, **kwargs):
        raise TimeoutExpired(args[0], kwargs["timeout"])

    monkeypatch.setattr(story.subprocess, "run", timeout)
    with pytest.raises(ProtocolError, match="reservation did not answer"):
        story.story_preflight(config)


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


def test_compact_comparison_import_allows_egl_configuration_before_mujoco():
    import os
    import subprocess
    import sys

    code = '''
import sys
import run_compact_motor_comparison as runner

assert "mujoco" not in sys.modules
runner.configure_renderer(runner.settings())
import smart_swarm.motor_full
from mujoco.rendering.classic.gl_context import GLContext

assert GLContext.__module__ == "mujoco.egl"
'''
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src") + os.pathsep + str(ROOT / "scripts")}
    subprocess.run([sys.executable, "-B", "-c", code], cwd=ROOT, env=env, check=True, timeout=20)


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
    from smart_swarm.episode import atomic, sha
    from smart_swarm.physical_story_plan import prepare

    studio = tmp_path / "studio"
    config = {"data_root": tmp_path, "studio_project": studio}
    first, failure = prepare(config)["cards"][:2]
    root = tmp_path / "physical-stories"
    export = studio / "runtime/smart-swarm-pilot-exports/fixture"
    meta = export / "provenance/export.json"
    meta.parent.mkdir(parents=True)
    atomic(meta, {"status": "complete", "dataset_sha256": "export", "repo_id": "native/export",
                  "source": "hardware", "test_only": False})
    for card in (first, failure):
        trial = root / card["story_id"]
        trial.mkdir(parents=True)
        atomic(trial / "trial.json", {"card": card})
    selected = {"scene_group_id": first["source_group_id"], "split": first["split"], "sample_count": 40,
                "outcome": "placement_succeeded", "native_frame_manifest_sha256": "original"}
    binding = {"path": str(export), "export_metadata_sha256": sha(meta), "dataset_sha256": "export",
               "repo_id": "native/export", "source_frame_manifest_sha256": "original", "source": "hardware",
               "technical_only": False, "camera_keys": list(corpus.CAMERA_KEYS),
               "live_calibration_snapshot_hashes": {"leader": "a", "follower": "b"}}
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
    atomic(meta, {"source": "synthetic_test"})
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
    monkeypatch.setattr(story, "native_status", lambda cfg: {})
    monkeypatch.setattr(story, "native_idle", lambda status: None)
    monkeypatch.setattr(story, "fleet_manifest", lambda cfg, capture: {"phase": "failed", "error": "Native camera timed out"})
    trial = story.story_workspace(config) / "lost-witness"
    trial.mkdir()
    state = {"status": "recording", "card": {"family": "F1"}, "fleet_capture_id": "recorded-fault",
             "markers": {"placement_closed": {}, "relocation_closed": {}}}
    atomic(trial / "trial.json", state)

    def prepare(cfg, path, journal):
        journal.update(motor_segment_admitted=True, native_episode_id="accepted-arm-episode")
        return path, {}

    monkeypatch.setattr(story, "prepare_native_arm", prepare)
    result = story.predict_selected(config, "lost-witness")
    stored = load(trial / "trial.json")
    assert result == {"status": "incomplete", "motor_segment_admitted": True, "reason": "Native camera timed out"}
    assert stored["native_episode_id"] == "accepted-arm-episode"
    assert stored["observed_terminal_outcome"] == "unknown" and "decision" not in stored
