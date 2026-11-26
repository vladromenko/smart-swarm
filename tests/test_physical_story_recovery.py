from contextlib import contextmanager
from subprocess import TimeoutExpired
import time

import numpy as np
import pytest

from smart_swarm import physical_story_operator as operator
from smart_swarm import physical_story_motor as motor
from smart_swarm import physical_story_replay as replay
from smart_swarm import physical_reveal
from smart_swarm import physical_story_source as source
from smart_swarm.contract import interval, time_at
from smart_swarm.episode import DELIVERIES, TRUTH, Recorder, atomic, load
from smart_swarm.memory import QUERY, predict


def test_native_interruption_with_missing_markers_keeps_f3_insertion(monkeypatch, tmp_path):
    config = {"data_root": tmp_path}
    trial = source.story_workspace(config) / "interrupted-f3"
    trial.mkdir()
    state = {"status": "recording", "card": {"family": "F3"}, "fleet_capture_id": "a" * 32,
             "markers": {"insertion_closed": {"observed_placement_outcome": "placement_succeeded"}}}
    atomic(trial / "trial.json", state)
    monkeypatch.setattr(replay, "card_for", lambda *_: None)
    monkeypatch.setattr(replay, "native_status", lambda *_: {})
    monkeypatch.setattr(replay, "native_idle", lambda *_: None)
    monkeypatch.setattr(replay, "fleet_manifest", lambda *_: {"phase": "interrupted", "error": "Native clock expired"})

    def preserve(config, path, status):
        assert path == trial and set(status["markers"]) == {"insertion_closed"}
        status.update(native_episode_id="recorded-studio", motor_segment_admitted=True)
        atomic(path / "motor-intake.json", {"status": "reviewable_prefix"})

    monkeypatch.setattr(replay, "prepare_native_arm", preserve)
    result = replay.predict_selected(config, trial.name)
    assert result["motor_segment_admitted"] is True
    assert "Native clock expired" in result["reason"] and "relocation_closed" in result["reason"]
    stored = load(trial / "trial.json")
    assert stored["status"] == "incomplete" and "decision" not in stored
    assert stored["observed_terminal_outcome"] == "unknown"
    assert replay.predict_selected(config, trial.name) == result


def test_operator_resumes_acknowledged_open_card_before_new_card(monkeypatch, tmp_path):
    config = {"data_root": tmp_path}
    first, active = {"story_id": "first"}, {"story_id": "existing"}
    monkeypatch.setattr(operator, "prepare", lambda _: {"cards": [first, active]})
    trial = source.story_workspace(config) / active["story_id"]
    trial.mkdir()
    atomic(trial / "trial.json", {"status": "preparing", "markers": {}, "start": source.anchor()})
    atomic(trial / "fleet-start.json", {"id": "a" * 32})
    card, state = operator.next_story(config)
    assert card == active and state["status"] == "recording" and state["fleet_capture_id"] == "a" * 32
    assert load(trial / "trial.json") == state
    monkeypatch.setattr(operator, "fleet_manifest", lambda *_: {"phase": "failed"})
    assert operator.physical_steps(config, trial.name, state) == state


def test_unacknowledged_prepare_waits_for_matching_worker_and_existing_capture(monkeypatch, tmp_path):
    state = {"status": "preparing", "start": {"unix_ns": time.time_ns() - 20_000_000_000}}
    marker = {"pid": 42, "started_ticks": "1"}
    atomic(tmp_path / "fleet-launch.json", marker)
    monkeypatch.setattr(source, "worker_identity", lambda pid: marker)
    assert source.restore_start({}, tmp_path, state)["status"] == "preparing"
    monkeypatch.setattr(source, "worker_identity", lambda pid: {"pid": pid, "started_ticks": "2"})
    monkeypatch.setattr(source, "data", lambda *_: {"active": "b" * 32})
    assert source.restore_start({}, tmp_path, state)["status"] == "preparing"
    monkeypatch.setattr(source, "data", lambda *_: {"active": None})
    assert source.restore_start({}, tmp_path, state)["status"] == "incomplete"
    assert load(tmp_path / "trial.json")["observed_terminal_outcome"] == "unknown"


def test_fleet_manifest_recognizes_native_interruption_without_mutating_sources(monkeypatch):
    identifier = "a" * 32
    active = {"id": identifier, "phase": "recording"}
    interrupted = {"id": identifier, "phase": "interrupted", "error": "Native capture exited"}
    unrelated = {"id": "b" * 32, "phase": "complete"}
    responses = {f"/api/fleet/captures/{identifier}/manifest.json": active,
                 "/api/fleet/captures": {"records": [unrelated, interrupted]}}
    monkeypatch.setattr(source, "data", lambda config, path: responses[path])
    assert source.fleet_manifest({}, identifier) == interrupted
    assert active["phase"] == "recording"


def test_late_arm_review_prefers_recording_inside_original_story_interval(monkeypatch):
    state = {"studio_dataset_id": "native", "preexisting_native_episode_ids": ["before"],
             "start": {"monotonic_ns": 10_000_000_000},
             "markers": {"insertion_closed": {"monotonic_ns": 20_000_000_000}}}

    def episode(name, first, last):
        return {"id": name, "storage": "episodes",
                "metadata": {"first_timestamp_ns": first * 10**9, "last_timestamp_ns": last * 10**9}}

    records = {row["id"]: row for row in (episode("before", 11, 20), episode("intended", 12, 19),
                                         episode("later", 21, 28))}
    monkeypatch.setattr(source, "episode_ids", lambda *_: records)
    assert source.corresponding_native_episode({}, state, boundary="insertion_closed")["id"] == "intended"
    records["also-inside"] = episode("also-inside", 13, 20)
    with pytest.raises(ValueError, match="Exactly one"):
        source.corresponding_native_episode({}, state, boundary="insertion_closed")


def test_terminal_preview_rejects_non_camera_only_or_stale_feed():
    preview = {"active": True, "camera_only": True, "torque_enabled": False,
               "cameras": {"top": {"connected": True, "last_frame_age_ms": 80., "frames_received": 20}}}
    status = {"preview": preview, "recording": {"active": False}, "teleoperation": {"active": False}}
    assert physical_reveal.preview_summary(status) == {"top_frame_sequence": 20, "status_frame_age_ms": 80.}
    preview["cameras"]["top"]["last_frame_age_ms"] = 700.
    with pytest.raises(replay.ProtocolError, match="native camera-only"):
        physical_reveal.preview_summary(status)
    preview["cameras"]["top"]["last_frame_age_ms"] = 80.
    preview["torque_enabled"] = True
    with pytest.raises(replay.ProtocolError, match="native camera-only"):
        physical_reveal.preview_summary(status)


def test_reveal_resumes_from_native_manifest_without_new_observation(monkeypatch, tmp_path):
    config = {"data_root": tmp_path}
    trial = source.story_workspace(config) / "reveal"
    root = trial / "private-replay"
    clock = {"host_id": "homebrain", "boot_id": "fixture", "kind": "monotonic"}
    recorder = Recorder(root, trial.name, "group", {"rosmaster-1": []},
                        {"run_mode": "hardware", "native_recording_clock": clock}, clock=clock)
    recorder.capture("rosmaster-1", {}, 5, [], [])
    decision = {"at": time_at(15, clock), "query_all": {"decision": "request_inspection"}}
    recorder.append("decisions", "prediction", decision)
    image = recorder.frame(TRUTH, "evaluator/top", 1, np.zeros((4, 4, 3), dtype=np.uint8), 20)
    recorder.append(TRUTH, "reveal", {"camera": image, "ground_truth": "surface", "observed_at": time_at(20, clock),
                                      "target_state_time_or_interval": interval(15, 20, clock),
                                      "relations_preserved_until_reveal": True})
    recorder.close(10, 15, 20)
    atomic(trial / "trial.json", {"status": "pending_reveal", "decision": decision, "query_ns": 10})
    monkeypatch.setattr(replay, "card_for", lambda *_: None)
    monkeypatch.setattr(replay, "capture_clip", lambda *_: pytest.fail("A sealed reveal needs no new image"))
    result = replay.terminal_reveal(config, trial.name, "surface", True)
    assert result["status"] == "complete" and result["recorded_at_ns"] == 20
    with pytest.raises(ValueError, match="already has a sealed"):
        replay.terminal_reveal(config, trial.name, "left", True)


def reveal_sample(index):
    low = 1_000_000_000 + index * 1_030_000_000
    status = {"top_frame_sequence": 20 + index * 30, "status_frame_age_ms": 80.,
              "native_stop_generation": 1, "stream": {"width": 4, "height": 4, "device": "/dev/top",
              "device_resolved": "/dev/video2", "timestamp_basis": "monotonic"}}
    proof = {"earliest_possible_capture_ns": str(low), "request_started_ns": str(low + 1_000_000_000),
             "received_ns": str(low + 1_020_000_000), "preview_status_before": status,
             "preview_status_after": {**status, "top_frame_sequence": status["top_frame_sequence"] + 1},
             "native_jpeg_sha256": "a" * 64}
    return np.zeros((4, 4, 3), dtype=np.uint8), proof


def test_new_reveal_requires_full_clip_and_checks_first_pixels(tmp_path):
    clock = {"host_id": "homebrain", "boot_id": "fixture", "kind": "monotonic"}
    root = tmp_path / "replay"
    meta = {"run_mode": "hardware", physical_reveal.FIELD: physical_reveal.PROFILE, "native_recording_clock": clock}
    recorder = Recorder(root, "fixture", "group", {"rosmaster-1": []}, meta, clock=clock)
    recorder.capture("rosmaster-1", {}, 5, [], [])
    samples = [reveal_sample(index) for index in range(3)]
    with pytest.raises(ValueError, match="incomplete"):
        physical_reveal.store_clip(recorder, samples[:2], 15, "unknown")
    assert not list((root / TRUTH).iterdir())
    observed = physical_reveal.store_clip(recorder, samples, 15, "unknown")
    recorder.close(10, 15, observed)
    manifest = load(root / "manifest.json")
    row = load(root / TRUTH / "reveal.jsonl")
    assert physical_reveal.validate_sealed_clip(root, row, manifest) == 3
    legacy = {key: value for key, value in row.items() if key != "reveal_clip"}
    with pytest.raises(ValueError, match="complete three-sample clip"):
        physical_reveal.validate_sealed_clip(root, legacy, manifest)
    first = row["reveal_clip"]["frames"][0]["camera"]
    (root / first["frame_ref"]).write_bytes(b"tampered first PNG")
    with pytest.raises(ValueError, match="Frame content does not match hash"):
        physical_reveal.validate_sealed_clip(root, row, manifest)


def test_reveal_rejects_stop_change_and_overlapping_frame_window():
    _, first = reveal_sample(0)
    _, second = reveal_sample(1)
    assert physical_reveal.validate_sample(second, first, 15) == int(second["received_ns"])
    with pytest.raises(ValueError, match="overlap"):
        physical_reveal.validate_sample(first, first, 15)
    second["preview_status_after"]["native_stop_generation"] += 1
    with pytest.raises(ValueError, match="same stream"):
        physical_reveal.validate_sample(second, first, 15)


def test_query_reopens_committed_prediction_when_live_sources_are_unavailable(monkeypatch, tmp_path):
    config = {"data_root": tmp_path}
    trial = source.story_workspace(config) / "projection"
    root = trial / "private-replay"
    capture = "a" * 32
    clock = {"host_id": "homebrain", "boot_id": "fixture", "kind": "monotonic"}
    camera_grants = {agent: [] for agent in replay.AGENTS}
    metadata = {"run_mode": "hardware", "native_recording_clock": clock,
                "fleet_capture": capture, "studio_episode": "native-arm"}
    recorder = Recorder(root, trial.name, "group", camera_grants, metadata, clock=clock)
    for agent in replay.AGENTS:
        recorder.capture(agent, {}, 5, [], [])
    recorder.append("communication", "queries", {"issued_time": time_at(10, clock), "text": QUERY})
    (root / DELIVERIES).touch()
    decision = {"at": time_at(15, clock), "agent_id": replay.AGENTS[2], "query_all": predict([], 15)}
    recorder.append("decisions", "prediction", decision)
    atomic(trial / "motor-intake.json", {"status": "reviewable_prefix"})
    atomic(trial / "trial.json", {"status": "recording", "fleet_capture_id": capture,
                                 "markers": {"relocation_closed": {"monotonic_ns": 7}}})
    monkeypatch.setattr(replay, "card_for", lambda *_: None)
    monkeypatch.setattr(replay, "native_status", lambda *_: pytest.fail("A saved prediction needs no live source"))
    assert replay.predict_selected(config, trial.name) == decision
    saved = load(trial / "trial.json")
    assert saved["status"] == "pending_reveal" and saved["query_ns"] == 10
    assert saved["motor_segment_admitted"] and saved["native_episode_id"] == "native-arm"


def test_story_prediction_commits_while_the_official_motor_export_is_pending(monkeypatch, tmp_path):
    state = {"card": {"family": "F1"}, "studio_dataset_id": "ds_1234567890123456",
             "status": "recording", "markers": {"placement_closed": {}}}
    row = {"id": "ep_1", "metadata": {"frames_sha256": "original"}}
    proof = {"episode_id": row["id"], "dataset_id": state["studio_dataset_id"], "frames_sha256": "original"}
    monkeypatch.setattr(replay, "corresponding_native_episode", lambda *_, **__: row)
    monkeypatch.setattr(replay, "validate_story_episode", lambda *_: proof)
    monkeypatch.setattr(replay, "segment_choice", lambda *_: ({"native_frame_manifest_sha256": "original"}, None))
    decision = {"query_all": {"decision": "inspect"}}

    def project(config, trial, current, through, fleet):
        assert current["native_story_validation"] == proof
        assert load(trial / "motor-intake.json")["status"] == "pending_review"
        return decision

    monkeypatch.setattr(replay, "admit_sources", project)
    monkeypatch.setattr(motor, "accepted_export", lambda *_: pytest.fail("The reveal waits on a whole-dataset export"))
    assert replay.project_ready({"studio_project": tmp_path}, tmp_path, state, {}) == decision
    stored = load(tmp_path / "trial.json")
    assert stored["status"] == "pending_reveal" and stored["decision"] == decision
    assert not motor.review_pending_motor({"data_root": tmp_path})["attempted"]


@contextmanager
def admitted_motor(*_):
    yield {}, lambda: None


def queued_cards(config, dataset):
    root = source.story_workspace(config)
    for name, status in (("done1", "complete"), ("done2", "incomplete"), ("waiting", "pending_reveal")):
        trial = root / name
        trial.mkdir()
        atomic(trial / "trial.json", {"status": status, "studio_dataset_id": dataset,
                                      "native_story_validation": {"frames_sha256": name}, "decision": {"saved": True}})
        atomic(trial / "motor-intake.json", {"status": "pending_review"})
    return root


def test_motor_batch_preserves_reveal_and_retries_timeout_with_one_export(monkeypatch, tmp_path):
    config = {"data_root": tmp_path, "studio_project": tmp_path / "native"}
    dataset = "ds_1234567890123456"
    root = queued_cards(config, dataset)
    monkeypatch.setattr(motor, "admitted", admitted_motor)
    attempts, receipts = [], []

    def export(config, raw, cache):
        assert cache.is_dir() and raw.name == cache.name == dataset
        attempts.append(raw)
        if len(attempts) == 1:
            raise TimeoutExpired("native exporter", 180)
        return tmp_path / "official", {}

    def save(config, trial, state, destination, *, storage_digest):
        receipts.append((trial.name, destination.name, storage_digest))
        atomic(trial / "motor-intake.json", {"status": "reviewable_prefix", "segment": "motor-segment.json"})
        return True

    monkeypatch.setattr(motor, "accepted_export", export)
    monkeypatch.setattr(motor, "export_storage_digest", lambda _: "frozen")
    monkeypatch.setattr(motor, "save_prefix", save)
    first = motor.review_pending_motor(config)
    assert first["attempted"] == 2 and dataset in first["errors"] and len(attempts) == 1 and not receipts
    assert all(item["status"] == "pending_review" for item in first["states"].values())
    second = motor.review_pending_motor(config)
    assert not second["errors"] and len(attempts) == 2
    assert receipts == [("done1", "official", "frozen"), ("done2", "official", "frozen")]
    assert load(root / "done1/trial.json")["decision"] == {"saved": True}
    assert load(root / "waiting/trial.json")["status"] == "pending_reveal"
    assert not motor.review_pending_motor(config)["attempted"]


def native_clock_row(stamp, boot="original", uncertainty_ms=7.):
    info = {"boot_id": boot, "clock_kind": "midpoint_estimate_not_hardware_sync", "uncertainty_ms": uncertainty_ms}
    return {"updated_at": stamp, "running": True, "read_only": True, "motion_commands_sent": 0, "robots": {"tank": info}}


def test_native_phase_rejects_timestamp_overlap_and_changed_remote_boot(tmp_path):
    from smart_swarm.physical_fleet_timing import FleetClock, PhysicalStoryRecorder, eligible_frames

    clock = FleetClock("tank", native_clock_row(1.), [native_clock_row(1.05), native_clock_row(1.15)])
    sample = {"source_unix_ns": 1_070_000_000, "received_unix_ns": 1_100_000_000, "after_s": .1, "age_s": .1}
    phase = (50_000_000, 110_000_000)
    admitted, audit = eligible_frames([sample], clock, 1_000_000_000, phase)
    assert not admitted and audit["excluded_reasons"] == {"full RGB interval and receipt must fit the permitted phase": 1}
    admitted, audit = eligible_frames([sample, sample], clock, 1_000_000_000, (1, 120_000_000))
    assert len(admitted) == 1 and audit["excluded_reasons"] == {"repeated native camera source timestamp": 1}
    robot = "tank-1"
    sensor = robot + "/front"
    mapped_clock = {"host_id": "homebrain", "boot_id": "fixture", "kind": "monotonic"}
    recorder = PhysicalStoryRecorder(tmp_path / "story", "story", "group", {robot: [sensor]},
                                     {"run_mode": "hardware"}, clock=mapped_clock)
    packet = recorder.fleet_packet(robot, sensor, np.zeros((2, 2, 3), dtype=np.uint8), admitted[0])
    assert packet["cameras"][0]["capture_interval"] == interval(38_000_000, 100_000_000, mapped_clock)
    assert packet["cameras"][0]["exposure_time"] is None
    next_boot = FleetClock("tank", native_clock_row(1.), [native_clock_row(1.05), native_clock_row(1.15, boot="different")])
    rejected, detail = eligible_frames([sample], next_boot, 1_000_000_000, (1, 120_000_000))
    assert not rejected and detail["excluded_reasons"] == {"native clock boot changed during the recording": 1}


def test_physical_steps_preserve_recording_from_an_older_homebrain_boot(monkeypatch):
    state = {"status": "recording", "homebrain_boot_id": "earlier-boot", "markers": {}}
    monkeypatch.setattr(source, "homebrain_boot", lambda: "current-boot")
    monkeypatch.setattr(operator, "is_capturing", lambda *_: pytest.fail("The old recording cannot prompt for motion"))
    assert operator.physical_steps({}, "existing", state) is state
