from contextlib import contextmanager

import numpy as np
import pytest

from smart_swarm import physical_learned, physical_offline, physical_story_replay
from smart_swarm.contract import ProtocolError, time_at
from smart_swarm.episode import LABELS, POLICY, TRUTH, Recorder, atomic, load, rows
from smart_swarm.evidence_inputs import AGENTS
from smart_swarm.primary_inputs import json_lines


def create_recording(tmp_path, *, measured=False):
    clock = {"host_id": "homebrain", "boot_id": "fixture", "kind": "monotonic"}
    cameras = {AGENTS[0]: [], **{agent: [agent + "/front"] for agent in AGENTS[1:]}}
    recorder = Recorder(tmp_path / "source", "example", "development", cameras,
                        {"run_mode": "hardware", "private_card_metadata": "never projected"}, clock=clock)
    arm = recorder.capture(AGENTS[0], {}, 5, [3] if measured else [], ["degrees"] if measured else [])
    frame = np.zeros((12, 16, 3), dtype=np.uint8)
    tank = [(recorder.capture(AGENTS[1], {AGENTS[1] + "/front": frame}, tick, [], []), frame) for tick in range(7, 15)]
    recorder.capture(AGENTS[2], {AGENTS[2] + "/front": frame}, 15, [], [])
    manifest = {**recorder.meta, "sample_counts": recorder.counts, "query_boundary": time_at(16, recorder.clock)}
    return recorder, manifest, [(arm, {})], tank


def test_native_scoped_window_withholds_cards_truth_and_labels(tmp_path):
    recorder, manifest, _, _ = create_recording(tmp_path)
    source = recorder.root
    with physical_learned.scoped_window(source, manifest) as (_, original):
        pass
    recorder.append(TRUTH, "injected", {"terminal": "right"})
    recorder.append(LABELS, "injected", {"placement": "forbidden"})
    arm_path = source / POLICY / f"{AGENTS[0]}.jsonl"
    arm, = rows(arm_path)
    json_lines(arm_path, [{**arm, "hidden_private_mark": "left"}])
    altered = {**manifest, "private_card_metadata": "new sentinel"}
    with physical_learned.scoped_window(source, altered) as (scoped, digest):
        assert digest == original
        assert {file.name for file in scoped.iterdir()} == {POLICY, "manifest.json"}
        visible = load(scoped / "manifest.json")
        assert set(visible) == {"policy_cameras", "query_boundary", "decision_boundary", "world_episode_id"}
        assert visible["world_episode_id"] == "world"
        assert "hidden_private_mark" not in rows(scoped / POLICY / arm_path.name)[0]


def test_native_scope_digest_includes_effective_policy_deadline(tmp_path):
    recorder, manifest, _, _ = create_recording(tmp_path)
    with physical_learned.scoped_window(recorder.root, manifest) as (scoped, original):
        configured = load(scoped / "manifest.json")
        stream = physical_learned.packets(scoped, configured)
        changed = {**configured, "decision_boundary": time_at(16 + 1_000_000, recorder.clock)}
        assert physical_learned.window_digest(configured, stream, projected=True) == original
        assert physical_learned.window_digest(changed, stream, projected=True) != original
        assert (physical_learned.window_digest(changed, stream, legacy=True)
                == physical_learned.window_digest(configured, stream, legacy=True))


def test_withheld_native_arm_does_not_transfer_a_raw_frame_timestamp(tmp_path):
    source = tmp_path / "native"
    source.mkdir()
    state = {"card": {"family": "F6"}, "start": {"monotonic_ns": 10},
             "markers": {"placement_closed": {"monotonic_ns": 100}}}
    packets = []
    for outcome, acquired in (("left", 20), ("surface", 30)):
        json_lines(source / "frames.jsonl", [{"observation_timestamp_ns": acquired, "hidden_outcome": outcome}])
        recorder = Recorder(tmp_path / outcome, "example", "development", {AGENTS[0]: []}, {"run_mode": "hardware"})
        (packet, views), = physical_story_replay.studio_clips(recorder, source, state)
        assert views == {}
        assert int(packet["source_time"]["monotonic_ns"]) == state["start"]["monotonic_ns"]
        packets.append(packet)
    assert packets[0] == packets[1]


def test_native_scoped_window_refuses_state_in_withheld_arm(tmp_path):
    recorder, manifest, _, _ = create_recording(tmp_path, measured=True)
    with pytest.raises(ValueError, match="state or commands in a withheld view"):
        with physical_learned.scoped_window(recorder.root, manifest):
            pytest.fail("The withheld arm's state was exposed")


def test_rule_decision_preserves_a_failed_native_reservation(tmp_path, monkeypatch):
    recorder, manifest, arm, tank = create_recording(tmp_path)

    def busy(*_):
        raise ProtocolError("RESOURCE_BUSY", "The native lease is already held")

    monkeypatch.setattr(physical_learned, "admitted", busy)
    monkeypatch.setattr(physical_story_replay, "anchor", lambda: {"monotonic_ns": 20})
    decision = physical_story_replay.persist_decision({}, recorder, arm, tank, 16)
    deferred = decision["learned_evidence"]
    assert deferred["status"] == physical_learned.DEFERRED
    assert deferred["online_attempt"] == {"status": "unavailable", "stage": "compute_admission",
                                          "reason": "RESOURCE_BUSY: The native lease is already held"}
    assert physical_learned.verify_optional(recorder.root, manifest, decision) == deferred
    assert decision["prediction_source"] == "deterministic local RGB rules and finite joint filter"
    assert rows(recorder.root / "decisions/prediction.jsonl") == [decision]
    assert list((recorder.root / TRUTH).iterdir()) == []


def test_busy_admission_does_not_defer_an_invalid_private_window(tmp_path, monkeypatch):
    recorder, _, arm, tank = create_recording(tmp_path, measured=True)

    def busy(*_):
        raise ProtocolError("RESOURCE_BUSY", "The native lease is already held")

    monkeypatch.setattr(physical_learned, "admitted", busy)
    monkeypatch.setattr(physical_story_replay, "anchor", lambda: {"monotonic_ns": 20})
    decision = physical_story_replay.persist_decision({}, recorder, arm, tank, 16)
    assert decision["learned_evidence"]["status"] == "unavailable"
    assert "Blind offline input seal failed" in decision["learned_evidence"]["reason"]
    assert "state or commands in a withheld view" in decision["learned_evidence"]["reason"]


def test_admitted_worker_failure_keeps_the_unavailable_online_result(tmp_path, monkeypatch):
    recorder, manifest, _, _ = create_recording(tmp_path)

    @contextmanager
    def available(*_):
        yield {}, lambda: None

    def damaged_model(*_):
        raise ValueError("The original checkpoint digest differs")

    monkeypatch.setattr(physical_learned, "admitted", available)
    monkeypatch.setattr(physical_learned, "run_worker", damaged_model)
    result = physical_learned.inference({}, recorder, int(manifest["query_boundary"]["monotonic_ns"]))
    assert result == {"status": "unavailable", "reason": "The original checkpoint digest differs"}


def test_offline_seal_rejects_changed_rgb_and_ignores_evaluator_truth(tmp_path, monkeypatch):
    recorder, manifest, arm, tank = create_recording(tmp_path)
    monkeypatch.setattr(physical_story_replay, "anchor", lambda: {"monotonic_ns": 20})
    monkeypatch.setattr(physical_learned, "admitted", lambda *_: pytest.fail("Sealing invoked inference"))
    decision = physical_story_replay.persist_decision({}, recorder, arm, tank, 16, evaluation_mode="blind_offline")
    assert decision["learned_evidence"]["status"] == physical_learned.DEFERRED
    recorder.append(TRUTH, "injected", {"terminal": "right"})
    assert physical_learned.verify_optional(recorder.root, manifest, decision) == decision["learned_evidence"]
    frame = next((recorder.root / POLICY).glob("*.png"))
    frame.write_bytes(b"corrupted camera frame")
    with pytest.raises(ValueError, match="Frame content does not match hash"):
        physical_learned.verify_optional(recorder.root, manifest, decision)


def test_automatic_offline_forecast_waits_for_completed_story(tmp_path, monkeypatch):
    config = {"data_root": tmp_path}
    trial = tmp_path / "physical-stories" / "card"
    trial.mkdir(parents=True)
    state = {"status": "pending_reveal", "decision": {"learned_evidence": {"status": physical_learned.DEFERRED}}}
    journal = trial / "trial.json"
    atomic(journal, state)
    calls = []
    monkeypatch.setattr(physical_offline, "score_trial", lambda _, target: calls.append(target))
    attempted = {}
    assert physical_offline.observe_pending(config, attempted) is None and not calls
    state["status"] = "complete"
    atomic(journal, state)
    assert physical_offline.observe_pending(config, attempted) == "card" and calls == [trial]
    assert physical_offline.observe_pending(config, attempted) is None and calls == [trial]
