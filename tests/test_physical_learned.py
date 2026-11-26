import numpy as np
import pytest

from smart_swarm import physical_learned, physical_story_replay
from smart_swarm.contract import ProtocolError, time_at
from smart_swarm.episode import LABELS, POLICY, TRUTH, Recorder, load, rows
from smart_swarm.evidence_inputs import AGENTS


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
    altered = {**manifest, "private_card_metadata": "new sentinel"}
    with physical_learned.scoped_window(source, altered) as (scoped, digest):
        assert digest == original
        assert {file.name for file in scoped.iterdir()} == {POLICY, "manifest.json"}
        visible = load(scoped / "manifest.json")
        assert set(visible) == {"policy_cameras", "query_boundary", "decision_boundary", "world_episode_id"}
        assert visible["world_episode_id"] == "world"


def test_native_scoped_window_refuses_state_in_withheld_arm(tmp_path):
    recorder, manifest, _, _ = create_recording(tmp_path, measured=True)
    with pytest.raises(ValueError, match="state or commands in a withheld view"):
        with physical_learned.scoped_window(recorder.root, manifest):
            pytest.fail("The withheld arm's state was exposed")


def test_rule_decision_preserves_a_failed_native_reservation(tmp_path, monkeypatch):
    recorder, _, arm, tank = create_recording(tmp_path)

    def busy(*_):
        raise ProtocolError("RESOURCE_BUSY", "The native lease is already held")

    monkeypatch.setattr(physical_learned, "admitted", busy)
    monkeypatch.setattr(physical_story_replay, "anchor", lambda: {"monotonic_ns": 20})
    decision = physical_story_replay.persist_decision({}, recorder, arm, tank, 16)
    assert decision["learned_evidence"] == {"status": "unavailable", "reason": "RESOURCE_BUSY: The native lease is already held"}
    assert decision["prediction_source"] == "deterministic local RGB rules and finite joint filter"
    assert rows(recorder.root / "decisions/prediction.jsonl") == [decision]
    assert list((recorder.root / TRUTH).iterdir()) == []
