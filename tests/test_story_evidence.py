from pathlib import Path
import shutil

import numpy as np
import pytest

from smart_swarm.contract import time_at
from smart_swarm.datasets import audit_lineage
from smart_swarm.episode import TRUTH, Replay, json_bytes, load, rows, validate
from smart_swarm.vision import center

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "evidence/worlds"


def evidence_worlds():
    return [(path.parent, load(path)) for path in sorted(EVIDENCE.glob("*/manifest.json"))]


@pytest.mark.skipif(not EVIDENCE.is_dir(), reason="Run the B1 suite to verify captured physics")
def test_shared_story_contact_checks_causal_views_and_lineage():
    worlds = evidence_worlds()
    assert len(worlds) == 4
    assert audit_lineage([manifest for _, manifest in worlds])["scene_groups"] == 1
    for root, manifest in worlds:
        validate(root, manifest)
        check = manifest["physics_check"]
        truth = rows(root / TRUTH / "reveal.jsonl")[0]
        client = Replay(root, "rosmaster-1")
        assert client.predict(client.query)["accepted_evidence_ids"] == []
        assert client.predict(client.query + 50_000_000)["decision"] != "left"
        assert check["physics_steps"] >= 40_000 and check["box_displacement_m"] > .12
        assert manifest["snapshot_check"]["max_qpos_delta"] <= 1e-10
        assert manifest["tank_travel_m"] > .02
        assert check["bilateral_grasp"] == check["contained_after_relocation"]
        if check["bilateral_grasp"]:
            assert check["peak_sampled_lift_m"] > .1 and check["carry_retained"]
        prediction = client.predict()
        expected = "request_inspection" if manifest["family"] == "F6" else truth["ground_truth"]
        assert prediction["decision"] == expected
        frame = client.packets()[0]["cameras"][0]["frame_id"]
        current = np.asarray(client.image(frame))
        assert not any(center(current, cue) for cue in ("cube", "cyan", "violet"))
        assert (root / "replay.mp4").stat().st_size > 50_000


@pytest.mark.skipif(not EVIDENCE.is_dir(), reason="Run the B1 suite to verify captured physics")
def test_privileged_injection_cannot_change_private_decision(tmp_path):
    source, manifest = next((path, row) for path, row in evidence_worlds() if row["family"] == "F1")
    root = tmp_path / "copy"
    shutil.copytree(source, root)
    replay = Replay(root, "rosmaster-1")
    baseline = replay.packets(), replay.predict()
    path = root / TRUTH / "reveal.jsonl"
    row = rows(path)[0]
    row["ground_truth"] = "injected wrong label"
    row["answer_overlay"] = "privileged coords"
    path.write_bytes(json_bytes(row) + b"\n")
    assert baseline == (replay.packets(), replay.predict())
    row["target_state_time_or_interval"] = time_at(client_decision(manifest))
    path.write_bytes(json_bytes(row) + b"\n")
    with pytest.raises(ValueError, match="cannot label"):
        validate(root, manifest)


def client_decision(manifest):
    return int(manifest["decision_boundary"]["monotonic_ns"])


def view_signature(root, agent):
    client = Replay(root, agent)
    return [(packet["measured_state"], [camera["sha256"] for camera in packet["cameras"]])
            for packet in client.packets()]


@pytest.mark.skipif(not EVIDENCE.is_dir(), reason="Run the B1 suite to verify captured physics")
def test_f6_counterfactual_private_frames_match_and_reveal_differs():
    paths = [root for root, manifest in evidence_worlds() if manifest["family"] == "F6"]
    first, other = paths
    for agent in ("so101-1", "tank-1", "rosmaster-1"):
        assert view_signature(first, agent) == view_signature(other, agent)
    first_truth = rows(first / TRUTH / "reveal.jsonl")[0]
    other_truth = rows(other / TRUTH / "reveal.jsonl")[0]
    assert {first_truth["ground_truth"], other_truth["ground_truth"]} == {"left", "surface"}
    assert first_truth["camera"]["sha256"] != other_truth["camera"]["sha256"]
    assert Replay(first, "rosmaster-1").predict()["belief"] == Replay(other, "rosmaster-1").predict()["belief"]


def test_no_deliveries_are_distinct_from_a_missing_sealed_channel(tmp_path):
    from smart_swarm.episode import DELIVERIES, atomic, sha

    agent = "rosmaster-1"
    manifest = {"world_episode_id": "empty", "policy_cameras": {agent: []}, "query_boundary": time_at(10),
                "decision_boundary": time_at(20), "sidecar_sha256": {}}
    manifest_path = tmp_path / "manifest.json"
    atomic(manifest_path, manifest)
    replay = Replay(tmp_path, agent)
    assert replay.deliveries() == []
    assert replay.predict()["decision"] == "request_inspection"
    channel = tmp_path / DELIVERIES
    channel.parent.mkdir()
    channel.touch()
    with pytest.raises(ValueError, match="sealed provenance"):
        replay.deliveries()
    manifest["sidecar_sha256"][DELIVERIES] = sha(channel)
    atomic(manifest_path, manifest)
    assert Replay(tmp_path, agent).deliveries() == []
    channel.unlink()
    with pytest.raises(ValueError, match="missing"):
        Replay(tmp_path, agent).deliveries()
def test_fresh_inspection_readout_requires_visible_color_and_consistent_location():
    import numpy as np

    from smart_swarm.inspection_view import combine, interpret

    calibration = {"origin_m": [.4, 0, .8], "camera_to_station_rotation": np.eye(3).tolist(),
                   "vertical_fov_deg": 60}
    rgb = np.zeros((120, 160, 3), dtype=np.uint8)
    assert interpret(rgb, calibration)["answer"] == "unresolved"
    rgb[85:91, 48:54] = (250, 64, 5)
    surface = interpret(rgb, calibration)
    assert surface["answer"] == "surface" and surface["target_color_detected"]
    rgb.fill(0)
    rgb[38:44, 52:58] = (250, 64, 5)
    unsupported = interpret(rgb, calibration)
    assert unsupported["target_color_detected"] and unsupported["answer"] == "unresolved"
    rgb[28:34, 52:58] = (5, 240, 240)
    rgb[28:34, 108:114] = (196, 8, 210)
    located = interpret(rgb, calibration)
    assert located["answer"] == "left"
    assert combine([unsupported, located])["answer"] == "left"
    assert combine([surface, located]) == {"answer": "unresolved", "visible_target_cue": True,
                                           "conflicting_localizations": True, "fresh_frames": 2}


def test_inspection_pair_requires_equal_acquisitions_and_respects_selective_decision():
    from smart_swarm.inspection_comparison import follow_up, score_pair, summarize

    views = [{"at_ns": at, "motion": {"max_tracking_error_deg": .2}} for at in (150, 200)]
    scan = {"start_ns": 100, "stop_ns": 200, "samples": views,
            "post_inspection_decision": {"answer": "right"}}
    idle = {**scan, "samples": [{"at_ns": at, "motion": None} for at in (150, 200)],
            "post_inspection_decision": {"answer": "unresolved"}}
    branches = {"scan": scan, "stationary": idle}
    allowed = {arm: follow_up("request_inspection", branch) for arm, branch in branches.items()}
    proposed = {**branches, "methods": {"events_only": allowed}}
    endpoint = {"location": "right", "relation": "carrier"}
    evaluator = {arm: {"before": endpoint, "after": endpoint, "target_displacement_m": 0., "reset_exact": True}
                 for arm in branches}
    result = score_pair("right", proposed, evaluator)
    assert result["paired_scoring_valid"] and result["methods"]["events_only"]["scan"]["correct"]
    assert not result["methods"]["events_only"]["stationary"]["correct"]
    assert follow_up("left", scan) == {"initial_choice": "left", "final_choice": "left",
                                      "extra_rgb_frames": 0, "elapsed_ns": 0}
    row = {"source_group": "paired-reset", "family": "F6", "evaluation": result}
    assert summarize(20, [row])["optical"]["F6"]["difference"] == 1.
    with pytest.raises(ValueError, match="only once"):
        summarize(20, [row, row])
    idle["samples"][0]["at_ns"] += 1
    mismatch = score_pair("right", proposed, evaluator)
    assert not mismatch["paired_scoring_valid"] and mismatch["optical"]["scan"]["correct"] is None
