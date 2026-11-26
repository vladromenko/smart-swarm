from hashlib import sha256

import numpy as np
import pytest

from smart_swarm.contract import interval, json_bytes, time_at
from smart_swarm.episode import DELIVERIES, LABELS, TRUTH, Recorder, atomic, sha
from smart_swarm.evidence_inputs import AGENTS, runtime_signature
from smart_swarm.memory import QUERY
from smart_swarm import physical_evidence_dataset, physical_story_annotations
from smart_swarm.physical_story_annotations import apply_reviews, build_receipt, catalog
from smart_swarm.physical_story_condition import PHASE_DIGEST, phase_digest
from smart_swarm.physical_story_replay import placement_samples
from smart_swarm.primary_inputs import primary_policy_view


def image(markers):
    result = np.zeros((60, 120, 3), dtype=np.uint8)
    colors = {"cyan": (0, 220, 220), "violet": (180, 0, 240), "cube": (235, 80, 10)}
    for label, x, y in markers:
        result[y:y + 5, x:x + 5] = colors[label]
    return result


def story(tmp_path, outcome=None):
    clock = {"host_id": "homebrain", "boot_id": "test", "kind": "monotonic"}
    grants = {AGENTS[0]: [AGENTS[0] + "/top"], **{agent: [agent + "/front"] for agent in AGENTS[1:]}}
    recorder = Recorder(tmp_path / "private-replay", "scope-test", "group", grants,
                        {"run_mode": "hardware", "native_recording_clock": clock}, clock=clock)
    for issue in (*range(11, 15), *range(31, 35)):
        rgb = image((("cyan", 20, 20), ("cube", 26 if issue < 20 else 90, 20)))
        recorder.capture(AGENTS[0], {grants[AGENTS[0]][0]: rgb}, issue, [], [])
    for issue in range(41, 46):
        rgb = image((("cyan", 18 if issue == 41 else 50, 20), ("violet", 80, 20)))
        recorder.capture(AGENTS[1], {grants[AGENTS[1]][0]: rgb}, issue, [], [])
    recorder.capture(AGENTS[2], {grants[AGENTS[2]][0]: image(())}, 58, [], [])
    recorder.append("communication", "queries", {"text": QUERY, "issued_time": time_at(60, clock)})
    (recorder.root / DELIVERIES).touch()
    reveal = recorder.frame(TRUTH, "evaluator/top", 1, image((("cube", 90, 20),)), 70)
    recorder.append(TRUTH, "reveal", {"camera": reveal, "relations_preserved_until_reveal": True,
                                      "ground_truth": outcome, "observed_at": time_at(70, clock),
                                      "target_state_time_or_interval": interval(65, 70, clock)})
    recorder.close(60, 65, 70)
    state = {"status": "complete", "card": {"story_id": "scope-test", "family": "F3",
                                           "requested_carrier_destination": "right"},
             "start": {"monotonic_ns": 10}, "markers": {"insertion_closed": {"monotonic_ns": 20},
             "placement_closed": {"monotonic_ns": 40}, "relocation_closed": {"monotonic_ns": 50}},
             "decision": {"at": time_at(65, clock)}}
    if outcome is not None:
        state["observed_terminal_outcome"] = outcome
    return recorder.root, state


def confirmation(item, value):
    return {"value": value, "basis": "The stationary relation is visible in the selected original frame.",
            "frame_ids": [item["review_images"][-1]["frame_id"]], "directly_observed": True}


def save_review(path, root, state, available, values):
    record = build_receipt(state, sha(root / "manifest.json"), values, available)
    checksum = sha256(json_bytes(record) + b"\n").hexdigest()
    target = path / LABELS / "reviewed-events" / f"{checksum}.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    atomic(target, record)


def test_f3_confirmed_partial_events_do_not_change_scoped_replay(tmp_path):
    root, state = story(tmp_path)
    available = catalog(root, state)
    assert {name: item["candidate"]["value"] for name, item in available.items()} == {
        "first_insertion": "cyan", "settled_placement": "surface", "relocation_assignment": "violet_left"}
    assert catalog(root, {**state, "card": state["card"] | {"requested_carrier_destination": "left"}}) == available
    labels = {"terminal": None, "settled_placement": None, "relocation_assignment": None, "intermediate_events": None}
    record = {"manifest_sha256": sha(root / "manifest.json"), "labels": labels}
    before = runtime_signature(root)
    assert apply_reviews(tmp_path, root, state, record)["labels"] == labels
    save_review(tmp_path, root, state, available,
                {"first_insertion": confirmation(available["first_insertion"], "cyan"),
                 "settled_placement": confirmation(available["settled_placement"], "surface")})
    result = apply_reviews(tmp_path, root, state, record)
    assert result["labels"] == {"terminal": None, "settled_placement": "surface", "relocation_assignment": None,
                                "intermediate_events": {"first_insertion": "cyan"}}
    assert runtime_signature(root) == before
    with primary_policy_view(root) as scoped:
        assert runtime_signature(scoped) == before
        assert not (scoped / LABELS).exists() and not (scoped / TRUTH).exists()
    save_review(tmp_path, root, state, available, {"settled_placement": confirmation(available["settled_placement"], "held")})
    conflict = apply_reviews(tmp_path, root, state, record)
    assert conflict["labels"]["settled_placement"] is None
    assert conflict["event_annotations"]["events"]["settled_placement"]["status"] == "conflicting_confirmations"


def test_unknown_terminal_can_be_reviewed_only_from_its_own_sealed_reveal(tmp_path):
    root, state = story(tmp_path, "unknown")
    available = catalog(root, state)
    labels = {"terminal": None, "settled_placement": None, "relocation_assignment": None, "intermediate_events": None}
    record = {"manifest_sha256": sha(root / "manifest.json"), "labels": labels}
    before = runtime_signature(root)
    original = sha(root / TRUTH / "reveal.jsonl")
    terminal = available["terminal"]
    assert terminal["status"] == "needs_confirmation"
    assert terminal["review_images"][0]["access_role"] == "evaluator_only"
    with pytest.raises(ValueError, match="wrong-phase"):
        save_review(tmp_path, root, state, available, {"terminal": confirmation(available["settled_placement"], "surface")})
    with pytest.raises(ValueError, match="wrong-phase"):
        save_review(tmp_path, root, state, available, {"settled_placement": confirmation(terminal, "surface")})
    save_review(tmp_path, root, state, available, {"terminal": confirmation(terminal, "surface")})
    result = apply_reviews(tmp_path, root, state, record)
    assert result["labels"] == {**labels, "terminal": "surface"}
    assert result["event_annotations"]["events"]["terminal"]["status"] == "confirmed"
    assert sha(root / TRUTH / "reveal.jsonl") == original
    with primary_policy_view(root) as scoped:
        assert runtime_signature(scoped) == before
        assert not (scoped / LABELS).exists() and not (scoped / TRUTH).exists()
    save_review(tmp_path, root, state, available, {"terminal": confirmation(terminal, "left")})
    conflict = apply_reviews(tmp_path, root, state, record)
    assert conflict["labels"] == labels
    assert conflict["event_annotations"]["events"]["terminal"]["status"] == "conflicting_confirmations"


def test_known_attended_outcome_cannot_be_relabelled(tmp_path):
    root, state = story(tmp_path, "right")
    available = catalog(root, state)
    terminal = available["terminal"]
    assert terminal["status"] == "confirmed_at_capture" and terminal["attended_outcome"] == "right"
    with pytest.raises(ValueError, match="already confirmed"):
        save_review(tmp_path, root, state, available, {"terminal": confirmation(terminal, "left")})


def test_interactive_review_cites_only_preselected_images(tmp_path, monkeypatch):
    from PIL import Image
    from smart_swarm import physical_annotation_review

    root, state = story(tmp_path, "unknown")
    available = catalog(root, state)
    before = runtime_signature(root)
    stored = {event: {"status": item["status"]} for event, item in available.items()}
    answers = iter(("", "", "2", "n", "3", "The overhead image shows the exposed cube on the surface."))
    monkeypatch.setattr("builtins.input", lambda _: next(answers))
    views = []

    def show(uri, new):
        assert new == 2
        with Image.open(uri.removeprefix("file://")) as image:
            views.append(image.size)
        return True

    monkeypatch.setattr(physical_annotation_review.webbrowser, "open", show)
    fields = physical_annotation_review.review(root, available, stored)
    receipt = build_receipt(state, sha(root / "manifest.json"), fields, available)
    assert views == [(360, 60), (360, 60), (360, 60), (120, 60)]
    assert set(receipt["observations"]) == {"terminal"}
    assert receipt["observations"]["terminal"]["value"] == "surface"
    assert [frame["frame_id"] for frame in receipt["observations"]["terminal"]["frames"]] == [
        frame["frame_id"] for frame in available["terminal"]["review_images"]]
    assert runtime_signature(root) == before


def test_attended_phase_attestation_stays_out_of_complete_runtime_input(tmp_path):
    from smart_swarm.episode import load

    root, _ = story(tmp_path)
    path = root / "manifest.json"
    manifest = load(path)
    before = runtime_signature(root)
    for outcome in ("placement_succeeded", "placement_failed"):
        digest = phase_digest({"placement_closed": {"observed_placement_outcome": outcome}})
        atomic(path, {**manifest, PHASE_DIGEST: digest})
        with primary_policy_view(root) as scoped:
            assert PHASE_DIGEST not in load(scoped / "manifest.json")
            assert runtime_signature(scoped) == before


def test_wrong_phase_hidden_window_and_unverified_slot_reference_refused(tmp_path):
    root, state = story(tmp_path)
    digest = sha(root / "manifest.json")
    available = catalog(root, state)
    late = confirmation(available["settled_placement"], "cyan")
    with pytest.raises(ValueError, match="wrong-phase"):
        build_receipt(state, digest, {"first_insertion": late}, available)
    with pytest.raises(ValueError, match="Complete the saved decision"):
        build_receipt({**state, "status": "pending_reveal"}, digest, {"settled_placement": late}, available)
    label = confirmation(available["relocation_assignment"], "violet_left") | {"rosmaster_left_reference_verified": False}
    with pytest.raises(ValueError, match="ROSMASTER left reference"):
        build_receipt(state, digest, {"relocation_assignment": label}, available)
    hidden = catalog(root, {**state, "card": state["card"] | {"family": "F6"}})
    assert hidden["settled_placement"]["status"] == "withheld_by_family"
    assert hidden["settled_placement"]["review_images"] == []


def test_f3_records_four_distinct_samples_on_both_sides_of_insertion_marker():
    source = [{"observation_timestamp_ns": value} for value in range(2, 20)]
    state = {"card": {"family": "F3"}, "markers": {"insertion_closed": {"monotonic_ns": 10}}}
    frames = placement_samples(source, state, False)
    assert len(frames) == len({row["observation_timestamp_ns"] for row in frames}) == 8
    assert frames[3]["observation_timestamp_ns"] == 10
    assert frames[4]["observation_timestamp_ns"] == 11


def test_annotation_command_publishes_and_retries_the_same_receipt(tmp_path, monkeypatch):
    workspace = tmp_path / "workspace"
    trial = workspace / "scope-test"
    root, state = story(trial, "unknown")
    atomic(trial / "trial.json", state)
    card = state["card"]
    record = {"path": str(root), "manifest_sha256": sha(root / "manifest.json"),
              "labels": {"terminal": None, "settled_placement": None, "relocation_assignment": None, "intermediate_events": None}}
    monkeypatch.setattr(physical_story_annotations, "story_workspace", lambda _config: workspace)
    monkeypatch.setattr(physical_story_annotations, "card_for", lambda *_: card)

    def reopen(*_):
        return apply_reviews(trial, root, state, record)

    publications = []
    monkeypatch.setattr(physical_evidence_dataset, "load_admitted", reopen)
    monkeypatch.setattr(physical_evidence_dataset, "publish",
                        lambda _config: publications.append(reopen()["labels"]) or {"manifest": "published"})
    before = physical_story_annotations.annotation_action({}, card["story_id"])
    submitted = tmp_path / "review.json"
    atomic(submitted, {"settled_placement": confirmation(before["events"]["settled_placement"], "surface"),
                       "terminal": confirmation(before["events"]["terminal"], "surface")})
    first = physical_story_annotations.annotation_action({}, card["story_id"], submitted)
    again = physical_story_annotations.annotation_action({}, card["story_id"], submitted)
    assert first["saved_receipt"] == again["saved_receipt"]
    assert first["evidence_corpus_manifest"] == again["evidence_corpus_manifest"] == "published"
    assert [row["settled_placement"] for row in publications] == ["surface", "surface"]
    assert [row["terminal"] for row in publications] == ["surface", "surface"]
