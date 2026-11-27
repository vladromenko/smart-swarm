from contextlib import contextmanager
import json
from subprocess import TimeoutExpired
import time

import numpy as np
import pytest

from smart_swarm import physical_story_operator as operator
from smart_swarm import physical_story_motor as motor
from smart_swarm import physical_story_replay as replay
from smart_swarm import physical_reveal
from smart_swarm import physical_story_source as source
from smart_swarm.contract import ProtocolError, interval, time_at
from smart_swarm.episode import DELIVERIES, TRUTH, Recorder, atomic, load, sha
from smart_swarm.memory import QUERY, predict


def test_native_interruption_with_missing_markers_keeps_f3_insertion(monkeypatch, tmp_path):
    config = {"data_root": tmp_path}
    trial = source.story_workspace(config) / "interrupted-f3"
    trial.mkdir()
    state = {"status": "recording", "card": {"family": "F3"}, "fleet_capture_id": "a" * 32,
             "markers": {"insertion_closed": {"observed_placement_outcome": "placement_succeeded"}}}
    atomic(trial / "trial.json", state)
    monkeypatch.setattr(replay, "card_for", lambda *_: None)
    status = {"recording_active": True, "teleoperation_active": False, "policy_active": False, "learning_job_active": True}
    monkeypatch.setattr(replay, "native_status", lambda *_: status)
    monkeypatch.setattr(replay, "fleet_manifest", lambda *_: {"phase": "interrupted", "error": "Native clock expired"})

    def preserve(config, path, status, *, motor_only):
        assert motor_only and path == trial and set(status["markers"]) == {"insertion_closed"}
        status.update(native_episode_id="recorded-studio", motor_segment_admitted=True)
        atomic(path / "motor-intake.json", {"status": "reviewable_prefix"})

    monkeypatch.setattr(replay, "prepare_native_arm", preserve)
    with pytest.raises(ProtocolError, match="native recording"):
        replay.predict_selected(config, trial.name)
    status["recording_active"] = False
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


def staged_native_fixture(monkeypatch, tmp_path):
    config = {"data_root": tmp_path, "studio_project": tmp_path / "studio"}
    now = {"monotonic_ns": 1_000_000_000, "unix_ns": 1_000_000_000}
    episodes = {}
    studio = dict.fromkeys(("recording_active", "teleoperation_active", "policy_active",
                            "native_arm_hardware_open", "native_stop_check_required"), False)
    monkeypatch.setattr(source, "anchor", lambda: now.copy())
    monkeypatch.setattr(source, "story_preflight", lambda *_, **_options: None)
    monkeypatch.setattr(source, "native_status", lambda *_: studio)
    monkeypatch.setattr(source, "dataset", lambda *_: "studio-dataset")
    monkeypatch.setattr(source, "episode_ids", lambda *_: episodes)

    def acknowledge(_config, trial):
        receipt = {"id": "a" * 32, "requested_seconds": 60,
                   "acknowledged": {"monotonic_ns": 101_000_000_000, "unix_ns": 101_000_000_000}}
        atomic(trial / "fleet-start.json", receipt)
        return receipt["id"]

    monkeypatch.setattr(source, "launch_fleet", acknowledge)
    return config, now, episodes


def write_staged_native_episode(config, name):
    directory = config["studio_project"] / "data/studio-dataset/episodes" / name
    directory.mkdir(parents=True)
    ledger = directory / "frames.jsonl"
    ledger.write_text("{}\n")
    meta = {"status": "complete", "first_timestamp_ns": 2_000_000_000, "last_timestamp_ns": 31_000_000_000,
            "frames_sha256": sha(ledger), "frames": 1}
    atomic(directory / "episode.json", meta)
    return {"id": name, "storage": "episodes", "metadata": meta}, directory / "episode.json"


def test_studio_phase_finishes_before_the_staged_native_capture(monkeypatch, tmp_path):
    config, now, episodes = staged_native_fixture(monkeypatch, tmp_path)
    card = source.prepare(config)["cards"][0]
    identifier = card["story_id"]
    state = source.begin(config, identifier, staged=True)
    assert state["status"] == "placement_recording" and "fleet_capture_id" not in state
    now.update(monotonic_ns=30_000_000_000, unix_ns=30_000_000_000)
    source.mark(config, identifier, "placement_closed", placement_outcome="placement_succeeded")
    with pytest.raises(source.StudioEpisodeUnlinked):
        source.start_relocation(config, identifier)
    source_row, episode_path = write_staged_native_episode(config, "matching-studio")
    episodes[source_row["id"]] = source_row
    now.update(monotonic_ns=100_000_000_000, unix_ns=100_000_000_000)
    with monkeypatch.context() as offline:
        def disconnected(_, *, require_attended=True):
            assert not require_attended
            raise ProtocolError("OFFLINE", "Tank RGB expired")

        offline.setattr(source, "story_preflight", disconnected)
        with pytest.raises(ProtocolError, match="Tank RGB expired"):
            source.start_relocation(config, identifier)
    selected, staged = operator.next_story(config)
    assert selected == card and staged["status"] == "placement_recording"
    assert staged["staged_studio_episode_id"] == source_row["id"] and staged["reason"] == "Tank RGB expired"
    sealed = staged["staged_studio_recording"]
    original = load(episode_path)
    source_row["metadata"] = {**original, "note": "different native metadata"}
    atomic(episode_path, source_row["metadata"])
    with pytest.raises(ValueError, match="exact Studio recording changed"):
        source.start_relocation(config, identifier)
    source_row["metadata"] = original
    atomic(episode_path, original)
    episodes["later"] = write_staged_native_episode(config, "later")[0]
    state = source.start_relocation(config, identifier)
    assert state["staged_studio_episode_id"] == source_row["id"]
    assert state["staged_studio_recording"] == sealed and "reason" not in state
    assert state["fleet_acknowledgement"]["monotonic_ns"] == 101_000_000_000
    assert state["fleet_window_s"] == 60 and state["status"] == "recording"
    assert state["fleet_settled_window_s"] == 4


def test_mobile_acknowledgement_failure_preserves_the_saved_placement(monkeypatch, tmp_path):
    state = {"status": "preparing", "start": {"unix_ns": 1}, "fleet_launch_requested": {"unix_ns": 2},
             "markers": {"placement_closed": {"observed_placement_outcome": "placement_succeeded"}}}
    monkeypatch.setattr(source, "data", lambda *_: {"active": None})

    def preserve(_config, trial, saved, reason):
        assert saved == state and "never acknowledged" in reason
        saved["motor_segment_admitted"] = True
        source.save_incomplete(trial, saved, reason)

    monkeypatch.setattr(replay, "preserve_finished_prefix", preserve)
    closed = source.restore_start({}, tmp_path, state)
    assert closed["status"] == "incomplete" and closed["motor_segment_admitted"] is True


def test_close_interrupted_studio_card_keeps_the_marker_and_opens_next_card(monkeypatch, tmp_path):
    config = {"data_root": tmp_path}
    first, second = source.prepare(config)["cards"][:2]
    trial = source.story_workspace(config) / first["story_id"]
    trial.mkdir()
    mark = {"monotonic_ns": 2, "observed_placement_outcome": "placement_succeeded"}
    atomic(trial / "trial.json", {"card": first, "status": "placement_recording", "start": {"monotonic_ns": 1},
                                 "studio_dataset_id": "studio-dataset", "preexisting_native_episode_ids": [],
                                 "markers": {"placement_closed": mark}})
    monkeypatch.setattr(replay, "native_status", lambda *_: {"recording_active": False})
    monkeypatch.setattr(source, "episode_ids", lambda *_: {})
    assert operator.next_story(config)[0] == first
    result = replay.close_interrupted_story(config, trial.name, "Studio did not save this take")
    stored = load(trial / "trial.json")
    before = sha(trial / "trial.json")
    assert result["status"] == "incomplete" and "Studio did not save" in result["reason"]
    assert stored["observed_terminal_outcome"] == "unknown" and stored["markers"]["placement_closed"] == mark
    assert load(trial / "motor-intake.json")["status"] == "pending_review"
    assert replay.close_interrupted_story(config, trial.name, "Another reason") == result
    assert sha(trial / "trial.json") == before
    assert operator.next_story(config) == (second, None)


@pytest.mark.parametrize("studio_active", [True, False])
def test_close_interrupted_story_waits_for_native_writers(monkeypatch, tmp_path, studio_active):
    config = {"data_root": tmp_path}
    card = source.prepare(config)["cards"][0]
    trial = source.story_workspace(config) / card["story_id"]
    trial.mkdir()
    journal = trial / "trial.json"
    atomic(journal, {"card": card, "status": "recording", "fleet_capture_id": "a" * 32, "markers": {}})
    before = sha(journal)
    monkeypatch.setattr(replay, "native_status", lambda *_: {"recording_active": studio_active})
    monkeypatch.setattr(replay, "fleet_manifest", lambda *_: {"phase": "recording"})
    expected = "Studio recording" if studio_active else "Explorer capture"
    with pytest.raises(ProtocolError, match=expected):
        replay.close_interrupted_story(config, trial.name, "Interrupted test")
    assert sha(journal) == before and not (trial / "motor-intake.json").exists()


def test_two_launchers_cannot_open_overlapping_story_journals(monkeypatch, tmp_path):
    from fcntl import LOCK_EX, LOCK_NB, flock

    config = {"data_root": tmp_path}
    card, other = source.prepare(config)["cards"][:2]
    monkeypatch.setattr(source, "story_preflight", lambda *_: None)
    monkeypatch.setattr(source, "dataset", lambda *_: "studio-dataset")
    monkeypatch.setattr(source, "episode_ids", lambda *_: {})
    workspace = source.story_workspace(config)
    with (workspace / ".journal.lock").open("a") as writer:
        flock(writer, LOCK_EX | LOCK_NB)
        with pytest.raises(ProtocolError, match="updating the story journal"):
            source.begin(config, other["story_id"], staged=True)
    interrupted = workspace / card["story_id"]
    interrupted.mkdir()
    temporary = interrupted / ("trial.json." + "a" * 32 + ".tmp")
    temporary.write_bytes(b"uncommitted draft")
    assert source.begin(config, card["story_id"], staged=True)["status"] == "placement_recording"
    assert temporary.read_bytes() == b"uncommitted draft"
    with pytest.raises(ProtocolError, match="Complete the active physical story"):
        source.begin(config, other["story_id"], staged=True)
    assert not (workspace / other["story_id"]).exists()


def test_mobile_countdown_uses_the_native_acknowledgement(monkeypatch, capsys):
    state = {"fleet_acknowledgement": {"monotonic_ns": 100_000_000_000}, "fleet_window_s": 60}
    ticks = [107_000_000_000]
    monkeypatch.setattr(operator, "is_capturing", lambda *_: True)
    monkeypatch.setattr(operator.time, "monotonic_ns", lambda: ticks[0])
    monkeypatch.setattr(operator.time, "sleep", lambda seconds: ticks.__setitem__(0, ticks[0] + round(seconds * 1e9)))
    assert operator.wait_stationary_lead_in({}, state)
    assert ticks[0] == 112_000_000_000
    assert "48 seconds remain" in capsys.readouterr().out
    ticks[0] = 161_000_000_000
    assert not operator.wait_stationary_lead_in({}, state)


def test_relocation_waits_for_recorded_rgb_and_preserves_a_late_start(monkeypatch, tmp_path, capsys):
    from smart_swarm.native_fleet_start import START_GATE

    config = {"data_root": tmp_path}
    identifier = source.prepare(config)["cards"][0]["story_id"]
    state = {"fleet_acknowledgement": {"monotonic_ns": 100_000_000_000}, "fleet_window_s": 60,
             "fleet_rgb_start_gate": START_GATE, "status": "recording", "fleet_capture_id": "a" * 32,
             "markers": {"placement_closed": {}}}
    ticks = [107_000_000_000]
    monkeypatch.setattr(operator, "is_capturing", lambda *_: True)
    monkeypatch.setattr(operator.time, "monotonic_ns", lambda: ticks[0])
    monkeypatch.setattr(operator.time, "sleep", lambda seconds: ticks.__setitem__(0, ticks[0] + round(seconds * 1e9)))
    monkeypatch.setattr(operator, "archived_rgb_started", lambda *_: ticks[0] >= 134_000_000_000)
    assert operator.wait_stationary_lead_in(config, state)
    assert ticks[0] == 134_000_000_000
    assert "26 seconds remain" in capsys.readouterr().out
    monkeypatch.setattr(operator, "archived_rgb_started", lambda *_: False)
    monkeypatch.setattr(operator, "observe_motion", lambda *_args, **_kwargs: pytest.fail("No relocation prompt"))
    journal = source.story_workspace(config) / identifier / "trial.json"
    journal.parent.mkdir()
    atomic(journal, state)
    persisted = operator.record_relocation(config, identifier, state)
    assert persisted == load(journal)
    assert ticks[0] == 145_000_000_000
    withheld = load(journal)["relocation_prompt_withheld"]
    reason = replay.capture_failure({"phase": "complete", "readback": {"passed": True}}, ["relocation_closed"], withheld=withheld)
    assert "usable RGB start" in reason


def test_reopening_after_relocation_prompt_keeps_native_recording_and_boundary(monkeypatch, tmp_path, capsys):
    config = {"data_root": tmp_path}
    card = source.prepare(config)["cards"][0]
    trial = source.story_workspace(config) / card["story_id"]
    trial.mkdir()
    capture = "1" * 32
    placement = {"observed_placement_outcome": "placement_succeeded"}
    state = {"status": "recording", "card": card, "fleet_capture_id": capture, "fleet_launch_requested": {},
             "markers": {"placement_closed": placement}}
    journal = trial / "trial.json"
    atomic(journal, state)
    fleet = {"phase": "recording", "readback": {"passed": True}}
    monkeypatch.setattr(source, "fleet_manifest", lambda *_: fleet)
    monkeypatch.setattr(operator, "wait_stationary_lead_in", lambda *_: True)

    def interrupt(*_):
        raise EOFError

    monkeypatch.setattr("builtins.input", interrupt)
    with pytest.raises(EOFError):
        operator.record_relocation(config, trial.name, state)
    assert capsys.readouterr().out.count("Move the carrier now") == 1
    persisted = load(journal)
    assert persisted["relocation_prompt_opened"]["monotonic_ns"] > 0
    assert source.claim_relocation_prompt(config, trial.name, capture) == (persisted, False)
    monkeypatch.setattr("builtins.input", lambda *_: pytest.fail("A reopened prompt requested fresh input"))
    assert operator.record_relocation(config, trial.name, persisted) == persisted
    assert "Move the carrier now" not in capsys.readouterr().out
    fleet["phase"] = "complete"
    monkeypatch.setattr(replay, "fleet_manifest", lambda *_: fleet)
    monkeypatch.setattr(replay, "native_status", lambda *_: dict.fromkeys(("recording_active", "teleoperation_active", "policy_active"), False))
    reviewed = []
    monkeypatch.setattr(replay, "prepare_native_arm", lambda *_, **kw: reviewed.append(kw["motor_only"]))
    result = replay.predict_selected(config, trial.name)
    assert result["status"] == "incomplete" and "without a confirmation" in result["reason"]
    assert load(journal)["markers"] == {"placement_closed": placement}
    assert reviewed == [True]


def test_tank_clip_keeps_two_observations_after_settling():
    nanosecond = 10**9
    phase = {"fleet_launch_requested": {"monotonic_ns": 10 * nanosecond},
             "fleet_acknowledgement": {"monotonic_ns": 11 * nanosecond}, "fleet_window_s": 60,
             "markers": {"placement_closed": {"monotonic_ns": 8 * nanosecond},
                         "relocation_closed": {"monotonic_ns": 25 * nanosecond}}}
    old_bounds = source.mobile_phase_bounds(phase, 40 * nanosecond)
    phase["fleet_settled_window_s"] = 4
    assert source.mobile_phase_bounds(phase, 40 * nanosecond) == (
        (old_bounds[0][0], 29 * nanosecond), (29 * nanosecond, old_bounds[1][1]))
    samples = [{"received_ns": second * nanosecond + 70,
                "exposure_bounds_ns": (second * nanosecond - 20, second * nanosecond + 20)}
               for second in (12, 14, 16, 18, 19, 20, 22, 24, 25, 26, 27)]
    with pytest.raises(ValueError, match="1 of 2 required settled"):
        replay.select_fleet_clip(samples[:-1], 8, "tank-1", phase, {})
    proof = {}
    selected = replay.select_fleet_clip(samples, 8, "tank-1", phase, proof)
    assert len(selected) == 8 and selected[-2:] == samples[-2:]
    assert samples[-3] not in selected
    assert proof["settled_selected"] == proof["settled_eligible"] == 2


def test_native_archive_retry_keeps_the_original_when_explorer_readback_fails(tmp_path):
    from smart_swarm.physical_archive_storage import retain_directory

    source_path = tmp_path / "native" / "capture"
    source_path.mkdir(parents=True)
    atomic(source_path / "manifest.json", {"phase": "complete"})
    (source_path / "camera").write_bytes(b"native-camera-bytes")
    destination = tmp_path / "storage" / "capture"

    def failed_readback():
        assert source_path.is_symlink()
        raise OSError("Native archive endpoint was unavailable")

    with pytest.raises(OSError, match="endpoint"):
        retain_directory(source_path, destination, failed_readback)
    assert source_path.is_dir() and not source_path.is_symlink()
    assert (source_path / "camera").read_bytes() == b"native-camera-bytes"
    proof = retain_directory(source_path, destination, lambda: None)
    assert proof["file_count"] == 2 and source_path.resolve() == destination
    assert (source_path / "camera").read_bytes() == b"native-camera-bytes"


def test_native_archive_keeps_a_divergent_recovery_copy(tmp_path):
    from smart_swarm.physical_archive_storage import retain_directory

    destination = tmp_path / "storage" / "capture"
    destination.mkdir(parents=True)
    atomic(destination / "manifest.json", {"phase": "complete"})
    source_path = tmp_path / "native" / "capture"
    backup = source_path.parent / ".smart-swarm-archive-transfer" / source_path.name
    backup.mkdir(parents=True)
    atomic(backup / "manifest.json", {"phase": "complete", "recovery_copy": True})
    source_path.symlink_to(destination, target_is_directory=True)
    with pytest.raises(ValueError, match="differs from the local copy"):
        retain_directory(source_path, destination, lambda: None)
    assert source_path.resolve() == destination
    assert load(backup / "manifest.json")["recovery_copy"]


def test_studio_archive_keeps_a_verified_prefix_path_and_refuses_other_links(monkeypatch, tmp_path):
    from smart_swarm import studio_export_storage as storage

    config = {"studio_project": tmp_path / "studio", "physical_archive_volume": tmp_path / "secondary"}
    source_path = config["studio_project"] / "data/ds_1234567890123456"
    original = config["studio_project"] / "runtime/smart-swarm-pilot-exports/abcdef000001-01234567"
    destination = tmp_path / "secondary" / original.name
    meta = original / "provenance/export.json"
    meta.parent.mkdir(parents=True)
    atomic(meta, {"status": "complete", "source_path": str(source_path), "dataset_sha256": "proof",
                  "official_verification": {"rgb_pixels_bit_exact": True}})
    monkeypatch.setattr(storage, "secondary_export_path", lambda *_: destination)
    validation = {"dataset_sha256": "proof"}
    scope = tmp_path / "intake"
    result = storage.archive_accepted_export(config, source_path, original, scope, validation, lambda: validation)
    assert result["state"] == "retained" and storage.checked_export_path(config, original) == destination
    assert load(meta)["dataset_sha256"] == "proof"
    repeat = storage.archive_accepted_export(config, source_path, original, scope, validation, lambda: pytest.fail("reopened"))
    assert repeat == result
    unrelated = original.with_name("abcdef000002-01234567")
    unrelated.symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(ValueError, match="leaves its original"):
        storage.checked_export_path(config, unrelated)


def test_operator_previews_next_block_and_displays_active_card_station(monkeypatch, tmp_path, capsys):
    config = {"data_root": tmp_path}
    plan = operator.prepare(config)
    original_plan = sha(tmp_path / "collection-plans" / "physical-stories-v1.json")
    root = source.story_workspace(config)
    for row in plan["cards"][:6]:
        trial = root / row["story_id"]
        trial.mkdir()
        atomic(trial / "trial.json", {"status": "incomplete"})
    with monkeypatch.context() as no_recovery:
        no_recovery.setattr(operator, "recover_story", lambda *_: pytest.fail("A preview performed native recovery"))
        operator.show(config)
    rendered = capsys.readouterr().out
    assert "Block 2 of 4" in rendered and "Next physical-stories-v1-b02-f1" in rendered
    assert "Assigned carrier station is right as seen by ROSMASTER." in rendered
    assert rendered.count("·") == 6 and "b01-f1" not in rendered

    monkeypatch.setattr(operator, "story_preflight", lambda *_: None)
    monkeypatch.setattr("builtins.input", lambda *_: "q")
    monkeypatch.setattr(operator, "begin", lambda *_: pytest.fail("A nonempty response started a recording"))
    operator.run(config)
    selected = capsys.readouterr().out
    assert "Block 2 · F1" in selected and "Assigned carrier station is right" in selected
    assert sha(tmp_path / "collection-plans" / "physical-stories-v1.json") == original_plan


def test_saved_reveal_prompt_does_not_repeat_setup_or_scene_change(capsys, tmp_path):
    card = operator.prepare({"data_root": tmp_path})["cards"][0]
    operator.show_selected_card(card, {"status": "pending_reveal", "markers": {}})
    text = capsys.readouterr().out
    assert "terminal relation unchanged" in text
    assert card["placement"] not in text and card["relocation"] not in text and card["reset"] not in text


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


@pytest.mark.parametrize("already_saved", [True, False])
def test_interrupted_f3_preserves_insertion_with_later_marker(monkeypatch, tmp_path, already_saved):
    config = {"data_root": tmp_path, "studio_project": tmp_path / "studio"}
    dataset, episode = "ds_1234567890123456", "recorded-insertion"
    trial = source.story_workspace(config) / "f3"
    trial.mkdir()
    native = config["studio_project"] / "data" / dataset / "episodes" / episode
    native.mkdir(parents=True)
    timestamps = [12_000_000_000 + i * 200_000_000 for i in range(32)]
    frames = [{"frame_id": index, "observation_timestamp_ns": ns, "action_timestamp_ns": ns,
               "state_valid": True, "action_valid": True} for index, ns in enumerate(timestamps)]
    manifest = native / "frames.jsonl"
    manifest.write_text("".join(json.dumps(row) + "\n" for row in frames))
    proof = {"episode_id": episode, "dataset_id": dataset, "frames_sha256": sha(manifest)}
    inventory = {episode: {"id": episode, "storage": "episodes",
                          "metadata": {**proof, "first_timestamp_ns": timestamps[0], "last_timestamp_ns": timestamps[-1]}}}
    card = {"story_id": "f3", "family": "F3", "source_group_id": "block-1", "split": "development"}
    insertion = {"monotonic_ns": timestamps[-2], "observed_placement_outcome": "placement_succeeded",
                 "assistance": "attended_native_motion"}
    state = {"card": card, "status": "recording", "preexisting_native_episode_ids": [],
             "start": {"monotonic_ns": 10_000_000_000}, "studio_dataset_id": dataset,
             "markers": {"insertion_closed": insertion, "placement_closed": {"monotonic_ns": 27_000_000_000}}}
    available = dict(inventory) if already_saved else {}
    monkeypatch.setattr(source, "episode_ids", lambda *_: available)
    monkeypatch.setattr(replay, "validate_story_episode", lambda *_: proof)
    result = replay.preserve_finished_prefix(config, trial, state, "Explorer capture interrupted")
    saved = load(trial / "trial.json")
    assert result["status"] == "incomplete" and "decision" not in saved
    assert load(trial / "motor-intake.json")["status"] == "pending_review"
    available.update(inventory)
    with pytest.raises(source.StudioEpisodeUnlinked):
        replay.prepare_native_arm(config, trial, saved)
    linked = motor.linked_sources(config, [(trial, saved)])
    assert list(linked) == [dataset] and saved["native_episode_id"] == episode
    segment, reason = replay.segment_choice(config, saved)
    assert reason is None and segment["end_boundary"] == "insertion_closed" and segment["sample_count"] == 31


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
    atomic(trial / "trial.json", {"status": "pending_reveal", "decision": decision, "query_ns": 10,
                                 "homebrain_boot_id": clock["boot_id"]})
    monkeypatch.setattr(replay, "card_for", lambda *_: None)
    monkeypatch.setattr(source, "homebrain_boot", lambda: "later-boot")
    monkeypatch.setattr(replay, "capture_clip", lambda *_: pytest.fail("A sealed reveal needs no new image"))
    sealed = (root / "manifest.json").read_bytes()
    assert replay.predict_selected(config, trial.name) == decision
    assert load(trial / "trial.json")["status"] == "complete"
    assert (root / "manifest.json").read_bytes() == sealed
    with pytest.raises(ValueError, match="already has a sealed outcome"):
        replay.close_interrupted_story(config, trial.name, "Interrupted test")
    assert load(trial / "trial.json")["observed_terminal_outcome"] == "surface"
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


def test_terminal_reveal_can_finish_while_only_a_learning_job_is_active(monkeypatch, tmp_path):
    config = {"data_root": tmp_path}
    trial = source.story_workspace(config) / "learning-job-active"
    root = trial / "private-replay"
    clock = replay.boot_clock()
    metadata = {"run_mode": "hardware", "native_recording_clock": clock, physical_reveal.FIELD: physical_reveal.PROFILE}
    recorder = Recorder(root, trial.name, "group", {"rosmaster-1": []}, metadata, clock=clock)
    recorder.capture("rosmaster-1", {}, 5, [], [])
    decision = {"at": time_at(15, clock), "query_all": {"decision": "request_inspection"}}
    recorder.append("decisions", "prediction", decision)
    atomic(trial / "trial.json", {"status": "pending_reveal", "decision": decision, "query_ns": 10})
    status = {"recording_active": False, "teleoperation_active": False, "policy_active": True, "learning_job_active": True}
    monkeypatch.setattr(replay, "card_for", lambda *_: None)
    monkeypatch.setattr(replay, "native_status", lambda *_: status)
    monkeypatch.setattr(replay, "capture_clip", lambda *_: [reveal_sample(index) for index in range(3)])
    with pytest.raises(ProtocolError, match="policy job"):
        replay.terminal_reveal(config, trial.name, "surface", True)
    assert not list((root / TRUTH).iterdir())
    status["policy_active"] = False
    result = replay.terminal_reveal(config, trial.name, "surface", True)
    assert result["status"] == "complete" and result["observed_terminal_outcome"] == "surface"
    assert physical_reveal.validate_sealed_clip(root, load(root / TRUTH / "reveal.jsonl"), load(root / "manifest.json")) == 3


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
def admitted_motor(*_, archive_only=False):
    assert archive_only
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


def test_motor_review_rebinds_a_rejected_export_and_keeps_the_old_receipt(monkeypatch, tmp_path):
    from smart_swarm.physical_motor_prefix import store_prefix
    from smart_swarm.studio import reject_export_storage

    config = {"data_root": tmp_path, "studio_project": tmp_path / "native"}
    dataset = "ds_1234567890123456"
    root = queued_cards(config, dataset)
    trial = root / "done1"
    intake = trial / "motor-intake.json"
    original, replacement = tmp_path / "old-export", tmp_path / "new-export"
    choice = {"source": "native1", "scene_group_id": "block-1", "sample_count": 40}
    binding = {"path": str(original), "joint_units": ["degrees"], "source_frame_manifest_sha256": "ledger"}
    store_prefix(trial, choice, binding)
    atomic(intake, {"status": "reviewable_prefix", "segment": "motor-segment.json"})
    atomic(root / "done2/motor-intake.json", {"status": "excluded"})
    previous = sha(trial / "motor-segment.json")
    reject_export_storage(root / "_native-exports" / dataset, original, "before", "after")

    def save(_config, path, state, export, *, storage_digest):
        assert path == trial and load(intake)["status"] == "pending_review" and storage_digest == "verified"
        store_prefix(path, choice, {**binding, "path": str(export)})
        atomic(intake, {"status": "reviewable_prefix", "segment": "motor-segment.json"})
        return True

    monkeypatch.setattr(motor, "admitted", admitted_motor)
    monkeypatch.setattr(motor, "accepted_export", lambda *_: (replacement, {}))
    monkeypatch.setattr(motor, "export_storage_digest", lambda *_: "verified")
    monkeypatch.setattr(motor, "save_prefix", save)
    report = motor.review_pending_motor(config)
    assert not report["errors"] and report["attempted"] == 1
    assert sha(trial / "motor-segment-history" / f"{previous}.json") == previous
    assert load(trial / "motor-segment.json")["native_export"]["path"] == str(replacement)
    assert load(trial / "trial.json")["decision"] == {"saved": True}
    assert not motor.review_pending_motor(config)["attempted"]


def test_motor_recovery_refuses_a_changed_witnessed_range_or_contract(tmp_path):
    from smart_swarm.physical_motor_prefix import store_prefix

    choice = {"scene_group_id": "block-1", "episode_row_stop_exclusive": 40}
    binding = {"joint_units": ["degrees"], "live_calibration_snapshot_hashes": {"leader": "a", "follower": "b"}}
    store_prefix(tmp_path, choice, binding)
    previous = sha(tmp_path / "motor-segment.json")
    with pytest.raises(ValueError, match="witnessed placement range"):
        store_prefix(tmp_path, {**choice, "episode_row_stop_exclusive": 41}, binding)
    with pytest.raises(ValueError, match="calibration or action contract"):
        store_prefix(tmp_path, choice, {**binding, "joint_units": ["radians"]})
    assert sha(tmp_path / "motor-segment.json") == previous


def test_automatic_motor_review_waits_for_closed_block_and_idle_story(monkeypatch, tmp_path):
    from smart_swarm.physical_story_plan import prepare

    config = {"data_root": tmp_path, "studio_project": tmp_path / "native"}
    root = source.story_workspace(config)
    cards = prepare(config)["cards"]
    dataset = "ds_1234567890123456"
    for index, card in enumerate(cards[:7]):
        trial = root / card["story_id"]
        trial.mkdir()
        journal = trial / "trial.json"
        state = {"status": "complete", "studio_dataset_id": dataset}
        if index < 5:
            atomic(journal, state)
        elif index == 5:
            last = journal
        else:
            active = journal
    atomic(root / cards[0]["story_id"] / "motor-intake.json", {"status": "pending_review"})
    monkeypatch.setattr(motor, "review_sources", lambda *args: pytest.fail("Automatic export started while blocked"))
    assert motor.motor_review_generation(config) is None
    atomic(last, state)
    assert motor.motor_review_generation(config)
    atomic(active, {"status": "placement_recording"})
    assert motor.review_pending_motor(config, closed_blocks_only=True)["attempted"] == 0
    active.unlink()
    monkeypatch.setattr(motor, "ready_for_archive_review", lambda _: busy_recording())
    blocked = motor.review_pending_motor(config, closed_blocks_only=True)
    assert blocked["attempted"] == 0 and blocked["deferred_reason"] == "Native recording active"


def busy_recording():
    raise ProtocolError("RESOURCE_BUSY", "Native recording active")


def test_motor_retry_waits_for_cpu_and_bounds_export_failures():
    from smart_swarm.operator import intake_is_due, motor_retry_receipt

    generation = {"catalogues": {"ds": "a"}}
    denied = {"reservation_acquired": False, "errors": {"shared_reservation": "cpu_compute busy"}}
    pending = {"reservation_acquired": True, "errors": {}, "states": {"trial": {"status": "pending_review"}}}
    retry = motor_retry_receipt({}, generation, denied, 100)
    assert not intake_is_due(retry, generation, 159) and intake_is_due(retry, generation, 160)
    for instant in (160, 220, 280, 340):
        retry = motor_retry_receipt(retry, generation, denied, instant)
        assert retry["state"] == "retry_pending" and not retry["transient_failures"]
    for instant in (400, 410, 440, 500):
        retry = motor_retry_receipt(retry, generation, pending, instant)
    assert retry["state"] == "retry_exhausted"
    assert not intake_is_due(retry, generation, 999)
    assert intake_is_due(retry, {"catalogues": {"ds": "changed"}}, 999)


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


def test_reboot_in_mobile_phase_keeps_bound_placement_for_motor_review(monkeypatch, tmp_path):
    config = {"data_root": tmp_path}
    card = source.prepare(config)["cards"][0]
    trial = source.story_workspace(config) / card["story_id"]
    trial.mkdir()
    placement = {"observed_placement_outcome": "placement_succeeded"}
    receipt = {"episode_id": "original-arm-episode"}
    state = {"status": "recording", "card": card, "homebrain_boot_id": "earlier-boot",
             "markers": {"placement_closed": placement}, "staged_studio_recording": receipt}
    atomic(trial / "trial.json", state)
    calls = []

    def review(_, target, current, *, motor_only):
        assert target == trial and motor_only and current["staged_studio_recording"] == receipt
        calls.append(receipt)
        atomic(trial / "motor-intake.json", {"status": "pending_review"})

    monkeypatch.setattr(source, "homebrain_boot", lambda: "current-boot")
    monkeypatch.setattr(replay, "native_status", lambda *_: pytest.fail("Reboot recovery opened live observation"))
    monkeypatch.setattr(replay, "prepare_native_arm", review)
    result = replay.predict_selected(config, trial.name)
    saved = load(trial / "trial.json")
    assert saved["markers"] == {"placement_closed": placement}
    assert saved["status"] == "incomplete" and "rebooted" in result["reason"]
    assert saved["observed_terminal_outcome"] == "unknown"
    assert load(trial / "motor-intake.json")["status"] == "pending_review"
    assert replay.predict_selected(config, trial.name) == result and calls == [receipt]
