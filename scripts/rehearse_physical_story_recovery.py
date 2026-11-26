#!/usr/bin/env python3
"""Rehearse interrupted story intake on archived native sources."""

import argparse
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter

from smart_swarm.cli import settings
from smart_swarm.episode import Recorder, atomic, load, rows, sha, validate_agent
from smart_swarm.fleet_adapter import FleetArchive, archive_rehearsal
from smart_swarm.physical_fleet_timing import FleetClock, PhysicalStoryRecorder, eligible_frames
from smart_swarm.physical_story_operator import physical_steps
from smart_swarm.physical_story_plan import prepare
from smart_swarm.physical_story_replay import boot_clock, fleet_clips, predict_selected, studio_clips
from smart_swarm.physical_story_source import fleet_manifest, story_workspace
from smart_swarm.studio import validate_story_episode

PILOT = Path(__file__).resolve().parents[1] / "evidence/learning-pilot-v3"


def ended_window(config):
    source = load(PILOT / "physical-story-projection.json")["source"]
    native = fleet_manifest(config, source)
    if native["phase"] != "complete" or not native.get("readback", {}).get("passed"):
        raise ValueError("The original Explorer archive must retain its native readback")
    cameras = archive_rehearsal(config, source)
    card = prepare(config)["cards"][0]
    trial = story_workspace(config) / card["story_id"]
    trial.mkdir()
    state = {"status": "recording", "card": card, "fleet_capture_id": source, "markers": {}}
    atomic(trial / "trial.json", state)
    if physical_steps(config, trial.name, state) != state:
        raise ValueError("The operator path wrote a new phase after the recording ended")
    first = predict_selected(config, trial.name)
    second = predict_selected(config, trial.name)
    stored = load(trial / "trial.json")
    exclusion = load(trial / "motor-intake.json")
    if (first != second or stored["status"] != "incomplete" or stored["observed_terminal_outcome"] != "unknown"
            or first["motor_segment_admitted"] or "decision" in stored or exclusion["status"] != "excluded"):
        raise ValueError("The expired native archive was promoted or the readback was not idempotent")
    return {"archived_capture": source, "native_phase": native["phase"],
            "private_camera_readback": cameras, "operator_state_unchanged_after_native_close": True,
            "story_result": first, "motor_intake": exclusion}


def withheld_fixture(config):
    receipt = load(PILOT / "native-placement-prefix-check.json")["native_export"]
    meta = load(Path(receipt["path"]) / "provenance/export.json")
    if meta["source"] != "synthetic_test" or meta["test_only"] is not True:
        raise ValueError("Use the existing Studio technical fixture for this rehearsal")
    relative = "episodes/ep_1/frames.jsonl"
    original = Path(meta["source_path"]) / relative
    digest = sha(original)
    if digest != meta["source_hashes"][relative]:
        raise ValueError("The existing native frame manifest changed")
    native = rows(original)
    times = [entry["observation_timestamp_ns"] for entry in native]
    fixture = {"card": {"family": "F6"}, "start": {"monotonic_ns": min(times) - 1},
               "markers": {"placement_closed": {"monotonic_ns": max(times)}}}
    clock = boot_clock()
    recorder = Recorder(config["data_root"] / "withheld-technical-view", "fixture", "technical",
                        {"so101-1": []}, {"run_mode": "hardware", "native_recording_clock": clock}, clock=clock)
    clips = studio_clips(recorder, original.parent, fixture)
    (packet, views), = clips
    if not native[0]["observation.state"] or views or packet["cameras"] or packet["measured_state"] or packet["state_units"]:
        raise ValueError("A withheld technical projection retained raw arm state or camera inputs")
    if int(packet["source_time"]["monotonic_ns"]) != fixture["start"]["monotonic_ns"]:
        raise ValueError("The empty arm packet retained a timestamp from the withheld native archive")
    return {"native_fixture_sha256": digest, "raw_state_dimension": len(native[0]["observation.state"]),
            "runtime_camera_count": len(packet["cameras"]), "runtime_state_dimension": len(packet["measured_state"]),
            "runtime_timestamp_from_story_start": True}


def validated_native_fixture(config):
    episode = config["studio_project"] / config["studio_recovery_fixture"]
    native = rows(episode / "frames.jsonl")
    start = perf_counter()
    proof = validate_story_episode(config, episode, technical=True)
    duration = perf_counter() - start
    try:
        validate_story_episode(config, episode)
    except ValueError as error:
        production_rejection = str(error)
    else:
        raise ValueError("An archived technical episode passed the physical source gate")
    timestamps = [entry["observation_timestamp_ns"] for entry in native]
    state = {"card": {"family": "F1"}, "start": {"monotonic_ns": min(timestamps)},
             "markers": {"placement_closed": {"monotonic_ns": max(timestamps)}}, "native_story_validation": proof}
    clock = boot_clock()
    cameras = ["so101-1/wrist", "so101-1/top"]
    recorder = Recorder(config["data_root"] / "native-episode-technical", "technical-episode", "technical",
                        {"so101-1": cameras}, {"run_mode": "hardware", "native_recording_clock": clock}, clock=clock)
    clips = studio_clips(recorder, episode, state)
    if len(clips) != 8 or any(list(views) != cameras or len(packet["measured_state"]) != 6 for packet, views in clips):
        raise ValueError("The archived episode failed the ordered private projection")
    if (sha(episode / "frames.jsonl") != proof["frames_sha256"]
            or sha(episode / "episode.json") != proof["episode_metadata_sha256"]
            or sha(episode.parent.parent / "dataset.json") != proof["dataset_manifest_sha256"]):
        raise ValueError("Archived native source metadata changed during readback")
    return {"native_episode": str(episode), "software_only": True, "validation_seconds": duration, "validation": proof,
            "private_packets": len(clips), "ordered_cameras": cameras,
            "production_rejection": production_rejection, "whole_dataset_export_started": False}


def archive_phase_projection(config, *, staged=False):
    identifier = load(PILOT / "physical-story-projection.json")["source"]
    fleet = fleet_manifest(config, identifier)
    epoch = round(fleet["created_at"] * 1e9)
    origin = 120_000_000_000 if staged else 0

    def at(seconds):
        tick = seconds * 1_000_000_000
        return {"monotonic_ns": origin + tick, "unix_ns": epoch + tick}

    state = {"fleet_capture_id": identifier, "start": at(1),
             "markers": {"placement_closed": at(5), "relocation_closed": at(17)}}
    if staged:
        state.update(start=at(-110), fleet_launch_requested=at(4), fleet_acknowledgement=at(5), fleet_window_s=60)
        state["markers"]["placement_closed"] = at(-85)
    agents = ("tank-1", "rosmaster-1")
    cameras = {agent: [agent + "/front"] for agent in agents}
    clock = {"host_id": "technical-native-archive", "boot_id": identifier, "kind": "monotonic"}
    recorder = PhysicalStoryRecorder(config["data_root"] / "archived-phase-projection", "clock-archive", "technical",
                                    cameras, {"run_mode": "hardware"}, clock=clock)
    audit = {}
    for agent in agents:
        fleet_clips(config, recorder, state, agent, at(21)["monotonic_ns"], fleet, audit, current_anchor=at(21))
        validate_agent(recorder.root, {**recorder.meta, "sample_counts": recorder.counts}, agent, cameras[agent],
                       at(21)["monotonic_ns"], clock)
    if recorder.counts != {"tank-1": 8, "rosmaster-1": 1}:
        raise ValueError("The archived placement and receiver projections did not satisfy their view quotas")
    if staged:
        from smart_swarm.physical_evidence_dataset import verify_fleet_windows

        atomic(config["data_root"] / "fleet-timing-audit.json", audit)
        verify_fleet_windows(config["data_root"], recorder.root, {**state, "query_ns": at(21)["monotonic_ns"]})
    report = {"native_capture_id": identifier, "scope": "actual native RGB and clock rows on an archive-relative technical timeline",
              "phase_audit": audit, "output_frames": recorder.counts, "clock": clock,
              "boundary_case": archive_boundary_case(config, identifier, fleet, epoch), "physical_outcome_labeled": False}
    if staged:
        report.update(timeline={key: value for key, value in state.items() if key != "fleet_capture_id"}, corpus_timing_verified=True)
    return report


def archive_boundary_case(config, identifier, fleet, epoch):
    archive = FleetArchive(config, identifier, "tank-1")
    frame = archive.packets(through_s=60)[0]
    begin = frame["source_unix_ns"] - epoch - 1
    stop = frame["received_unix_ns"] - epoch + 1
    clock = FleetClock(archive.role, fleet["initial_hub_status"], archive.clock_history())
    accepted, details = eligible_frames([frame], clock, epoch, (begin, stop))
    reason = "full RGB interval and receipt must fit the permitted phase"
    if accepted or details["excluded_reasons"] != {reason: 1}:
        raise ValueError("A real native frame crossing the phase boundary was admitted")
    return {"original_frame": frame["frame_ref"], "nominal_timestamp_inside": True,
            "phase_overlap_rejection": reason, "clock_radius_ns_including_home_margin": clock.radius(frame["received_unix_ns"])}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--staged-mobile-only", action="store_true", help="Test the relocated mobile phase on the native archive")
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Choose a new rehearsal report path")
    with TemporaryDirectory(prefix="smart-swarm-physical-recovery-") as directory:
        config = {**settings(), "data_root": Path(directory)}
        if args.staged_mobile_only:
            result = {"projection": archive_phase_projection(config, staged=True), "new_physical_attempts": 0}
        else:
            result = {"scope": "Native archived observation and test-only Studio fixture",
                      "expired_recording": ended_window(config), "withheld_projection": withheld_fixture(config),
                      "one_episode_validation_and_projection": validated_native_fixture(config),
                      "clock_bracket_projection": archive_phase_projection(config),
                      "new_physical_attempts": 0}
    atomic(args.output, result)
    print(args.output)


if __name__ == "__main__":
    main()
