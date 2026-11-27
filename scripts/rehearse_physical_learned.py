#!/usr/bin/env python3
"""Exercise the pending physical inference path with archives and simulation controls."""

import argparse
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from compare_primary_supervision import score_paths
from rehearse_physical_story_recovery import NATIVE_FIXTURE, PILOT
from smart_swarm.cli import settings
from smart_swarm.contract import ProtocolError, interval, time_at
from smart_swarm.episode import POLICY, TRUTH, atomic, load, rows
from smart_swarm.physical_fleet_timing import PhysicalStoryRecorder
from smart_swarm.physical_learned import (DEFERRED, HEADS, METHOD, blind_inference, inference, offline_seal, packets, run_worker, scoped_window,
                                          verify_optional)
from smart_swarm.physical_offline import FILE as OFFLINE_FILE, observe_pending, score_trial
from smart_swarm.physical_story_replay import AGENTS, fleet_clips, studio_clips, validate_pending_decision, write_decision
from smart_swarm.physical_story_source import fleet_manifest, learned_story_progress
from smart_swarm.research_jobs import admitted
from smart_swarm.studio import validate_story_episode


def forecast(source, manifest):
    with scoped_window(source, manifest) as (scoped, digest):
        prediction = run_worker(scoped, digest)
    if verify_optional(source, manifest, {"learned_evidence": {"status": "scored", "receipt": prediction}})["receipt"] != prediction:
        raise ValueError("The pre-reveal receipt did not reopen from the source packet digest")
    return prediction


def preserve_prediction(recorder, state, arm, tank, through):
    query = through["monotonic_ns"]
    manifest = {**recorder.meta, "sample_counts": recorder.counts, "query_boundary": time_at(query, recorder.clock)}
    learned = forecast(recorder.root, manifest)
    record = write_decision(recorder, arm, tank, query, query + 1_000_000_000, {"status": "scored", "receipt": learned})
    checked, issued = validate_pending_decision(recorder, recorder.root, state)
    if checked != record or issued != query or any((recorder.root / TRUTH).iterdir()):
        raise ValueError("The learned archive prediction was not saved before evaluator input")
    return learned, manifest


def native_case(config, root, proof, source, family, *, blind=False, busy_online=False):
    identifier = load(PILOT / "physical-story-projection.json")["source"]
    fleet = fleet_manifest(config, identifier)
    epoch = round(fleet["created_at"] * 1_000_000_000)
    times = [entry["observation_timestamp_ns"] for entry in rows(source / "frames.jsonl")]
    offset = epoch + 5_000_000_000 - max(times)

    def at(seconds):
        wall = epoch + seconds * 1_000_000_000
        return {"monotonic_ns": wall - offset, "unix_ns": wall}

    clock = {"host_id": "archive-rehearsal", "boot_id": identifier, "kind": "monotonic"}
    cameras = {"so101-1": [] if family == "F6" else ["so101-1/wrist", "so101-1/top"],
               **{agent: [agent + "/front"] for agent in AGENTS[1:]}}
    state = {"fleet_capture_id": identifier, "card": {"family": family}, "native_story_validation": proof,
             "start": {"monotonic_ns": min(times), "unix_ns": min(times) + offset},
             "markers": {"placement_closed": at(5), "relocation_closed": at(17)}}
    provenance = {"run_mode": "hardware", "studio_episode": proof["episode_id"], "fleet_capture": identifier,
                  "native_recording_clock": clock}
    recorder = PhysicalStoryRecorder(root, family, "technical", cameras, provenance, clock=clock)
    arm = studio_clips(recorder, source, state)
    audit = {}
    through = at(21)
    tank, receiver = AGENTS[1:]
    clips = fleet_clips(config, recorder, state, tank, through["monotonic_ns"], fleet, audit, current_anchor=through)
    fleet_clips(config, recorder, state, receiver, through["monotonic_ns"], fleet, audit, current_anchor=through)
    if blind or busy_online:
        return rehearse_blind(config, recorder, state, arm, clips, through["monotonic_ns"], busy_online=busy_online)
    learned, manifest = preserve_prediction(recorder, state, arm, clips, through)
    if family == "F6":
        arm, = packets(root, manifest)[AGENTS[0]]
        if arm["cameras"] or arm["measured_state"] or arm["applied_command_ref"] is not None:
            raise ValueError("The withheld arm leaked private inputs to the F6 projection")
        if any(row["decision"] != "request_inspection" for row in learned["heads"].values()):
            raise ValueError("The archived withheld physical stream did not request inspection")
    return {"role_packet_counts": recorder.counts, "native_explorer_capture": identifier,
            "native_studio_episode": proof["episode_id"], "technical_only": proof["technical_only"],
            "prediction_saved_and_reopened_before_reveal": True, "prediction": learned}


def denied_online_seal(config, recorder, query):
    with admitted(config, "smart-swarm-fallback-lease-fixture", archive_only=True) as (_, heartbeat):
        seal = inference(config, recorder, query)
        heartbeat()
    if seal["status"] != DEFERRED or "cpu_compute" not in seal.get("online_attempt", {}).get("reason", ""):
        raise ValueError("The native busy CPU did not save an online admission failure and a valid offline seal")
    return seal


def score_queued_fixture(config, trial, state, prediction):
    journal = {**state, "status": "complete", "decision": prediction, "evaluation_mode": "online"}
    atomic(trial / "trial.json", journal)
    if observe_pending(config) != trial.name:
        raise ValueError("The background queue did not find the native fallback input")
    status = learned_story_progress(trial, journal)
    if status["learned_evidence_status"] != "scored" or status["learned_forecast_mode"] != "blind_offline":
        raise ValueError("Story progress did not report the scored blind forecast")
    return score_trial(config, trial), {"queued_automatically": True, "reported_progress": status}


def rehearse_blind(config, recorder, state, arm, tank, query, *, busy_online=False):
    root = recorder.root
    trial = root.parent
    prediction_at, observed_at = query + 1_000_000_000, query + 2_000_000_000
    manifest = {**recorder.meta, "sample_counts": recorder.counts, "query_boundary": time_at(query, recorder.clock)}
    seal = denied_online_seal(config, recorder, query) if busy_online else offline_seal(root, manifest)
    prediction = write_decision(recorder, arm, tank, query, prediction_at, seal)
    if validate_pending_decision(recorder, trial, state) != (prediction, query) or list((root / TRUTH).iterdir()):
        raise ValueError("The archived private seal was not independently verified before reveal")
    camera = recorder.frame(TRUTH, "technical-evaluator", 1, np.zeros((4, 4, 3), dtype=np.uint8), observed_at)
    reveal = {"camera": camera, "observed_at": time_at(observed_at, recorder.clock), "ground_truth": "left",
              "target_state_time_or_interval": interval(prediction_at, observed_at, recorder.clock),
              "relations_preserved_until_reveal": True}
    recorder.append(TRUTH, "reveal", reveal)
    recorder.close(query, prediction_at, observed_at)
    saved, queue = score_queued_fixture(config, trial, state, prediction) if busy_online else (score_trial(config, trial), {})
    forecast_bytes = (trial / OFFLINE_FILE).read_bytes()
    if score_trial(config, trial) != saved or (trial / OFFLINE_FILE).read_bytes() != forecast_bytes:
        raise ValueError("Reopening the same recorded inputs changed the offline forecast")
    atomic(root / TRUTH / "reveal.jsonl", {**reveal, "ground_truth": "surface"})
    changed = blind_inference(config, root, manifest, seal)
    if changed["receipt"] != saved["receipt"] or any(
            head["decision"] != "request_inspection" for head in saved["receipt"]["heads"].values()):
        raise ValueError("The blind archive forecast changed under an evaluator-only substitution")
    return {"seal": seal, "forecast": saved, "reopened_unchanged": True, "technical_evaluator_substitution": ["left", "surface"],
            "role_packet_counts": recorder.counts, "native_explorer_capture": state["fleet_capture_id"],
            "new_physical_attempts": 0, "optimizer_updates": 0, **queue}


def simulation_cases(heartbeat):
    archive = {kind: load(path)["outcomes"] for kind, path in score_paths("validation").items()}
    selections = {}
    for family in ("F1", "F6"):
        expected = {}
        for kind in HEADS:
            group = next(row["outcomes"] for row in archive[kind]
                         if row["team_size"] == 3 and row["outcomes"][0]["family"] == family)
            expected[kind] = {row["source"]: row for row in group}
        cases = []
        for source, row in expected[HEADS[0]].items():
            heartbeat()
            path = Path(source)
            manifest = load(path / "manifest.json")
            result = forecast(path, manifest)
            for kind, prediction in result["heads"].items():
                saved = expected[kind][source][METHOD]
                error = max(abs(value - saved["belief"][slot]) for slot, value in prediction["belief"].items())
                if error > 2e-6 or saved["decision"] != prediction["decision"] or saved["wire"] != prediction["wire"]:
                    raise ValueError("The physical learned path disagrees with a proven simulator replay")
            cases.append({"source": source, "terminal_truth_for_scoring": row["truth"], "prediction": result})
        if family == "F1" and any(head["decision"] != cases[0]["terminal_truth_for_scoring"]
                                 for head in cases[0]["prediction"]["heads"].values()):
            raise ValueError("The resolvable control requires a correct container decision")
        if family == "F6":
            first, second = cases
            if first["prediction"] != second["prediction"] or first["terminal_truth_for_scoring"] == second["terminal_truth_for_scoring"]:
                raise ValueError("The same permitted F6 window changed its pre-inspection prediction")
        selections[family] = cases
    return selections


def rehearse(config, grant, heartbeat):
    source = config["studio_project"] / NATIVE_FIXTURE
    heartbeat()
    proof = validate_story_episode(config, source, technical=True)
    with TemporaryDirectory(prefix="smart-swarm-frozen-native-rehearsal-") as directory:
        archive = {}
        for family in ("F1", "F6"):
            heartbeat()
            archive[family] = native_case(config, Path(directory) / family, proof, source, family)
    simulation = simulation_cases(heartbeat)
    heartbeat()
    return {"scope": "Native technical archives and frozen MuJoCo calibration controls", "new_physical_attempts": 0,
            "optimizer_updates": 0, "native": archive, "simulation": simulation, "role_namespace": POLICY,
            "compute_reservation": grant}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--blind-offline-only", action="store_true", help="Test the blind replay mode on the native F6 archive")
    mode.add_argument("--busy-online-only", action="store_true", help="Test an occupied native online lease and the deferred forecast queue")
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Choose a new learned physical rehearsal receipt")
    config = settings()
    try:
        if args.blind_offline_only or args.busy_online_only:
            source = config["studio_project"] / NATIVE_FIXTURE
            proof = validate_story_episode(config, source, technical=True)
            with TemporaryDirectory(prefix="smart-swarm-blind-native-rehearsal-") as directory:
                scratch = {**config, "data_root": Path(directory)}
                trial = scratch["data_root"] / "physical-stories" / "F6"
                result = native_case(scratch, trial / "private-replay", proof, source, "F6", blind=True,
                                     busy_online=args.busy_online_only)
        else:
            with admitted(config, "smart-swarm-physical-evidence-rehearsal", archive_only=True) as (grant, heartbeat):
                result = rehearse(config, grant, heartbeat)
    except (ProtocolError, RuntimeError) as error:
        parser.exit(2, f"Native learned rehearsal stopped. {error}\n")
    atomic(args.output, result)
    print(args.output)


if __name__ == "__main__":
    main()
