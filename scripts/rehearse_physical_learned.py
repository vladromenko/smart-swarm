#!/usr/bin/env python3
"""Exercise the pending physical inference path with archives and simulation controls."""

import argparse
from pathlib import Path
from tempfile import TemporaryDirectory

from compare_primary_supervision import score_paths
from rehearse_physical_story_recovery import PILOT, native_fixture
from smart_swarm.cli import settings
from smart_swarm.contract import time_at
from smart_swarm.episode import POLICY, TRUTH, atomic, load, rows
from smart_swarm.physical_fleet_timing import PhysicalStoryRecorder
from smart_swarm.physical_learned import HEADS, METHOD, packets, run_worker, scoped_window, verify_optional
from smart_swarm.physical_story_replay import AGENTS, fleet_clips, studio_clips, validate_pending_decision, write_decision
from smart_swarm.physical_story_source import fleet_manifest
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


def native_case(config, root, proof, source, family):
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
    provenance = {"run_mode": "hardware", "studio_episode": proof["episode_id"], "fleet_capture": identifier}
    recorder = PhysicalStoryRecorder(root, family, "technical", cameras, provenance, clock=clock)
    arm = studio_clips(recorder, source, state)
    audit = {}
    through = at(21)
    tank, receiver = AGENTS[1:]
    clips = fleet_clips(config, recorder, state, tank, through["monotonic_ns"], fleet, audit, current_anchor=through)
    fleet_clips(config, recorder, state, receiver, through["monotonic_ns"], fleet, audit, current_anchor=through)
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


def simulation_cases():
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Choose a new learned physical rehearsal receipt")
    config = settings()
    source = native_fixture(config)
    proof = validate_story_episode(config, source, technical=True)
    with TemporaryDirectory(prefix="smart-swarm-frozen-native-rehearsal-") as directory:
        archive = {family: native_case(config, Path(directory) / family, proof, source, family) for family in ("F1", "F6")}
    result = {"scope": "Native technical archives and frozen MuJoCo calibration controls", "new_physical_attempts": 0,
              "optimizer_updates": 0, "native": archive, "simulation": simulation_cases(),
              "role_namespace": POLICY}
    atomic(args.output, result)
    print(args.output)


if __name__ == "__main__":
    main()
