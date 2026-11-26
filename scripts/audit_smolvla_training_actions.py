"""Verify the official SmolVLA training targets behind the observed approach error."""

import argparse
from hashlib import sha256
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from smart_swarm.episode import atomic, load, sha
from smart_swarm.motor_input import CLOCKED_PROFILE, verify_issue_time
from smart_swarm.training import ROOT

EPISODES = 80
STEPS = 510
ISSUES = (14, 29, 44, 59)
VECTORS = ("action", "observation.state")
COLUMNS = (*VECTORS, "index", "episode_index", "frame_index", "timestamp")
PHASES = (("approach", 0, 60), ("descend", 60, 120), ("grasp", 120, 180), ("lift", 180, 240),
          ("transfer", 240, 330), ("lower", 330, 390), ("release", 390, 450), ("withdraw", 450, STEPS))


def exported_columns(train):
    shards = sorted((train / "data").glob("**/*.parquet"))
    tables = [pq.ParquetFile(path).read(columns=list(COLUMNS), use_threads=False) for path in shards]
    dtypes = {name: "<f4" if name in (*VECTORS, "timestamp") else "<i8" for name in COLUMNS}
    return {name: np.concatenate([np.asarray(table[name].to_pylist(), dtype=dtypes[name]) for table in tables])
            for name in COLUMNS}


def verified_source(lineage, index, actions, states, schema):
    path = Path(lineage["raw_path"])
    manifest = load(path / "manifest.json")
    if (lineage["dataset_episode_index"] != index or lineage["source_split"] != "train"
            or lineage["frame_count"] != STEPS or manifest["sample_count"] != STEPS
            or manifest["sha256"] != lineage["source_sha256"] or manifest["status"] != "accepted"
            or manifest["world_episode_id"] != lineage["source_world_episode_id"]
            or manifest["scene_group_id"] != lineage["scene_group_id"]
            or manifest["target_marker"] != lineage["target_marker"]
            or manifest["action_schema_id"] != schema["action_schema"]
            or manifest["names"] != schema["joint_order"] or manifest["units"] != schema["joint_units"]
            or not manifest["physics"]["released_stable_containment"]):
        raise ValueError(f"Training source {path.name} differs from its official lineage or command schema")
    with np.load(path / "motor.npz", allow_pickle=False) as archive:
        verify_issue_time(archive["timestamps_ns"], archive["applied_interval_start_ns"])
        for observed, raw_key in ((actions, "requested"), (states, "state")):
            expected = archive[raw_key]
            if expected.dtype != np.float32 or not np.array_equal(observed.view("<u4"), expected.view("<u4")):
                raise ValueError(f"Published {raw_key} differs from the original values in {path.name}")
    return {"source": lineage["source_world_episode_id"], "marker": lineage["target_marker"],
            "source_manifest_sha256": sha(path / "manifest.json")}


def training_actions(train, schema):
    lineage = load(train / "smart-swarm-lineage.json")
    info = load(train / "meta/info.json")
    sources = lineage["episodes"]
    total = EPISODES * STEPS
    names = schema["joint_order"]
    fields = [info["features"][name] for name in VECTORS]
    if (lineage["input_profile"] != CLOCKED_PROFILE or lineage["control_timing"] != "aligned-sample-v2"
            or (info["total_episodes"], info["total_frames"], info["fps"]) != (EPISODES, total, 30)
            or len(sources) != EPISODES or len({row["scene_group_id"] for row in sources}) != EPISODES
            or any(row["names"] != names or row["dtype"] != "float32" for row in fields)):
        raise ValueError("The audit requires the published 80-source clocked split and ordered arm schema")
    rows = exported_columns(train)
    if (not np.array_equal(rows["index"], np.arange(total))
            or not np.array_equal(rows["episode_index"], np.repeat(np.arange(EPISODES), STEPS))
            or not np.array_equal(rows["frame_index"], np.tile(np.arange(STEPS), EPISODES))
            or not np.array_equal(rows["timestamp"], (rows["frame_index"] / 30).astype("<f4"))):
        raise ValueError("Native export indices or relative observation clocks are misaligned")
    actions, states = (rows[name].reshape(EPISODES, STEPS, len(names)) for name in VECTORS)
    proofs = [verified_source(row, i, actions[i], states[i], schema) for i, row in enumerate(sources)]
    return actions, states, proofs


def pan_summary(values):
    return {"rows": int(values.size), "mean_deg": float(values.mean(dtype=np.float64)),
            "min_deg": float(values.min()), "max_deg": float(values.max()),
            "positive": int((values > 0).sum()), "negative": int((values < 0).sum())}


def verified_comparison(approach, actions, source_proofs):
    cases = approach["case_rows"]
    identities = {case["seed"] for case in cases}
    if (len(cases) != 20 or len(identities) != 20 or approach["fresh_resets_read"] != 0
            or any([sample["issue"] for sample in case["samples"]] != list(ISSUES) for case in cases)):
        raise ValueError("The comparison requires the archived full matched validation approach audit")
    groups = {mark: [i for i, row in enumerate(source_proofs) if row["marker"] == mark]
              for mark in ("cyan", "violet")}
    if not all(groups.values()) or sum(map(len, groups.values())) != EPISODES:
        raise ValueError("The published training split must contain both admitted target markers")
    result = {}
    for issue in ISSUES:
        compared = [sample for case in cases for sample in case["samples"] if sample["issue"] == issue]
        result[str(issue)] = {"training": {mark: pan_summary(actions[indices, issue, 0])
                                         for mark, indices in groups.items()},
                              "validation_teacher": pan_summary(np.array([s["teacher_shoulder_pan_deg"] for s in compared])),
                              "validation_policy": pan_summary(np.array([s["policy_shoulder_pan_deg"] for s in compared]))}
    return result


def audit(approach_path):
    approach = load(approach_path)
    checkpoint = Path(approach["checkpoint"])
    source = load(checkpoint / "smart-swarm-input.json")
    train = Path(source["dataset"]) / "train"
    stats_path = train / "meta/stats.json"
    if (approach["training_statistics_sha256"] != sha(stats_path)
            or approach["checkpoint_config_sha256"] != sha(checkpoint / "config.json")):
        raise ValueError("The published training statistics or selected checkpoint binding changed")
    schema = load(ROOT / "handover/act-clocked-execution-handoff.json")["manifest"]
    actions, states, sources = training_actions(train, schema)
    stats = load(stats_path)["action"]
    overall = pan_summary(actions[:, :, 0])
    overall.update(stored_normalization_mean_deg=stats["mean"][0], stored_normalization_std_deg=stats["std"][0],
                   stored_normalization_count=stats["count"][0], measured_std_deg=float(actions[:, :, 0].std(dtype=np.float64)))
    return {"scope": "published training labels and archived matched validation commands", "training_split": str(train),
            "checkpoint": str(checkpoint), "training_sources": sources, "action_units": schema["joint_units"],
            "frame_count": actions.shape[0] * actions.shape[1], "action_schema": schema["joint_order"],
            "exported_actions_sha256": sha256(actions.tobytes()).hexdigest(),
            "exported_states_sha256": sha256(states.tobytes()).hexdigest(),
            "training_lineage_sha256": sha(train / "smart-swarm-lineage.json"), "training_statistics_sha256": sha(stats_path),
            "validation_approach_sha256": sha(approach_path), "training_shoulder_pan": overall,
            "training_shoulder_pan_by_phase": {phase: pan_summary(actions[:, begin:end, 0]) for phase, begin, end in PHASES},
            "matched_approach_issues": verified_comparison(approach, actions, sources),
            "additional_optimizer_updates": 0, "fresh_resets_read": 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--approach", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Choose a new immutable audit output")
    atomic(args.output, audit(args.approach.resolve()))
    print(args.output)


if __name__ == "__main__":
    main()
