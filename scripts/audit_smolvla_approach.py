"""Audit the first five stored validation rollouts with their original teacher traces."""

import argparse
from hashlib import sha256
from pathlib import Path

import numpy as np
from safetensors.numpy import load_file

from smart_swarm.episode import atomic, json_bytes, load, sha


def processor_statistics(checkpoint, stats_file):
    config = load(checkpoint / "config.json")
    features = ["observation.state", "observation.images.camera1", "observation.images.camera2"]
    normalization = {"VISUAL": "IDENTITY", "STATE": "MEAN_STD", "ACTION": "MEAN_STD"}
    if config["type"] != "smolvla" or list(config["input_features"]) != features or config["normalization_mapping"] != normalization:
        raise ValueError("The checkpoint differs from the declared compact SmolVLA input profile")
    stats = load(stats_file)
    expected = {f"{name}.{key}": value for name, row in stats.items() for key, value in row.items()}
    verified = {}
    for path in sorted(checkpoint.glob("*processor*.safetensors")):
        observed = load_file(path)
        if set(observed) != set(expected):
            raise ValueError(f"Stored normalization keys differ from the training split for {path.name}")
        reshaped = []
        for name, values in observed.items():
            reference = np.asarray(expected[name], dtype=values.dtype)
            if values.shape != reference.shape:
                if name not in {f"{key}.count" for key in features[1:]} or values.size != reference.size:
                    raise ValueError(f"Unexpected normalization shape for {name} in {path.name}")
                reshaped.append(name)
                reference = reference.reshape(values.shape)
            if not np.array_equal(values, reference):
                raise ValueError(f"Normalization values for {name} differ from the training split")
        verified[path.name] = {"sha256": sha(path), "exact_training_statistics": len(observed),
                               "camera_count_shape_changes": reshaped}
    if len(verified) != 2:
        raise ValueError("Expected the two official SmolVLA processor states")
    return verified


def matched_feedback(row, lineage):
    path = Path(lineage["raw_path"])
    manifest = load(path / "manifest.json")
    if (manifest["seed"] != row["seed"] or manifest["sha256"] != lineage["source_sha256"]
            or manifest["names"][0] != "shoulder_pan" or manifest["units"][0] != "deg"):
        raise ValueError(f"Validation reset {row['seed']} differs from its declared teacher source or action order")
    feedback = row["feedback"]
    if [sample["tick"] for sample in feedback] != list(range(15, 511, 15)):
        raise ValueError(f"Validation reset {row['seed']} has incomplete contact feedback")
    indices = np.array([sample["tick"] - 1 for sample in feedback])
    with np.load(path / "motor.npz", allow_pickle=False) as archive:
        times, requested = archive["timestamps_ns"][indices], archive["requested"][indices]
    observed_times = np.array([sample["observed_at_ns"] for sample in feedback])
    teacher_targets = np.array([sample["teacher_requested_at_index"] for sample in feedback], dtype=requested.dtype)
    if not np.array_equal(times, observed_times) or not np.array_equal(requested, teacher_targets):
        raise ValueError(f"Validation reset {row['seed']} differs from the original teacher issue clocks or commands")
    first = feedback[0]
    clipped = np.abs(np.asarray(first["decoded_and_clipped"]) - first["requested"])
    return {"seed": row["seed"], "marker": lineage["target_marker"], "retained_contact_samples": len(feedback),
            "feedback_row_sha256": sha256(json_bytes(row)).hexdigest(), "teacher_artifact_sha256": manifest["sha256"],
            "first_saved_issue": int(indices[0]), "first_failed_stage": row["first_failed_stage"],
            "shoulder_pan_requested_deg": first["requested"][0],
            "shoulder_pan_after_conversion_deg": first["decoded_and_clipped"][0],
            "teacher_shoulder_pan_requested_deg": first["teacher_requested_at_index"][0],
            "max_conversion_difference_at_first_issue": float(clipped.max()),
            "closest_gripper_to_cube_m": row["minimum_tcp_cube_distance_m"], "bilateral_contact": row["bilateral_contact"]}


def audit(validation):
    job = validation.parents[2]
    checkpoint = job / "training/checkpoints" / validation.parent.name / "pretrained_model"
    dataset = Path(load(checkpoint / "smart-swarm-input.json")["dataset"])
    stats = dataset / "train/meta/stats.json"
    receipt = load(validation)
    rows = receipt["resets"][:5]
    if receipt["evaluation_role"] != "validation" or len(rows) != 5:
        raise ValueError("Five saved validation rollouts are required")
    sources = load(dataset / "validation/smart-swarm-lineage.json")["episodes"][:5]
    cases = [matched_feedback(row, source) for row, source in zip(rows, sources, strict=True)]
    return {"scope": "read-only diagnostic of the first five published validation sources", "checkpoint": str(checkpoint),
            "checkpoint_config_sha256": sha(checkpoint / "config.json"), "training_statistics_sha256": sha(stats),
            "processor_proofs": processor_statistics(checkpoint, stats), "validation_receipt": str(validation),
            "case_rows": cases, "additional_optimizer_updates": 0, "fresh_resets_read": 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("The diagnostic output already exists")
    atomic(args.output, audit(args.validation.resolve()))
    print(args.output)


if __name__ == "__main__":
    main()
