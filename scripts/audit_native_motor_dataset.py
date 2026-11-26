#!/usr/bin/env python3
"""Exercise the combined prefix reader on the existing official Studio test export."""

import argparse
from pathlib import Path
from shutil import copytree
from tempfile import TemporaryDirectory

import torch
from torch.utils.data import DataLoader
from lerobot.datasets.lerobot_dataset import LeRobotDataset

from consume_studio_export import verify
from smart_swarm.episode import atomic, load, rows, sha
from smart_swarm.physical_motor_dataset import CAMERA_KEYS, PhysicalMotorDataset, corpus_record
from smart_swarm.physical_motor_prefix import NativePlacement, official_prefix


def technical_prefix(export, source, item):
    episode = source / "episodes" / item["episode_id"]
    frames = rows(episode / "frames.jsonl")
    return {"scene_group_id": "technical-rehearsal-session", "split": "development", "sample_count": 2,
            "frame_index_start_inclusive": frames[0]["frame_id"], "frame_index_end_inclusive": frames[1]["frame_id"],
            "native_export": official_prefix(export, episode, 0, 2, technical=True)}


def check_batch(reader, data, prefixes):
    batch = next(iter(DataLoader(reader, batch_size=len(reader))))
    expected_masks = [[False, False, True], [False, True, True]] * len(prefixes)
    if batch["action_is_pad"].tolist() != expected_masks:
        raise ValueError("The batched action window escaped a native placement boundary")
    original_rows = []
    for prefix in prefixes:
        binding = prefix["native_export"]
        start = binding["dataset_row_start"]
        original_rows.extend((data[start], data[start + 1]))
        if torch.equal(original_rows[-1]["action"], data[start + 2]["action"]):
            raise ValueError("The excluded native continuation needs a distinct action")
    for index, row in enumerate(original_rows):
        expected = [original_rows[index + offset]["action"] for offset in range(2 - index % 2)]
        expected = torch.stack(expected + [expected[-1]] * (3 - len(expected)))
        if (not torch.equal(batch["action"][index], expected)
                or not torch.equal(batch["observation.state"][index], row["observation.state"])
                or batch["task"][index] != row["task"]
                or any(not torch.equal(batch[key][index], row[key].float() / 255.) for key in CAMERA_KEYS)):
            raise ValueError("The combined loader changed native observation or action values")
    return {"model_batch_fields": sorted(batch), "action_shape": list(batch["action"].shape),
            "action_pad_masks": expected_masks, "rgb_shapes": {key: list(batch[key].shape) for key in CAMERA_KEYS},
            "excluded_continuations_distinguishable": True, "matches_studio_native_values": True}


def check_rejections(record):
    split = {**record["prefixes"][-1], "split": "validation"}
    checks = {"technical_record_for_physical_loader": lambda: PhysicalMotorDataset(record, "development", 3),
              "same_episode_twice": lambda: corpus_record([record["prefixes"][0]] * 2, technical=True),
              "same_scene_across_split": lambda: corpus_record([record["prefixes"][0], split], technical=True)}
    for name, run in checks.items():
        try:
            run()
        except ValueError:
            pass
        else:
            raise ValueError(f"Native placement rejection did not occur for {name}")
    return sorted(checks)


def check_storage_rejection(export, prefix):
    with TemporaryDirectory(prefix="smart-swarm-native-motor-check-") as directory:
        duplicate = Path(directory) / "export"
        copytree(export, duplicate)
        parquet = next((duplicate / "data").rglob("*.parquet"))
        with parquet.open("ab") as stream:
            stream.write(b"altered technical copy")
        binding = {**prefix["native_export"], "path": str(duplicate)}
        try:
            NativePlacement({**prefix, "native_export": binding}, technical=True)
        except ValueError as error:
            if "storage" not in str(error) and "bytes changed" not in str(error):
                raise
        else:
            raise ValueError("The prefix reader did not reject changed export storage bytes")


def check_interior_row_rejections(export, source):
    import pyarrow as pa
    import pyarrow.parquet as pq

    parquet, = sorted((export / "data").rglob("*.parquet"))
    table = pq.ParquetFile(parquet).read()
    if table.num_rows < 3 or len(set(table["episode_index"].to_pylist()[:3])) != 1:
        raise ValueError("The technical export must provide an interior row")
    changes = {"action": lambda v: [v[0] + .125, *v[1:]], "raw.wrist.timestamp_ns": lambda v: v - 1,
               "raw.action_valid": lambda v: not v, "raw.action_acknowledged": lambda v: not v,
               "timestamp": lambda v: v + .001}
    rejection = {}
    with TemporaryDirectory(prefix="smart-swarm-native-interior-rows-") as directory:
        for column, modify in changes.items():
            copy = Path(directory) / column
            copytree(export, copy)
            values = table[column].to_pylist()
            values[1] = modify(values[1])
            field = table.schema.field(column)
            changed = table.set_column(table.schema.get_field_index(column), field, pa.array(values, type=field.type))
            pq.write_table(changed, copy / parquet.relative_to(export))
            try:
                verify(source, copy, True)
            except ValueError as error:
                expected = "relative time" if column == "timestamp" else column
                if expected not in str(error):
                    raise
                rejection[column] = str(error)
            else:
                raise ValueError(f"The consumer accepted a changed interior {column} field")
    return rejection


def audit(export, source):
    meta = export / "provenance/export.json"
    before = sha(meta)
    verification = verify(source, export, True)
    file = load(meta)
    prefixes = [technical_prefix(export, source, item) for item in verification["episodes"]]
    record = corpus_record(prefixes, technical=True)
    if len(prefixes) != 2 or not verification["technical_only"] or record["source_group_count"] != 1:
        raise ValueError("Expected two original technical recordings in one grouping-control block")
    reader = PhysicalMotorDataset(record, "development", 3, technical=True)
    data = LeRobotDataset(file["repo_id"], root=export, return_uint8=True)
    checked = check_batch(reader, data, prefixes)
    check_storage_rejection(export, prefixes[0])
    changed_rows = check_interior_row_rejections(export, source)
    if sha(meta) != before:
        raise ValueError("Rehearsal changed the existing official export")
    return {"scope": "Native technical archive with existing LeRobot storage", "source_export": str(export),
            "export_metadata_sha256": before, "official_verified_frames": verification["frames"],
            "export_storage_sha256": prefixes[0]["native_export"]["export_storage_sha256"],
            "native_prefixes": len(prefixes), "rows_in_batch": len(reader), "group_count": record["source_group_count"],
            **checked, "rejected_inputs": [*check_rejections(record), "changed_export_storage"], "physical_prefixes_admitted": 0,
            "model_optimizer_updates": 0, "numeric_rows_verified": verification["numeric_rows_verified"],
            "measured_recording_timing": verification["episodes"], "changed_interior_rows_refused": changed_rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-prefix-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Choose a new rehearsal report path")
    export = Path(load(args.native_prefix_audit)["native_export"]["path"])
    source = Path(load(export / "provenance/export.json")["source_path"])
    atomic(args.output, audit(export, source))
    print(args.output)


if __name__ == "__main__":
    main()
