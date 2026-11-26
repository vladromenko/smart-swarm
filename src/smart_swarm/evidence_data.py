"""Generate bounded, grouped temporal-evidence histories using the B1 world."""

from pathlib import Path
import time
import uuid

import mujoco

from .episode import LABELS, TRUTH, atomic, load, rows
from .physics import World
from .story import Story, snapshot_check

GROUPS = {41_000: "train", 41_001: "train", 41_002: "train", 41_003: "validation", 41_004: "test"}
CASES = (
    ("F1", "F1", False, None),
    ("F2", "F2", True, None),
    ("F6-contained", "F6", False, "cube"),
    ("F6-surface", "F6", True, "surface_cube"),
    ("F6-resolvable", "F1", False, "cube"),
)
ORIGIN = uuid.UUID("098d3fef-c560-4168-aed6-7d50a6b764fa")


def variations(world, seed):
    shifts = world.rng.uniform(-.0014, .0014, size=2)
    q = int(world.model.joint("cube_pose").qposadr[0])
    world.data.qpos[q:q + 2] += shifts
    friction = float(world.rng.uniform(.97, 1.03))
    world.model.geom_friction[world.model.geom("cube_geom").id, 0] = friction
    fovy_offset = float(world.rng.uniform(-.12, .12))
    world.model.cam_fovy += fovy_offset
    mujoco.mj_forward(world.model, world.data)
    return {"seed": seed, "shift_xy_m": shifts.tolist(), "carried_cube_sliding_friction": friction,
            "camera_field_of_view_offset_deg": fovy_offset}


def settled_labels(story, case, swap, withheld):
    variant, family, _, _ = case
    marker = "violet" if swap else "cyan"
    contact = story.recorder.meta["physics_check"]
    contained = contact["contained_before_relocation"]
    admitted = None if family == "F6" else marker if contained else "surface"
    truth = {"target_slot": withheld["ground_truth"], "label_source": "post-decision evaluator reveal",
             "target_interval": withheld["target_state_time_or_interval"]}
    labels = {"settled_placement": admitted, "relocation_assignment": f"{marker}_left",
              "intermediate_events": None, "terminal": truth, "scope": "training_or_evaluation_only",
              "source": "simulator contact and identity after the trusted settled window",
              "placement_masked_reason": "no_witness_camera" if family == "F6" else None}
    story.recorder.append(LABELS, "settled-event", labels)
    return labels


def generate_one(assets, root, seed, split, case):
    variant, family, miss, target = case
    identifier = "history-" + uuid.uuid5(ORIGIN, f"{seed}:{variant}").hex
    path = root / identifier
    if (path / "manifest.json").is_file():
        return path
    swap = bool(seed % 2)
    world = World(assets, seed=seed, miss=family == "F2", swap_markers=swap, f6_target=target, timeout_s=150)
    try:
        physical = variations(world, seed)
        story = Story(path, family, world, seed, miss)
        story.recorder.meta["lineage"]["split"] = split
        story.recorder.meta.update(evidence_variant=variant, physical_variation=physical,
                                   snapshot_check=snapshot_check(world))
        story.scene()
        query, decision, report = story.ask()
        reveal, withheld = story.reveal(decision)
        settled_labels(story, case, swap, withheld)
        story.recorder.meta["status"] = "valid"
        story.recorder.close(query, decision, reveal)
    except (Exception, KeyboardInterrupt) as error:
        if path.is_dir() and "story" in locals():
            story.recorder.quarantine(str(error) or type(error).__name__)
        raise
    finally:
        world.close()
    return path


def row_for(path):
    manifest = load(path / "manifest.json")
    label = rows(path / LABELS / "settled-event.jsonl")[0]
    decision = rows(path / "decisions" / "prediction.jsonl")[0]["query_all"]["decision"]
    reveal = rows(path / TRUTH / "reveal.jsonl")[0]
    return {"path": str(path), "world_episode_id": manifest["world_episode_id"],
            "scene_group_id": manifest["scene_group_id"], "split": manifest["lineage"]["split"],
            "variant": manifest["evidence_variant"], "settled_placement": label["settled_placement"],
            "relocation_assignment": label["relocation_assignment"], "terminal": reveal["ground_truth"],
            "untrained_rule_decision": decision, "intermediate_event_labels": 0}


def generate(assets, root, wall_seconds=900):
    if not 30 <= wall_seconds <= 900:
        raise ValueError("Evidence capture wall time must be between 30 and 900 seconds")
    root.mkdir(parents=True, exist_ok=True)
    records, failures = [], []
    deadline = time.monotonic() + wall_seconds
    for seed, split in GROUPS.items():
        for case in CASES:
            if time.monotonic() < deadline:
                try:
                    path = generate_one(assets, root, seed, split, case)
                    records.append(row_for(path))
                except (OSError, ValueError, RuntimeError) as error:
                    failures.append({"seed": seed, "variant": case[0], "reason": str(error)})
                atomic(root / "portfolio.json", summary(records, failures))
    return summary(records, failures)


def summary(records, failures):
    splits = {split: len({row["scene_group_id"] for row in records if row["split"] == split})
              for split in ("train", "validation", "test")}
    return {"independent_scene_groups": len({row["scene_group_id"] for row in records}),
            "groups_by_split": splits, "recorded_variants": len(records), "planned_variants": len(GROUPS)*len(CASES),
            "placement_event_labels": sum(row["settled_placement"] is not None for row in records),
            "relocation_assignment_labels": len(records), "terminal_reveals": len(records),
            "intermediate_event_labels": 0, "records": records, "failed_attempts": failures}
