"""Versioned F1–F6 capture and split accounting for the learning pilot."""

from collections import Counter
import shutil
import time

from .contract import ProtocolError
from .episode import LABELS, atomic, load
from .evidence_data import row_for
from .evidence_inputs import runtime_signature
from .evidence_rollouts import EvidenceStory, prepare_world, reveal

FAMILIES = tuple(f"F{i}" for i in range(1, 7))
SPLITS = {"train": (50, 100_000), "validation": (10, 200_000), "test": (10, 300_000)}
LIMITS = {"wall_s": 9000, "rollout_s": 100, "storage_mib": 1024, "attempts_per_group": 5}


def labels_for(story, terminal, slot):
    family = story.family
    carrier = "violet" if story.recorder.meta["simulation_recipe_seed"] % 2 else "cyan"
    other = "cyan" if carrier == "violet" else "violet"
    gate = story.recorder.meta["physical_story_gate"]
    placement = carrier if gate["contents_after_witness"] == "carrier" else "surface"
    mapping = f"{carrier if slot == 'left' else other}_left"
    blocked = family in ("F4", "F6")
    return {"settled_placement": None if blocked else placement,
            "relocation_assignment": None if family == "F4" else mapping,
            "intermediate_events": None, "terminal": terminal, "resolvable": not blocked,
            "scope": "training_or_evaluation_only",
            "placement_masked_reason": {"F4": "unmarked identities", "F6": "camera absent"}.get(family),
            "relocation_masked_reason": "unmarked identities" if family == "F4" else None}


def capture_one(assets, path, seed, split, family, variant, time_limit):
    if (path / "manifest.json").is_file():
        return row_for(path)
    world, variation, pickup = prepare_world(assets, seed, family, variant=variant, seconds=time_limit)
    try:
        story = EvidenceStory(path, family, world, seed, variant == "hidden-surface" or family == "F2")
        story.recorder.quota_bytes = 768 * 1024
        story.recorder.meta["lineage"]["split"] = split
        story.recorder.meta.update(evidence_variant=variant, physical_variation=variation)
        story.scene(pickup)
        query, decision, _ = story.ask()
        finish, terminal, slot = reveal(story, decision)
        if not terminal["relations_preserved_until_reveal"]:
            raise ValueError("Post-decision reveal no longer represents query state")
        story.recorder.append(LABELS, "settled-event", labels_for(story, terminal, slot))
        story.recorder.meta["status"] = "valid"
        story.recorder.close(query, decision, finish)
        return {**row_for(path), "family": family, "resolvable": family not in ("F4", "F6")}
    except (Exception, KeyboardInterrupt) as error:
        if "story" in locals():
            story.recorder.quarantine(str(error) or type(error).__name__)
        raise
    finally:
        world.close()


def group_plan():
    for index in range(50):
        for split, (count, base) in SPLITS.items():
            if index < count:
                for family_index, family in enumerate(FAMILIES):
                    for repeat in range(LIMITS["attempts_per_group"]):
                        # Keep marker balance by index even after a failed acquisition.
                        yield split, family, index, base + 1000 * family_index + 2 * index + 10_000 * repeat + index % 2


def capture_group(assets, root, group, seed, deadline):
    split, family, index = group
    variants = ("hidden-contained", "hidden-surface") if family == "F6" else ("witnessed",)
    paths = [root / f"{split}-{family}-{index:02}-{seed}-{variant}" for variant in variants]
    records = []
    for path, variant in zip(paths, variants, strict=True):
        time_limit = min(LIMITS["rollout_s"], deadline - time.monotonic())
        if time_limit < 1:
            raise TimeoutError("Story generation wall cap reached")
        record = capture_one(assets, path, seed, split, family, variant, time_limit)
        records.append({**record, "family": family, "resolvable": family not in ("F4", "F6")})
    if family == "F6" and (runtime_signature(paths[0]) != runtime_signature(paths[1]) or
                           records[0]["terminal"] == records[1]["terminal"]):
        raise ValueError("F6 permitted inputs differ or held-back outcomes fail to differ")
    return records


def summary(attempts):
    accepted = [item for item in attempts if "records" in item]
    records = [row for item in accepted for row in item["records"]]
    splits = Counter(item["split"] for item in accepted)
    masks = {split: {name: dict(Counter(str(row[name]) for row in records if row["split"] == split))
                     for name in ("settled_placement", "relocation_assignment", "terminal")}
             for split in SPLITS}
    return {"limits": LIMITS, "attempted": len(attempts), "accepted_independent": len(accepted),
            "rejected": len(attempts) - len(accepted), "recorded_variants": len(records),
            "groups_by_split": dict(splits), "label_masks_by_split": masks, "attempts": attempts,
            "records": records, "complete": splits == {"train": 300, "validation": 60, "test": 60}}


def index_attempt(output, attempts, status, elapsed):
    attempts.append(status)
    result = {**summary(attempts), "generation_elapsed_s": elapsed()}
    atomic(output, result)
    return result


def generate(assets, root, heartbeat, wall_budget=None):
    root.mkdir(exist_ok=True, parents=True)
    output = root / "portfolio.json"
    prior = load(output) if output.is_file() else {}
    attempts = prior.get("attempts", [])
    spent = prior.get("generation_elapsed_s", 0.)
    started = time.monotonic()
    deadline = started + min(wall_budget or LIMITS["wall_s"], LIMITS["wall_s"] - spent)

    def elapsed():
        return spent + time.monotonic() - started

    accepted = {(row["split"], row["family"], row["index"]) for row in attempts if "records" in row}
    tried = {row["seed"] for row in attempts}
    usage = sum(item.stat().st_size for item in root.rglob("*") if item.is_file())
    result = {**summary(attempts), "generation_elapsed_s": spent}
    for split, family, index, seed in group_plan():
        group = (split, family, index)
        if seed not in tried and group not in accepted and time.monotonic() < deadline:
            heartbeat()
            if usage + 2 * 1024**2 >= LIMITS["storage_mib"] * 1024**2 or shutil.disk_usage(root).free < 2 * 1024**2:
                return result
            status = {"split": split, "family": family, "index": index, "seed": seed}
            try:
                status["records"] = capture_group(assets, root, group, seed, deadline)
                accepted.add(group)
            except KeyboardInterrupt:
                if "records" not in status:
                    status["rejected"] = "Interrupted before source group completed"
                index_attempt(output, attempts, status, elapsed)
                raise
            except (OSError, RuntimeError, ValueError, TimeoutError, ProtocolError) as error:
                status["rejected"] = str(error)
            result = index_attempt(output, attempts, status, elapsed)
            staged = root.glob(f"{split}-{family}-{index:02}-{seed}-*")
            usage += sum(file.stat().st_size for path in staged for file in path.rglob("*") if file.is_file())
    return result
