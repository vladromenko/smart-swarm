"""Capture a fixed split for changing relocation witnesses and camera placements."""

import argparse
from collections import Counter
from contextlib import closing
from hashlib import sha256
from pathlib import Path
import shutil
import time

from rehearse_witness_mounts import CASE_LIMIT_S, CASE_QUOTA_BYTES, OWNERS, aim_mount, witness_views
from smart_swarm.cli import ROOT, settings
from smart_swarm.episode import LABELS, TRUTH, atomic, load, rows, sha
from smart_swarm.rendering import configure_renderer
from smart_swarm.research_jobs import admitted

LIMIT_S = 1200
DISK_CAP_BYTES = 96 * 1024**2
FREE_RESERVE_BYTES = 1024**3
SCHEMA = "smart-swarm-witness-transfer-v1"
ORIGINAL_PROFILE = ROOT / "evidence/learning-pilot-v3/primary-profiles-frozen.json"
RELEASE_QUALIFICATION = ROOT / "evidence/learning-pilot-v3/witness-transfer-release-qualification.json"


def recipes():
    rows = []
    combinations = (("placed", "left", "cyan_left", 0), ("placed", "left", "violet_left", 1),
                    ("placed", "right", "cyan_left", 5), ("placed", "right", "violet_left", 4),
                    ("missed_grasp", "surface", "cyan_left", 10), ("early_release", "surface", "violet_left", 12))
    for split, start, view_shift, spacing in (("validation", 731000, 0., .018), ("test", 731080, .045, .032)):
        for task, terminal, assignment, offset in combinations:
            seed = start + offset
            right_layout = bool(seed // 4 % 2)
            heights = {"tank-1": (.36 if right_layout else .06) + view_shift, "tank-2": .12 + view_shift}
            extra = (None,) if offset == 5 else ()
            rows.append({"seed": seed, "source_group": f"witness-transfer-{seed}", "split": split,
                         "task": task, "expected_terminal": terminal, "expected_assignment": assignment,
                         "mount_heights_m": heights, "station_aim_m": [.49 + view_shift, .10, .26],
                         "decoy_y_offset_m": spacing, "owners": [*OWNERS, *extra]})
    return rows


def new_design():
    qualification = load(RELEASE_QUALIFICATION)
    if qualification["status"] != "verified" or qualification["task"] != "early_release":
        raise ValueError("An independently checked premature release is required before allocating these groups")
    groups = recipes()
    check_allocation(groups)
    dependencies = ("scripts/capture_witness_transfer.py", "scripts/rehearse_witness_mounts.py",
                    "src/smart_swarm/evidence_rollouts.py", "src/smart_swarm/scaling.py",
                    "src/smart_swarm/physics.py", "src/smart_swarm/scene.py", "src/smart_swarm/teacher.py")
    return {"schema": SCHEMA, "scope": "new MuJoCo camera transfer comparison",
            "qualification_sha256": sha(RELEASE_QUALIFICATION), "original_profiles_sha256": sha(ORIGINAL_PROFILE),
            "dependency_sha256": {name: sha(ROOT / name) for name in dependencies}, "sources": groups,
            "case_wall_seconds": CASE_LIMIT_S, "total_capture_wall_seconds": LIMIT_S, "disk_limit_bytes": DISK_CAP_BYTES,
            "team_size": 4, "fit_split": "validation", "independent_evaluation_split": "test"}


def check_allocation(groups):
    if len({row["seed"] for row in groups}) != len(groups):
        raise ValueError("An independent source seed crosses the declared split")
    poses = {}
    for split in ("validation", "test"):
        selected = [row for row in groups if row["split"] == split]
        if (Counter(row["expected_terminal"] for row in selected) != {"left": 2, "right": 2, "surface": 2}
                or Counter(row["expected_assignment"] for row in selected) != {"cyan_left": 3, "violet_left": 3}):
            raise ValueError("Each split requires independently balanced terminals and assignments")
        poses[split] = {(actor, row["mount_heights_m"][actor], tuple(row["station_aim_m"]))
                        for row in selected for actor in OWNERS}
    if poses["test"] & poses["validation"]:
        raise ValueError("A held-out witness and camera-pose combination also occurs in fitting")


class ReleaseBeforeCarrier:
    """Use the stock teacher with an early release over the station surface."""

    def __init__(self, teacher):
        self.teacher = teacher

    def advance(self, phase, point, grip, speed, seconds, sample):
        height = {"transfer": .145, "lower": .05, "release": .05, "withdraw": .168}.get(phase)
        target = (.24, -.076, height) if height is not None else point
        self.teacher.advance(phase, target, grip, speed, seconds, sample)


def early_release_scene(story, pickup):
    from smart_swarm.contract import time_at
    from smart_swarm.evidence_rollouts import settled_truth
    from smart_swarm.story import AGENTS
    from smart_swarm.teacher import run_insertion
    from smart_swarm.vision import insertion

    story.approach()
    story.capture(AGENTS[0])
    run_insertion(ReleaseBeforeCarrier(story.teacher), story.arm_sample, pickup=pickup)
    story.world.step(200)
    before = settled_truth(story.world)
    packet, pictures = story.capture(AGENTS[0])
    story.add_evidence(insertion(packet, pictures["so101-1/top"]))
    story.relocate()
    story.capture(AGENTS[2])
    final = settled_truth(story.world)
    story.recorder.append(TRUTH, "physics", {"at": time_at(story.world.ns), "teacher_phase": "post_relocation", **final})
    contact = [row for phase in ("lift", "transfer", "lower") for row in story.stage_data[phase]
               if len(row["gripper_contact_bodies"]) >= 2 and row["cube_height_above_table_m"] >= .05]
    shift = final["carrier_xyz_m"][0] - before["carrier_xyz_m"][0]
    if (len(contact) < 5 or shift < .12 or any(row["cube_relation"] != "surface" or not row["released"]
            or not row["stationary"] for row in (before, final))):
        raise ValueError(f"Early release failed contact or settling checks with {len(contact)} lifted samples and {final['cube_relation']}")
    story.recorder.meta["physical_story_gate"] = {"first_insertion": False, "carry_contact_samples": len(contact),
        "contents_after_witness": "surface", "final_relation": "surface", "displacement_m": shift,
        "controlled_failure_mechanism": "early_release_after_supported_lift", "nested": False, "removed": False}


def story_capture(world, path, recipe, variation, pickup):
    from smart_swarm.evidence_portfolio import labels_for
    from smart_swarm.evidence_rollouts import reveal
    from smart_swarm.scaling import PassiveReceiverStory, describe_capture

    task = recipe["task"]
    family = "F1" if task == "placed" else "F2"
    story = PassiveReceiverStory(path, family, world, recipe["seed"], task != "placed")
    story.recorder.quota_bytes = CASE_QUOTA_BYTES
    started = time.monotonic()
    try:
        lineage = {"source_group": recipe["source_group"], "split": recipe["split"]}
        describe_capture(story, variation, "witness-transfer", 4, True, lineage)
        if task == "early_release":
            early_release_scene(story, pickup)
        else:
            story.scene(pickup)
        query, decision, report = story.ask()
        finish, terminal, slot = reveal(story, decision)
        labels = labels_for(story, terminal, slot)
        story.recorder.append(LABELS, "settled-event", labels)
        story.recorder.meta.update(status="valid", measured_capture_wall_s=time.monotonic() - started,
                                   experiment_role="witness-transfer-" + recipe["split"])
        story.recorder.close(query, decision, finish)
    except (Exception, KeyboardInterrupt) as error:
        story.recorder.quarantine(str(error) or type(error).__name__)
        raise
    return terminal["ground_truth"], labels["relocation_assignment"], story.recorder.meta["physical_story_gate"], report


def capture(config, path, recipe, owner, allowance):
    import mujoco
    from smart_swarm.evidence_rollouts import prepare_world
    from smart_swarm.scaling import audit_record

    family = "F2" if recipe["task"] == "missed_grasp" else "F1"
    world, variation, pickup = prepare_world(config["asset_root"], recipe["seed"], family,
                                              variant="witnessed", seconds=allowance, team_size=4)
    with closing(world):
        decoy = world.model.body("decoy")
        decoy.pos[1] += recipe["decoy_y_offset_m"]
        mujoco.mj_forward(world.model, world.data)
        mounts = {actor: aim_mount(world, actor, recipe["station_aim_m"], recipe["mount_heights_m"][actor],
                                  toward=actor == owner) for actor in OWNERS}
        initial = sha256(world.snapshot().integration.tobytes()).hexdigest()
        terminal, assignment, gate, report = story_capture(world, path, recipe, variation, pickup)
        final = sha256(world.snapshot().integration.tobytes()).hexdigest()
    result = {"path": str(path), "owner": owner, "mounts": mounts, "initial_state_sha256": initial,
              "final_state_sha256": final, "terminal": terminal, "association": assignment, "physical_gate": gate,
              "engineering_choice": report["query_all"]["decision"], "manifest_sha256": sha(path / "manifest.json"),
              "contact_trace_sha256": sha(path / TRUTH / "physics.jsonl"), "capture": audit_record(path)}
    return {**result, "scoped_view_audit": witness_views(result)}


def validate_case(recipe, case):
    if case["terminal"] != recipe["expected_terminal"] or case["association"] != recipe["expected_assignment"]:
        raise ValueError(f"The physical outcome differs from the preallocated balance in {recipe['source_group']}")
    views = case["scoped_view_audit"]
    documented = {actor for actor in OWNERS if all(row["supported"] for row in views[actor].values())}
    expected = {case["owner"]} if case["owner"] else set()
    if documented != expected:
        raise ValueError(f"The recorded RGB support differs from the planned camera condition: {documented}")


def validate_group(recipe, cases):
    from smart_swarm.scaling import matching_budgets

    if [case["owner"] for case in cases] != recipe["owners"]:
        raise ValueError("The sealed group omitted or repeated a companion view")
    matching_budgets([case["capture"] for case in cases])
    keys = ("initial_state_sha256", "final_state_sha256", "contact_trace_sha256", "physical_gate")
    for case in cases:
        validate_case(recipe, case)
        if any(case[key] != cases[0][key] for key in keys):
            raise ValueError("The changed view modified the physical scene or contact trajectory")


def qualify(config, root):
    root.mkdir(exist_ok=False, parents=True)
    recipe = {**recipes()[5], "seed": 730012, "source_group": "witness-transfer-release-development", "split": "development"}
    report = {"status": "running", "scope": "separate simulated release development", "task": recipe["task"],
              "new_physical_attempts": 0, "seed": recipe["seed"]}
    try:
        result = capture(config, root / "release", recipe, "tank-2", CASE_LIMIT_S)
        report["result"] = result
        validate_case(recipe, result)
        report["status"] = "verified"
    except (OSError, ValueError, RuntimeError) as error:
        report.update(status="rejected", reason=str(error))
        raise
    finally:
        atomic(root / "qualification.json", report)
    atomic(RELEASE_QUALIFICATION, report)
    return RELEASE_QUALIFICATION


def sealed_design(root):
    path = root / "design.json"
    requested = new_design()
    if path.is_file() and load(path) != requested:
        raise ValueError("The frozen transfer allocation or implementation has changed")
    if not path.is_file():
        root.mkdir(exist_ok=True, parents=True)
        atomic(path, requested)
    return requested


def prior_attempts(root, design):
    attempts = []
    for source in design["sources"]:
        path = root / source["source_group"] / "receipt.json"
        if path.is_file():
            saved = load(path)
            if saved["recipe"] != source:
                raise ValueError("Original paired recipe changed during capture")
            if saved["status"] == "running":
                saved.update(status="interrupted", reason="An admitted capture exited before sealing",
                             wall_seconds=len(source["owners"]) * CASE_LIMIT_S)
                atomic(path, saved)
            attempts.append(saved)
    return attempts


def one_group(config, target, source, allowance, heartbeat):
    target.mkdir(exist_ok=False)
    path = target / "receipt.json"
    result = {"recipe": source, "status": "running", "cases": []}
    started = time.monotonic()
    atomic(path, result)
    try:
        for owner in source["owners"]:
            heartbeat()
            limit = min(CASE_LIMIT_S, allowance - (time.monotonic() - started))
            if limit < 1:
                raise TimeoutError("The paired case reached the remaining capture allowance")
            case = capture(config, target / (owner or "both-away"), source, owner, limit)
            result["cases"].append(case)
            atomic(path, result)
        validate_group(source, result["cases"])
        result["status"] = "verified"
    except (OSError, ValueError, RuntimeError, TimeoutError) as error:
        result.update(status="rejected", reason=f"{type(error).__name__}: {error}")
    finally:
        result["wall_seconds"] = time.monotonic() - started
        atomic(path, result)
    return result


def run(config, root, count, heartbeat):
    design = sealed_design(root)
    attempts = prior_attempts(root, design)
    completed = {row["recipe"]["source_group"] for row in attempts}
    pending = [row for row in design["sources"] if row["source_group"] not in completed]
    for source in pending[:count]:
        remaining = LIMIT_S - sum(row["wall_seconds"] for row in attempts)
        disk = sum(file.stat().st_size for file in root.rglob("*") if file.is_file())
        size = len(source["owners"]) * CASE_QUOTA_BYTES
        if remaining > 1 and disk + size <= DISK_CAP_BYTES and shutil.disk_usage(root).free >= FREE_RESERVE_BYTES + size:
            attempts.append(one_group(config, root / source["source_group"], source, remaining, heartbeat))
    report = {"design_sha256": sha(root / "design.json"), "allocated_sources": len(design["sources"]),
              "attempted_sources": len(attempts), "verified_sources": sum(row["status"] == "verified" for row in attempts),
              "available_capture_seconds": max(0, LIMIT_S - sum(row["wall_seconds"] for row in attempts)),
              "attempts": [{key: value for key, value in row.items() if key != "cases"} for row in attempts]}
    atomic(root / "progress.json", report)
    return root / "progress.json"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-groups", type=int, choices=range(1, 13), default=1)
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--qualify-release", action="store_true")
    action.add_argument("--seal-allocation", action="store_true")
    args = parser.parse_args()
    if args.seal_allocation:
        sealed_design(args.output)
        print(args.output / "design.json")
        return
    config = settings()
    configure_renderer(config)
    with admitted(config, "smart-swarm-witness-transfer-capture") as (_, heartbeat):
        result = qualify(config, args.output) if args.qualify_release else run(config, args.output, args.max_groups, heartbeat)
    print(result)


if __name__ == "__main__":
    main()
