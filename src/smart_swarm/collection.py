"""Generate resumable, preassigned operator cards for later platform recording."""

from pathlib import Path
import uuid

from .contract import VERSION, ProtocolError
from .episode import atomic, load

ROOT = Path(__file__).resolve().parents[2]


def prepare(data_root, handover, name):
    profile = load(ROOT / "configs/collection.json")[name]
    root = data_root / "collection-plans"
    queue = root / f"{name}.json"
    if queue.is_file():
        return {**load(queue), "plan_path": str(queue)}
    root.mkdir(exist_ok=True, parents=True)
    plan_id = name + "-" + uuid.uuid4().hex[:10]
    record = {"protocol_version": VERSION, "plan": name, "plan_id": plan_id,
              "state": "blocked_on_platform_preflight", "run_mode_requested": "hardware",
              "platform_handover_found": handover.is_dir(), "accepted_count": 0,
              "recorded_count": 0, "required_roles": profile["required_roles"],
              "estimated_attempt_minutes": profile["estimated_attempt_minutes"],
              "next_card": 0, "cards": cards(plan_id, profile)}
    atomic(queue, record)
    return {**record, "plan_path": str(queue)}


def cards(plan_id, profile):
    choices = list(profile["families"])
    return [card(plan_id, idx, choices[idx % len(choices)], idx // len(choices), profile)
            for idx in range(profile["target_attempts"])]


def card(plan_id, index, family, repeat, profile):
    return {"card_id": f"{plan_id}-{index+1:02d}", "family": family,
            "task": profile["families"][family], "source_group": f"{plan_id}-setup-{repeat+1}",
            "split": "development", "starting_setup": profile["starting_setup"],
            "target_station_choice": ("left", "right")[repeat % 2],
            "checks": ["Operator owns controls", "Check view separation and acquisition",
                       "Confirm neutral controls and approved STOP", "Confirm available storage"],
            "start": "Record the reset and establish a common timestamped session boundary.",
            "end": "Record the decision or motor withdrawal before collecting outcome truth.",
            "reveal": "Expose the contents to the declared evaluator camera without changing their relation.",
            "reset": "Restore marked starting poses and confirm the camera views before retry.",
            "status": "queued"}


def collect(config, args):
    preparation = prepare(config["data_root"], config["platform_handover"], args.plan)
    if not args.plan_only:
        raise ProtocolError("OFFLINE", f"Physical collection awaits platform preflight. Prepared {preparation['plan_path']}")
    return preparation
