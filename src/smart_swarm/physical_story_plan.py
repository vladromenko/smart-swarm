"""Preassign a four-block physical story pilot and preserve group provenance."""

from pathlib import Path

from .episode import atomic, load

ROOT = Path(__file__).resolve().parents[2]
PLAN_VERSION = "physical-stories-v1"


def story_card(profile, round_number, family, source):
    content = profile["families"][family]
    return {"story_id": f"{source}-b{round_number:02}-{family.lower()}", "family": family, "block": round_number,
            "source_group_id": f"{source}-block-{round_number:02}", "split": "development",
            "requested_carrier_destination": "left" if round_number % 2 else "right",
            "placement_regions": profile["placement_regions"], "permitted_views": profile["agents"],
            **content, "opening_boundary": "Confirm neutral native controls and record the marked stationary scene.",
            "closing_boundary": "Stop relevant native control and finish native capture before asking ROSMASTER.",
            "query_boundary": "Save the timestamped pre-inspection choice or inspection request before showing the reveal.",
            "reveal_boundary": "Record actual containment or surface outcome in the evaluator scope only after the query result is stored.",
            "reset": "With motion stopped, restore taped positions, remove temporary shields and recheck all three assigned views.",
            "native_recording": {"SO101": "SO101 Studio two-camera recording when placement is attempted",
                                 "Tank_and_ROSMASTER": "Explorer native read-only observation capture"},
            "record_assistance": True, "physical_status": "unrecorded"}


def prepare(config):
    profile = load(ROOT / "configs/physical-stories.json")
    root = config["data_root"] / "collection-plans"
    root.mkdir(exist_ok=True, parents=True)
    target = root / f"{PLAN_VERSION}.json"
    if not target.is_file():
        rows = [story_card(profile, index, family, PLAN_VERSION)
                for index in range(1, 5) for family in profile["families"]]
        atomic(target, {"title": profile["title"], "plan_version": PLAN_VERSION, "cards": rows,
                        "blocks": 4, "cards_per_block": 6, "planned_stories": 24,
                        "source_group_count": 4, "setup": profile["operator_action"],
                        "link_recordings_by": "native source IDs and explicit timestamped boundaries",
                        "physical_story_count": 0})
    return {"plan_path": str(target), **load(target)}
