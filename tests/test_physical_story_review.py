from pathlib import Path
from zipfile import ZipFile

from smart_swarm.episode import atomic, load, sha
from smart_swarm.physical_evidence_score import RULE
from smart_swarm.physical_paper import verify
from smart_swarm.physical_story_plan import prepare
from smart_swarm.physical_story_review import index_records, publish_ready, summarize


def finish(root, card, status):
    trial = root / card["story_id"]
    trial.mkdir(exist_ok=True)
    atomic(trial / "trial.json", {"status": status, "reason": "Technical rehearsal with no native placement"})


def test_closed_block_and_cumulative_review_preserve_rejections(tmp_path):
    config = {"data_root": tmp_path}
    plan = prepare(config)
    cards = plan["cards"]
    root = tmp_path / "physical-stories"
    root.mkdir()
    for card in cards[:5]:
        finish(root, card, "incomplete")
    assert publish_ready(config, refresh=True) == []
    initial, = (root / "paper-results").glob("*.zip")
    initial_result = verify(initial)
    assert initial_result["attempted_histories"] == 5 and not initial_result["figure_reproduced"]
    finish(root, cards[5], "pending_reveal")
    assert publish_ready(config) == []
    finish(root, cards[5], "complete")
    report, = publish_ready(config)
    assert report["planned_attempts"] == 6 and report["capture_counts"] == {"incomplete": 5, "rejected": 1}
    assert report["unknown_terminal_outcomes"] == 0 and report["admitted_source_groups"] == 0
    assert report["next_prepared_cards"][0]["story_id"] == cards[6]["story_id"]
    assert report["motor_prefix_counts"] == {"not_reviewed": 6}
    assert "no native placement" in report["cases"][0]["reason"]
    package = Path(report["paper_package"])
    assert package != initial and verify(package)["scored_modes"] == []
    with ZipFile(package) as archive:
        assert b"Attempted 6 of 24 planned cards" in archive.read("table.md")
        assert "forecast-outcomes.pdf" not in archive.namelist()
    assert publish_ready(config) == []
    for card in cards[6:]:
        finish(root, card, "incomplete")
    assert {Path(row["path"]).stem for row in publish_ready(config, refresh=True)} == {
        "block-01", "block-02", "block-03", "block-04", "all-24"}
    cumulative = load(root / "block-reviews/all-24.json")
    assert cumulative["capture_counts"] == {"incomplete": 23, "rejected": 1}
    assert cumulative["next_prepared_cards"] == []
    assert publish_ready(config, refresh=True) == []


def admitted_record(root, card, terminal):
    root.mkdir()
    path = root / "decisions/prediction.jsonl"
    path.parent.mkdir()
    belief = {"left": .8, "right": .05, "surface": .05, "held": .05, "outside": .05}
    decision = {"at": "test decision time", "query_all": {"belief": belief, "decision": "left"}}
    atomic(path, decision)
    manifest = root / "manifest.json"
    atomic(manifest, {"decision_boundary": decision["at"], "sidecar_sha256": {"decisions/prediction.jsonl": sha(path)},
                      "policy_cameras": {"so101-1": [], "tank-1": ["tank-1/front"]}})
    return {"story_id": card["story_id"], "scene_group_id": card["source_group_id"], "family": card["family"],
            "split": card["split"], "camera_packets": {"so101-1": 1, "tank-1": 8},
            "condition_audit": {"status": "withholding_only"}, "attended_placement_outcomes": {},
            "assistance_by_phase": {}, "pre_inspection_decision": "left", "path": str(root), "manifest_sha256": sha(manifest),
            "labels": {"terminal": terminal, "settled_placement": None, "relocation_assignment": None,
                       "intermediate_events": None}}


def test_coverage_counts_views_and_verified_outcome_errors_only(tmp_path):
    card = prepare({"data_root": tmp_path})["cards"][5]
    record = admitted_record(tmp_path / "fixture", card, "surface")
    stories = {"records": [record], "exclusions": []}
    motors = {"prefixes": [], "corpus_rejections": [{"story": card["story_id"], "status": "excluded", "reason": "Withheld"}]}
    report = summarize([card], index_records(stories, motors), {}, [])
    assert report["usable_camera_histories"] == {"tank-1/front": 1}
    assert report["wrong_answers"] == [{"method": RULE, "mode": "online", "observed": "surface", "predicted": "left", "count": 1}]
    assert report["motor_prefix_counts"] == {"excluded": 1}
    record["labels"]["terminal"] = None
    report = summarize([card], index_records(stories, motors), {}, [])
    assert report["wrong_answers"] == [] and report["unknown_terminal_outcomes"] == 1
