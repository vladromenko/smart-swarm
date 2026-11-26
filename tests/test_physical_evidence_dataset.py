from pathlib import Path

from smart_swarm import physical_evidence_dataset as corpus
from smart_swarm.episode import atomic, sha


def test_shared_native_capture_never_counts_twice(tmp_path, monkeypatch):
    config = {"data_root": tmp_path}
    plan = corpus.prepare(config)
    entries = {}
    for index, card in enumerate(plan["cards"][:3]):
        trial = tmp_path / "physical-stories" / card["story_id"]
        trial.mkdir(parents=True)
        atomic(trial / "trial.json", {"status": "complete"})
        entries[card["story_id"]] = {"story_id": card["story_id"], "family": card["family"],
                                     "scene_group_id": card["source_group_id"], "split": card["split"],
                                     "native_explorer_capture": "shared" if index < 2 else "unique",
                                     "native_studio_episode": str(index), "native_studio_frames_sha256": str(index),
                                     "labels": dict.fromkeys(("terminal", "settled_placement", "relocation_assignment",
                                                              "intermediate_events"))}
    monkeypatch.setattr(corpus, "load_admitted", lambda _config, _trial, card, _journal: entries[card["story_id"]])
    index = corpus.publish(config)
    assert index["attempted"] == 3 and index["admitted_histories"] == 1
    assert index["independent_source_groups"] == 1
    assert index["label_coverage"]["terminal"] == {"unknown": 1}
    assert {row["story_id"] for row in index["exclusions"]} == {card["story_id"] for card in plan["cards"][:2]}
    assert {row["reason"] for row in index["exclusions"]} == {"Repeated native source native_explorer_capture"}
    assert corpus.publish(config)["manifest"] == index["manifest"]
    path = Path(index["manifest"])
    assert sha(path) == path.stem
