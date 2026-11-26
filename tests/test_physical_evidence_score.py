import pytest

from smart_swarm.episode import atomic, sha
from smart_swarm.physical_evidence_score import DECISION_FILE, HEADS, INSPECT, METHOD, PROFILE
from smart_swarm.physical_evidence_score import checked_forecast, matched_comparison, saved_decision, scoring_case


def forecast(winner, confidence):
    belief = dict.fromkeys(("left", "right", "surface", "held", "outside"), (1 - confidence) / 4)
    belief[winner] = confidence
    return {"belief": belief, "decision": winner if confidence >= .75 else INSPECT, "threshold": .75}


def case(identifier, group, family, terminal, predictions):
    record = {"story_id": identifier, "scene_group_id": group, "family": family, "split": "development",
              "labels": {"terminal": terminal}}
    learned = {"status": "unavailable", "reason": "reservation busy"}
    if predictions is not None:
        learned = {"status": "scored", "receipt": {"policy": METHOD, "profiles_sha256": sha(PROFILE),
                   "heads": dict(zip(HEADS, predictions, strict=True))}}
    decision = {"query_all": forecast("left", .8), "learned_evidence": learned}
    return scoring_case(record, decision)


def test_paired_score_weights_source_groups_and_excludes_unknowns_and_missing_heads():
    inspect = forecast("left", .4)
    cases = [case("f1", "block-1", "F1", "left", (forecast("left", .9), forecast("right", .9))),
             case("f6-left", "block-2", "F6", "left", (inspect, inspect)),
             case("f6-surface", "block-2", "F6", "surface", (inspect, inspect)),
             case("f6-unknown", "block-2", "F6", None, (inspect, inspect)),
             case("f2-unavailable", "block-3", "F2", "surface", None)]
    result = matched_comparison(cases)
    assert (result["labeled_matched_histories"], result["labeled_matched_scene_groups"]) == (3, 2)
    assert result["metrics_by_method"][HEADS[0]]["group_weighted_cost"] == .125
    assert result["metrics_by_method"][HEADS[1]]["group_weighted_cost"] == .625
    assert result["events_plus_reveal_minus_events_only"]["group_weighted_cost"] == .5
    assert result["f6_inspection_counts"][HEADS[0]] == 3


def test_score_refuses_invalid_distributions_and_changed_saved_forecast(tmp_path):
    predicted = forecast("left", .8)
    with pytest.raises(ValueError, match="invalid terminal distribution"):
        checked_forecast({**predicted, "belief": {**predicted["belief"], "right": .8}}, .75)
    with pytest.raises(ValueError, match="inspection threshold"):
        checked_forecast({**predicted, "decision": INSPECT}, .75)
    path = tmp_path / DECISION_FILE
    path.parent.mkdir()
    result = {"at": "saved decision", "query_all": predicted}
    atomic(path, result)
    manifest = tmp_path / "manifest.json"
    atomic(manifest, {"decision_boundary": result["at"], "sidecar_sha256": {DECISION_FILE: sha(path)}})
    record = {"path": str(tmp_path), "manifest_sha256": sha(manifest), "pre_inspection_decision": "left"}
    assert saved_decision(record) == result
    atomic(path, {**result, "query_all": forecast("right", .8)})
    with pytest.raises(ValueError, match="pre-reveal prediction changed"):
        saved_decision(record)
