import pytest

from smart_swarm.episode import atomic, sha
from smart_swarm.physical_evidence_score import DECISION_FILE, HEADS, INSPECT, METHOD, METHODS, PROFILE
from smart_swarm.physical_evidence_score import checked_forecast, matched_comparison, saved_decision, scoring_case
from smart_swarm.physical_inspection_controls import inspection_controls
from smart_swarm.physical_witness_score import USABLE, diagnose, top_class


def forecast(winner, confidence):
    belief = dict.fromkeys(("left", "right", "surface", "held", "outside"), (1 - confidence) / 4)
    belief[winner] = confidence
    return {"belief": belief, "decision": winner if confidence >= .75 else INSPECT, "threshold": .75}


def case(identifier, group, family, terminal, predictions, *, condition=None, mode="online", carrier="confirmed"):
    record = {"story_id": identifier, "scene_group_id": group, "family": family, "split": "development",
              "labels": {"terminal": terminal}}
    if condition is not None:
        record["condition_audit"] = {"status": condition, "attended_markers_sealed": True,
                                     "checks": {"carrier_transition": carrier}}
    learned = {"status": "unavailable", "reason": "reservation busy"}
    if predictions is not None:
        learned = {"status": "scored", "mode": mode, "receipt": {"policy": METHOD, "profiles_sha256": sha(PROFILE),
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


def test_inspection_controls_require_a_same_mode_block_and_confirmed_transitions():
    direct, inspect = forecast("left", .9), forecast("left", .4)
    first = case("b1-f1", "block-1", "F1", "left", (direct, inspect), condition="observed")
    hidden = case("b1-f6", "block-1", "F6", None, (inspect, inspect), condition="withholding_only")
    offline = case("b2-f6", "block-2", "F6", None, (inspect, inspect), condition="withholding_only", mode="blind_offline")
    unmatched = case("b2-f1", "block-2", "F1", "left", (direct, direct), condition="observed")
    deviating = case("b3-f6", "block-3", "F6", "surface", (inspect, inspect), condition="withholding_only", carrier="deviation")
    result = inspection_controls([first, hidden, unmatched, offline, deviating], METHODS)
    online = result["by_forecast_mode"]["online"]
    assert online["matched_source_blocks"] == 1 and online["eligible_f1_histories"] == 2
    assert online["eligible_f6_histories"] == 1 and not online["blocks"][0]["f6_terminal_available"]
    assert online["by_method"][HEADS[0]]["both_controls"] == 1
    assert online["by_method"][HEADS[1]]["both_controls"] == 0
    assert online["by_method"]["always_inspect"] == {"f1_correct_direct": 0, "f1_inspection": 1,
                                                     "f6_inspection": 1, "both_controls": 0}
    assert result["by_forecast_mode"]["blind_offline"]["matched_source_blocks"] == 0


def test_inspection_controls_refuse_unbound_choice_and_duplicate_family():
    direct = forecast("left", .9)
    first = case("b1-f1", "block-1", "F1", "left", (direct, direct), condition="observed")
    with pytest.raises(ValueError, match="multiple copies"):
        inspection_controls([first, first], METHODS)
    first["inspection_control_case"]["choices"][HEADS[0]] = INSPECT
    with pytest.raises(ValueError, match="differs from its saved story"):
        inspection_controls([first], METHODS)


def test_scored_online_fallback_counts_only_as_blind_offline():
    original = forecast("left", .8)
    replay = {"status": "scored", "mode": "blind_offline",
              "receipt": {"profiles_sha256": sha(PROFILE), "policy": METHOD, "heads": dict.fromkeys(HEADS, original)}}
    record = {"story_id": "fallback", "scene_group_id": "block-1", "family": "F1", "split": "development",
              "labels": {"terminal": "left"}, "blind_offline_forecast": replay}
    decision = {"query_all": original, "learned_evidence": {"status": "deferred_blind_offline",
                "online_attempt": {"status": "unavailable", "reason": "Recorder admission was busy", "stage": "compute_admission"}}}
    scored = scoring_case(record, decision)
    assert scored["learned_forecast_mode"] == "blind_offline"
    assert scored["online_admission_failure"] == "Recorder admission was busy"
    assert set(scored["forecasts"]) == set(METHODS)


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


def test_witness_score_requires_visible_support_verified_reference_and_on_time_reply():
    ad = {"factor": "association", "sender": "tank-1", "recipient": "rosmaster-1",
          "capture_end_ns": "90", "frame_count": 1, "delay_ns": 10}
    request = {"sender": "rosmaster-1", "recipients": ["tank-1"], "issued_ns": "100"}
    reply = {**ad, "frame_ids": ["tank-1/front:1"], "sent_ns": "100", "delivered_ns": "110",
             "probabilities": {"cyan_left": "0.9", "violet_left": "0.1"}}
    wire = {"advertisements": [ad], "requests": [request], "replies": [reply]}
    receipt = {"modeled_reply_window_ns": 10, "slot_mapping": "same",
               "marker_support_by_witness": {"tank-1": {"supported": True}},
               "heads": {head: {"wire": wire} for head in HEADS}}
    labels = {"relocation_assignment": "violet_left"}

    def association(source):
        return diagnose(source, labels, 100)["heads"][HEADS[0]]["association"]

    scored = association(receipt)
    assert scored["status"] == USABLE and top_class(scored) == "violet_left"
    assert association({**receipt, "modeled_reply_window_ns": 9})["status"] == "after_deadline"
    prior = {**receipt, "marker_support_by_witness": {"tank-1": {"supported": False}}}
    assert association(prior)["status"] == "prior_only"
    assert "probabilities" not in association(prior)
    with pytest.raises(ValueError, match="unverified side mapping"):
        association({**receipt, "slot_mapping": "unverified"})
