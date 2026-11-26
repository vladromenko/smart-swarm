import pytest

from smart_swarm.episode import atomic
from smart_swarm.physical_story_condition import OBSERVED_CHANGE, audit_condition, phase_digest
from smart_swarm.physical_story_operator import observe_motion
from smart_swarm.physical_story_plan import prepare
from smart_swarm.physical_story_source import mark, story_workspace


@pytest.mark.parametrize(("family", "initial", "terminal", "relocation", "removal", "status"), [
    ("F1", "placement_succeeded", "right", "confirmed", None, "observed"),
    ("F1", "placement_failed", "surface", "confirmed", None, "deviation"),
    ("F2", "placement_failed", "surface", "confirmed", None, "observed"),
    ("F2", "placement_succeeded", "left", "confirmed", None, "deviation"),
    ("F3", "placement_succeeded", "surface", "confirmed", "confirmed", "observed"),
    ("F3", "placement_succeeded", "surface", "confirmed", "unknown", "unverified"),
    ("F4", "placement_succeeded", "left", "confirmed", None, "observed"),
    ("F4", "placement_succeeded", "left", "different", None, "deviation"),
    ("F5", "placement_succeeded", "right", "confirmed", None, "observed"),
    ("F5", "placement_succeeded", None, "confirmed", None, "unverified"),
    ("F6", "placement_succeeded", "left", "confirmed", None, "withholding_only"),
    ("F6", "placement_failed", "surface", "confirmed", None, "withholding_only"),
])
def test_observed_conditions_are_distinct_from_the_assigned_card(family, initial, terminal, relocation, removal, status):
    observed_at = "insertion_closed" if family == "F3" else "placement_closed"
    record = {"family": family, "labels": {"terminal": terminal}, "attended_placement_outcomes": {observed_at: initial}}
    markers = {"relocation_closed": {OBSERVED_CHANGE: relocation}, "placement_closed": {OBSERVED_CHANGE: removal}}
    output = audit_condition(record, markers, phase_digest(markers))
    assert output["status"] == status and output["assigned_family"] == family
    assert record["labels"]["terminal"] == terminal
    if family == "F6":
        assert set(output["checks"]) == {"carrier_transition"}
    else:
        assert audit_condition(record, markers, None)["status"] == "unverified"


def test_phase_marker_preserves_deviation_and_binds_checks(tmp_path, monkeypatch):
    config = {"data_root": tmp_path}
    card = next(row for row in prepare(config)["cards"] if row["family"] == "F5")
    trial = story_workspace(config) / card["story_id"]
    trial.mkdir()
    initial = {"monotonic_ns": 10, "observed_placement_outcome": "placement_succeeded",
               "assistance": "attended_native_motion"}
    atomic(trial / "trial.json", {"status": "recording", "card": card, "markers": {"placement_closed": initial}})
    monkeypatch.setattr("builtins.input", lambda *_: "hd")
    actor, check = observe_motion("relocation", "human_relocation", "the inner carrier nest inside the tray")
    result = mark(config, card["story_id"], "relocation_closed", assistance=actor, observed_change=check)
    markers = result["markers"]
    record = {"family": "F5", "labels": {"terminal": "right"},
              "attended_placement_outcomes": {"placement_closed": "placement_succeeded"}}
    digest = phase_digest(markers)
    assert markers["relocation_closed"]["assistance"] == "human_relocation"
    assert markers["relocation_closed"][OBSERVED_CHANGE] == "different"
    assert audit_condition(record, markers, digest)["status"] == "deviation"
    markers["relocation_closed"][OBSERVED_CHANGE] = "confirmed"
    with pytest.raises(ValueError, match="phase observations changed"):
        audit_condition(record, markers, digest)
