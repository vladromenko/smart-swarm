import pytest

from smart_swarm.contract import json_bytes
from smart_swarm.evidence_queries import reply_wire
from smart_swarm.particle_memory import FACTORS
from smart_swarm.team_evidence import exchange, pooled_factors


def advert(sender, factor, delay):
    ad = {"sender": sender, "recipient": "rosmaster-1", "factor": factor, "capture_end_ns": "0000000000000000000",
          "frame_ids": [f"{sender}/front:0001"], "delay_ns": delay}
    dummy = dict.fromkeys(FACTORS[factor], "0.00000000")
    return {**ad, "reply_bytes": len(json_bytes(reply_wire(ad, 0, dummy)))}


def test_distinct_senders_decode_own_responses_and_cover_factors_first():
    ads = [advert("tank-2", "association", 70), advert("so101-1", "containment", 40),
           advert("tank-1", "association", 70)]
    references = {factor: dict.fromkeys(classes, 1 / len(classes)) for factor, classes in FACTORS.items()}
    gains = {"association": 0.1, "containment": 0.2}
    seen = []

    def posterior(sender, kind):
        seen.append((sender, kind))
        return references[kind]

    full = exchange("broadcast", (0, 90, ads), 3072, gains, posterior, references, references["containment"])
    assert seen == [("so101-1", "containment"), ("tank-1", "association"), ("tank-2", "association")]
    seen.clear()
    compact = exchange("factor_coverage", (0, 90, ads), 3072, gains, posterior, references, references["containment"])
    assert seen == [("so101-1", "containment"), ("tank-1", "association")]
    assert compact["serialized_bytes"] < full["serialized_bytes"]
    assert compact["wire"]["advertisements"] == full["wire"]["advertisements"]
    seen.clear()
    single = exchange("recent_per_factor", (0, 90, ads), 3072, gains, posterior, references, references["containment"],
                      max_replies=1)
    assert seen == [("so101-1", "containment")]
    assert single["wire"]["advertisements"] == full["wire"]["advertisements"]
    assert single["serialized_bytes"] < compact["serialized_bytes"]


def test_identical_reports_do_not_sharpen_one_event_and_retries_are_ignored():
    factors = {"association": {"cyan_left": .5, "violet_left": .5}}
    first = {"factor": "association", "probabilities": {"cyan_left": ".9", "violet_left": ".1"}, "frame_ids": ["one"]}
    second = {**first, "frame_ids": ["two"]}
    opposing = {**second, "probabilities": {"cyan_left": ".1", "violet_left": ".9"}}
    single = pooled_factors([first], factors)[0][1]
    same = pooled_factors([first, second], factors)[0][1]
    assert same == pytest.approx(single)
    assert pooled_factors([first, opposing], factors) == pooled_factors([first, opposing, first], factors)


def test_compact_recency_changes_sender_without_changing_transport_envelope():
    extra = {**advert("tank-2", "association", 70), "capture_end_ns": "0000000000000000001"}
    ads = [extra, advert("tank-1", "association", 70), advert("so101-1", "containment", 40)]
    reference = {name: dict.fromkeys(labels, 1 / len(labels)) for name, labels in FACTORS.items()}
    gain = {"containment": .2, "association": .1}
    posterior = lambda sender, factor: reference[factor]
    proposed = exchange("factor_coverage", (10, 100, ads), 3072, gain, posterior, reference, reference["containment"])
    recency = exchange("recent_per_factor", (10, 100, ads), 3072, gain, posterior, reference, reference["containment"])
    assert [row["sender"] for row in proposed["wire"]["replies"]] == ["so101-1", "tank-1"]
    assert [row["sender"] for row in recency["wire"]["replies"]] == ["so101-1", "tank-2"]
    assert proposed["wire"]["advertisements"] == recency["wire"]["advertisements"]
    assert proposed["serialized_bytes"] == recency["serialized_bytes"] <= 3072
