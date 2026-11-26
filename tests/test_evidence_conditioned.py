from smart_swarm.contract import json_bytes
from smart_swarm.evidence_conditioned import channel, fit_channels, risk_exchange
from smart_swarm.evidence_queries import reply_wire, time_field
from smart_swarm.learned_channel import fit_learned
from smart_swarm.particle_memory import FACTORS


def advert(sender, factor, delay_ms):
    row = {"sender": sender, "receiver": "rosmaster-1", "factor": factor, "capture_end_ns": time_field(0),
           "delay_ns": delay_ms * 1_000_000, "frame_ids": [sender + "/front:0001"], "visual_support": True}
    row["reply_bytes"] = len(json_bytes(reply_wire(row, 0, dict.fromkeys(FACTORS[factor], "0.00000000"))))
    return row


def clear_channels(advertisements):
    return {channel(ad): {"probabilities": {
        state: {outcome: .999 if outcome == state else .001 / (len(FACTORS[ad["factor"]]) - 1)
                for outcome in FACTORS[ad["factor"]]} for state in FACTORS[ad["factor"]]}}
            for ad in advertisements}


def test_duplicate_witness_views_have_one_independent_fit_weight():
    ad = advert("tank-7", "association", 70)
    row = {"advertisement": ad, "reported_class": "cyan_left", "event_label": "cyan_left", "source_group": "A"}
    assert fit_channels([row]) == fit_channels([row] * 5)
    other = {**row, "source_group": "B"}
    table = fit_channels([row, other])[channel(ad)]
    assert table["independent_group_support"]["cyan_left"] == 2
    assert table["probabilities"]["cyan_left"]["cyan_left"] > .8


def test_posterior_changes_followup_only_when_delay_permits_it():
    ads = [advert("arm", "containment", 40), advert("observer", "association", 70)]
    channels = clear_channels(ads)
    prior = {"cyan": .25, "violet": .25, "surface": .48, "held": .01, "outside": .01}

    def replay(window, content):
        def response(_sender, factor):
            choice = content if factor == "containment" else "cyan_left"
            return {key: float(key == choice) for key in FACTORS[factor]}

        return risk_exchange((0, window * 1_000_000, ads), 3072, response, channels, prior)

    parallel = replay(90, "surface")
    assert len(parallel["wire"]["requests"]) == 1 and len(parallel["wire"]["replies"]) == 2
    surface = replay(120, "surface")
    assert surface["decision"] == "surface" and [entry["sender"] for entry in surface["updates"]] == ["arm"]
    stored = replay(120, "cyan")
    assert stored["decision"] == "left" and [entry["sender"] for entry in stored["updates"]] == ["arm", "observer"]
    assert [entry["issued_ns"] for entry in stored["wire"]["requests"]] == [time_field(0), time_field(40_000_000)]


def test_learned_channel_ignores_sender_id_and_preserves_no_support():
    rows = []
    for factor, labels in FACTORS.items():
        camera = advert("usual-camera", factor, 40)
        for label in labels[:2]:
            rows.append({"advertisement": camera, "source_group": label, "event_label": label, "reported_class": label})
    empirical = fit_channels(rows)
    learned, report = fit_learned(rows, empirical)
    anonymous = [{**row, "advertisement": {**row["advertisement"], "sender": "other-camera"}} for row in rows]
    assert fit_learned(anonymous, fit_channels(anonymous)) == (learned, report)
    assert learned["containment/visible"]["probabilities"]["held"] == empirical["containment/visible"]["probabilities"]["held"]
    assert all(fit["final_fit_nll"] < fit["initial_fit_nll"] for fit in report["factors"].values())
