import pytest

from smart_swarm.contract import json_bytes
from smart_swarm.evidence_budget_replay import exchange
from smart_swarm.evidence_queries import reply_wire
from smart_swarm.particle_memory import FACTORS


def window(horizon):
    ads = []
    for sender, factor, delay in (("so101-1", "containment", 40), ("tank-1", "association", 70)):
        ad = {"sender": sender, "factor": factor, "capture_end_ns": "0000000000000000000",
              "frame_ids": [sender + ":0001"], "delay_ns": delay * 1_000_000}
        dummy = dict.fromkeys(FACTORS[factor], "0.00000000")
        ads.append({**ad, "reply_bytes": len(json_bytes(reply_wire(ad, 0, dummy)))})
    return 0, horizon * 1_000_000, ads


@pytest.mark.parametrize(
    ("strategy", "horizon", "placement", "scheduled", "latency"),
    [("sequential", 90, "cyan", ["containment"], 40),
     ("sequential", 120, "cyan", ["containment", "association"], 110),
     ("sequential", 120, "surface", ["containment"], 40),
     ("parallel", 90, "cyan", ["containment", "association"], 70)],
)
def test_schedules_fetch_only_deliverable_posteriors(strategy, horizon, placement, scheduled, latency):
    seen = []
    references = {factor: dict.fromkeys(classes, 1 / len(classes)) for factor, classes in FACTORS.items()}
    gains = {"containment": 1.0, "association": 0.0}

    def posterior(factor):
        seen.append(factor)
        choice = placement if factor == "containment" else "cyan_left"
        return {name: float(name == choice) for name in FACTORS[factor]}

    report = exchange(window(horizon), 2048, strategy, gains, posterior, references, references["containment"], .65)
    assert seen == scheduled
    assert report["modeled_latency_ms"] == latency
    if len(seen) == 2:
        final = report["wire"]["replies"][-1]
        assert int(final["sent_ns"]) == (40_000_000 if strategy == "sequential" else 0)
