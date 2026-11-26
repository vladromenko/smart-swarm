from copy import deepcopy

import pytest

from smart_swarm.datasets import audit_lineage
from smart_swarm.memory import CUBE_SUPPORT, Evidence, PERMUTATIONS, deliver, predict


def evidence(name, agent, factor, choices, expected, frames=("frame1",)):
    return Evidence(name, agent, frames, "5", "10", "15", factor,
                    {item: (.98 if item == expected else .012) for item in choices})


def test_delayed_duplicate_overlap_and_factorization():
    cube = evidence("contains", "so101-1", "containment", CUBE_SUPPORT, "cyan")
    box = evidence("moved", "tank-1", "association", PERMUTATIONS, "cyan_left", ("frame2",))
    first = deliver(cube, 20, 20)
    second = deliver(box, 20, 40)
    assert predict([first, second], 25)["accepted_evidence_ids"] == []
    assert predict([first, second], 55)["decision"] == "request_inspection"
    both = predict([second, first], 60)
    assert both["decision"] == "left"
    assert sum(both["belief"].values()) == pytest.approx(1)
    assert sum(row["p"] for row in both["posterior"]) == pytest.approx(1)
    duplicate = deepcopy(first)
    duplicate["evidence"]["evidence_id"] = "repackaged"
    duplicate["evidence"]["frame_ids"] = ["frame1", "frame3"]
    duplicate["sender"] = "tank-1"
    duplicate["delivered_ns"] = "50"
    assert predict([second, duplicate, first], 60) == both


def lineage(world, group, split, sources=None):
    return {"world_episode_id": world, "scene_group_id": group,
            "lineage": {"source_world_ids": sources or [], "split": split}}


def test_paired_and_derived_ancestry_cannot_straddle_splits():
    root = lineage("world1", "scene", "train")
    sibling = lineage("world2", "scene", "train")
    derived = lineage("world3", "crop-scene", "train", ["world2"])
    assert audit_lineage([root, sibling, derived])["scene_groups"] == 2
    with pytest.raises(ValueError, match="Scene group"):
        audit_lineage([root, lineage("world2", "scene", "test")])
    with pytest.raises(ValueError, match="Source world"):
        audit_lineage([root, sibling, lineage("world3", "crop", "test", ["world2"])])
    with pytest.raises(ValueError, match="Cyclic or unresolved"):
        audit_lineage([lineage("world1", "scene", "train", ["world1"])])


def test_joint_particle_filter_resamples_and_preserves_coupled_probabilities():
    from smart_swarm.memory import PRIOR
    from smart_swarm.particle_memory import exact_filter, particle_filter, total_variation

    reports = [("containment", {name: 1. if name == "held" else .0001 for name in CUBE_SUPPORT}),
               ("association", {name: .75 if index == 0 else .25 for index, name in enumerate(PERMUTATIONS)})]
    exact = exact_filter(PRIOR, reports)
    first = particle_filter(PRIOR, reports, 2048)
    assert first["resamples"] == 1
    assert first == particle_filter(PRIOR, reports, 2048)
    assert total_variation(first["joint"], exact["joint"]) < .01
    assert sum(first["belief"].values()) == pytest.approx(1)


def test_query_policy_can_open_only_the_allocated_reply():
    from smart_swarm.contract import json_bytes
    from smart_swarm.evidence_comparison import learned_prior_map
    from smart_swarm.evidence_queries import WITNESSES, exchange, reply_wire, time_field
    from smart_swarm.memory import PRIOR
    from smart_swarm.particle_memory import FACTORS

    ads = []
    for index, (agent, (factor, delay)) in enumerate(WITNESSES.items()):
        candidate = {"sender": agent, "recipient": "rosmaster-1", "factor": factor, "frame_ids": [agent + ":0001"],
                     "capture_end_ns": time_field(10 + index), "delay_ns": delay}
        dummy = dict.fromkeys(FACTORS[factor], "0.00000000")
        candidate["reply_bytes"] = len(json_bytes(reply_wire(candidate, 100, dummy)))
        ads.append(candidate)
    asked = []
    def posterior(factor):
        asked.append(factor)
        return dict.fromkeys(FACTORS[factor], 1. / len(FACTORS[factor]))

    gains = {"containment": 1., "association": 0.}
    priors = learned_prior_map(([.2] * 5, [.5, .5]))
    result = exchange("expected_information_gain", (100, 100_000_100, ads), gains, posterior, priors, PRIOR)
    assert asked == ["containment"]
    assert result["wire"]["request"]["recipients"] == ["so101-1"]
    assert result["serialized_bytes"] == sum(result[name] for name in ("metadata_bytes", "request_bytes", "reply_bytes")) <= 2048
    asked.clear()
    result = exchange("latest_witness", (100, 50_000_100, ads), gains, posterior, priors, PRIOR)
    assert asked == ["containment"] and result["modeled_latency_ms"] == 40
