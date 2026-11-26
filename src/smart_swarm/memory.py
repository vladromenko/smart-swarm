"""Joint filter for a single cube, two identities and two candidate slots."""

from dataclasses import dataclass, asdict
from itertools import product
import math

from .contract import json_bytes

TRACKS = ("cyan", "violet")
SLOTS = ("left", "right")
CUBE_SUPPORT = (*TRACKS, "surface", "held", "outside")
PRIOR = dict(zip(CUBE_SUPPORT, (.42, .42, .08, .04, .04), strict=True))
PERMUTATIONS = ("cyan_left", "violet_left")
STATES = tuple(product(CUBE_SUPPORT, PERMUTATIONS))
QUERY = "Which container holds the orange cube? You may report a visible surface or request inspection."
WITNESS_PROCESSING_NS = 10_000_000
WITNESS_TRANSPORT_NS = {"so101-1": 30_000_000, "tank-1": 60_000_000}


@dataclass(frozen=True)
class Evidence:
    evidence_id: str
    origin_agent: str
    frame_ids: tuple
    capture_start_ns: str
    capture_end_ns: str
    received_ns: str
    factor: str
    hypotheses: dict
    calibration: str = "fixed uncalibrated visual observation model"

    def as_record(self):
        return asdict(self)


def likelihood(evidence, state):
    cube, assignment = state
    return evidence["hypotheses"][cube if evidence["factor"] == "containment" else assignment]


def admissible(deliveries, boundary, recipient):
    sources = set()
    observations = []
    ordered = sorted(deliveries, key=lambda msg: int(msg["delivered_ns"]))
    for msg in ordered:
        evidence = msg["evidence"]
        frames = set(evidence["frame_ids"])
        capture, received, sent, delivered = map(int, (
            evidence["capture_end_ns"], evidence["received_ns"], msg["sent_ns"], msg["delivered_ns"]))
        if (msg["recipient"] == recipient and frames and not frames & sources
                and int(evidence["capture_start_ns"]) <= capture <= received <= sent <= delivered <= boundary):
            observations.append(evidence)
            sources.update(frames)
    return observations


def predict(deliveries, boundary, recipient="rosmaster-1"):
    evidence = admissible(deliveries, boundary, recipient)
    logs = {state: math.log(PRIOR[state[0]] / len(PERMUTATIONS))
            + sum(math.log(likelihood(item, state)) for item in evidence) for state in STATES}
    center = max(logs.values())
    weights = {state: math.exp(value - center) for state, value in logs.items()}
    norm = sum(weights.values())
    belief = dict.fromkeys((*SLOTS, "surface", "held", "outside"), 0.)
    for (cube, assignment), weight in weights.items():
        slot = "left" if assignment.startswith(cube) else "right"
        belief[slot if cube in TRACKS else cube] += weight / norm
    best = max(belief, key=belief.get)
    return {"belief": belief, "decision": best if belief[best] >= .75 else "request_inspection",
            "posterior": [{"location": cube, "assignment": assignment, "p": value / norm}
                          for (cube, assignment), value in weights.items()],
            "accepted_evidence_ids": [item["evidence_id"] for item in evidence]}


def deliver(evidence, sent_ns, delay_ns):
    return {"sender": evidence.origin_agent, "recipient": "rosmaster-1",
            "sent_ns": str(sent_ns), "delivered_ns": str(sent_ns + delay_ns),
            "evidence": evidence.as_record()}


def witness_delay_ns(agent):
    return WITNESS_PROCESSING_NS + WITNESS_TRANSPORT_NS[agent]


def serialized_cost(request, messages):
    responses = sum(len(json_bytes(message)) for message in messages)
    query_bytes = len(json_bytes(request))
    issued_ns = int(request["issued_time"]["monotonic_ns"])
    latency_ms = [(int(message["delivered_ns"]) - issued_ns)/1e6 for message in messages]
    return {"query_messages": 1, "query_bytes": query_bytes, "response_messages": len(messages),
            "response_bytes": responses, "total_bytes": query_bytes + responses,
            "response_latency_ms": latency_ms, "retry_messages": 0}
