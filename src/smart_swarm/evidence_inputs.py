"""Causal replay projection for evidence and complete-input counterfactual checks."""

from copy import deepcopy
import hashlib
from pathlib import Path
import shutil
import tempfile

import numpy as np

from .contract import json_bytes
from .episode import Replay, atomic, load, rows
from .memory import predict

AGENTS = ("so101-1", "tank-1", "rosmaster-1")


def packets_and_transcript(root):
    agents = load(root / "manifest.json")["policy_cameras"]
    clients = {agent: Replay(root, agent) for agent in agents}
    reference = clients[AGENTS[-1]]
    packets = {agent: client.packets() for agent, client in clients.items()}
    messages = reference.deliveries()
    query = rows(root / "communication/queries.jsonl")[0]
    return packets, messages, query, reference


def opaque_map(packets, world_id):
    substitutions = {world_id: "world"}
    for agent, records in packets.items():
        for packet in records:
            substitutions[packet["episode_id"]] = f"{agent}:episode"
            substitutions[packet["sample_id"]] = f"{agent}:sample:{packet['sequence']}"
            for camera in packet["cameras"]:
                frame_id = camera["frame_id"]
                replacement = f"{camera['sensor_id']}:frame:{packet['sequence']}"
                substitutions[frame_id] = replacement
                substitutions[camera["frame_ref"]] = replacement
    return substitutions


def canonical_packet(packet, mapping):
    packet = deepcopy(packet)
    for name in ("world_episode_id", "episode_id", "sample_id"):
        packet[name] = mapping[packet[name]]
    for camera in packet["cameras"]:
        for name in ("frame_id", "frame_ref"):
            camera[name] = mapping[camera[name]]
    return packet


def canonical_delivery(message, mapping):
    row = deepcopy(message)
    evidence = row["evidence"]
    normalized_frames = [mapping[frame] for frame in evidence["frame_ids"]]
    evidence["frame_ids"] = normalized_frames
    # Evidence IDs are derived from the origin, factor and final contributing frame.
    prefix = f"{evidence['origin_agent']}:{evidence['factor']}:"
    final_original = message["evidence"]["frame_ids"][-1]
    if evidence["evidence_id"] != prefix + final_original:
        raise ValueError("Unexpected replay evidence ID format")
    evidence["evidence_id"] = prefix + normalized_frames[-1]
    return row


def runtime_signature(root):
    packets, messages, query, reference = packets_and_transcript(root)
    mapping = opaque_map(packets, reference.world_id)
    private = {agent: [canonical_packet(packet, mapping) for packet in items]
               for agent, items in packets.items()}
    messages = [canonical_delivery(message, mapping) for message in messages]
    return {"private_history": private, "delivered_messages": messages, "query": query,
            "query_ns": reference.query, "decision_ns": reference.decision,
            "camera_grants": {agent: Replay(root, agent).allowed for agent in packets},
            "runtime_cache": "stateless_rule_and_cold_pilot_clip", "seed": 3,
            "unseen_subscriptions": []}


def verified_prediction(root):
    signature = runtime_signature(root)
    result = predict(signature["delivered_messages"], signature["decision_ns"])
    recorded = Replay(root, AGENTS[-1]).predict()
    if any(result[key] != recorded[key] for key in ("belief", "posterior", "decision")):
        raise ValueError("Opaque receipt identifiers unexpectedly changed inference")
    return signature, result


def perturb_truth(source, scratch):
    with tempfile.TemporaryDirectory(dir=scratch, prefix="truth-counterfactual-") as directory:
        path = Path(directory) / "episode"
        shutil.copytree(source, path)
        reveal = path / "evaluator_truth/reveal.jsonl"
        row = rows(reveal)[0] | {"ground_truth": "injected-evaluator-only", "answer_overlay": "unseen"}
        reveal.write_bytes(json_bytes(row) + b"\n")
        truth = path / "evaluator_truth/physics.jsonl"
        truth.write_bytes(json_bytes({"private_sentinel": "unseen"}) + b"\n")
        commands = path / "training_labels/arm-control.jsonl"
        commands.write_bytes(json_bytes({"private_training_sentinel": "unseen"}) + b"\n")
        return verified_prediction(path)


def audit_b1(root, scratch):
    entries = [(p.parent, load(p)) for p in (root / "worlds").glob("*/manifest.json")]
    pairs = [path for path, meta in entries if meta["family"] == "F6"]
    first, second = [verified_prediction(path) for path in pairs]
    perturbed = perturb_truth(pairs[0], scratch)
    reveals = [rows(path / "evaluator_truth/reveal.jsonl")[0]["ground_truth"] for path in pairs]
    controls = {meta["family"]: verified_prediction(path)[1]["decision"]
                for path, meta in entries if meta["family"] in ("F1", "F2")}
    if first != second or first != perturbed or len(set(reveals)) != 2:
        raise ValueError("Complete pre-inspection input/coupling invariance failed")
    if controls != {"F1": "left", "F2": "surface"} or first[1]["decision"] != "request_inspection":
        raise ValueError("Resolving controls do not distinguish useful uncertainty from always inspect")
    digest = hashlib.sha256(json_bytes(first[0])).hexdigest()
    return {"rule_prediction_invariant": True, "actual_truth_permutation": reveals,
            "privileged_evaluator_and_training_sidecars_perturbed": True, "permitted_input_sha256": digest,
            "included": list(first[0]), "decision": first[1]["decision"], "resolved_controls": controls,
            "actuator_record_scope": "per-agent applied-command refs retained; privileged command ledger excluded",
            "identity_fields": "world, episode, sample, frame, path and evidence IDs normalized bijectively",
            "prediction_engine": "B1 deterministic finite joint filter and hand-coded image rules",
            "coupled_randomness": "B1 filter deterministic; no hidden state cache"}


def clip_packets(client, length=8):
    packets = [packet for packet in client.packets() if packet["cameras"]]
    if not packets:
        return []
    if len(packets) < length:
        raise ValueError("An admitted temporal input has fewer than eight packets")
    indices = np.linspace(0, len(packets) - 1, num=length, dtype=int)
    return [packets[int(index)] for index in indices]


def clip(root, agent, length=8):
    client = Replay(root, agent)
    packets = clip_packets(client, length)
    if not packets:
        return None
    return np.stack([np.stack([np.asarray(client.image(camera["frame_id"]).convert("RGB"))
                              for camera in packet["cameras"]]) for packet in packets])
