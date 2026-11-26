from copy import deepcopy
from pathlib import Path

import pytest

from smart_swarm.contract import (VERSION, ProtocolError, check_contract, decode_envelope,
                                  envelope, read_ns, time_at)
from smart_swarm.episode import POLICY, Replay, json_bytes, load, rows, validate
from smart_swarm.mock import ACTION_SCHEMA, MockControl

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "contracts/fixture"
MS = 1_000_000


def test_frozen_contract_and_fixture():
    assert check_contract(ROOT / "contracts")["wire_examples"] == 13
    manifest = load(FIXTURE / "manifest.json")
    validate(FIXTURE, manifest)
    replay = Replay(FIXTURE, "so101-1")
    assert replay.acquire("replay")["state"] == "OBSERVE"
    packet = replay.packets()[0]
    assert len(packet["cameras"]) == 2 and packet["clock_uncertainty_ms"] == .35
    assert replay.image(packet["cameras"][1]["frame_id"]).size == (20, 16)
    records = rows(FIXTURE / "training_labels/commands.jsonl")
    request, reply = records[0]["request"], records[0]["response"]
    result = decode_envelope(reply, request["request_id"])
    assert result["requested_values"][-1] == 108 and result["adapter_values"][-1] == 100
    assert result["native_applied_values"] is None and packet["measured_state"][-1] == 49
    assert records[1]["response"]["error"]["code"] == "STALE_COMMAND"
    lifecycle = rows(FIXTURE / "communication/lifecycle.jsonl")
    assert [row["type"] for row in lifecycle] == ["reserved", "gap", "lease_expired", "recovered"]
    assert lifecycle[-1]["state"] == "DISARMED"
    with pytest.raises(ProtocolError, match="NOT_ARMED"):
        replay.acquire("hardware", "policy")


def make_command(session, identifier, sequence, issued, *, request_id="one"):
    return {"protocol_version": VERSION, "request_id": request_id, "command_id": identifier,
            "session_id": session.session_id, "lease_epoch": session.lease_epoch,
            "freshness_token": session.freshness_token, "agent_id": "so101-1",
            "action_schema_id": ACTION_SCHEMA, "issued_time": time_at(issued*MS),
            "source_sample_ids": [], "action_values": [0, 0, 0, 0, 0, 50],
            "sequence": sequence, "max_age_ms": 100, "receiver_ttl_ms": 20,
            "intent": "exercise_fixture_only"}


def test_no_motion_and_expired_owner_cannot_neutralize_other_owner():
    service = MockControl()
    first = service.reserve(0, ttl_ms=50)
    assert first.state == "DISARMED" and not service.dispatches
    with pytest.raises(ProtocolError, match="NOT_ARMED"):
        service.submit(make_command(first, "early", 0, 0), 0)
    with pytest.raises(ProtocolError, match="RESOURCE_BUSY"):
        service.reserve(1)
    service.arm(first.session_id, first.lease_epoch, 2)
    second = service.reserve(60*MS)
    service.arm(second.session_id, second.lease_epoch, 61*MS)
    service.release(first.session_id, first.lease_epoch, 62*MS)
    assert service.neutralized == []
    with pytest.raises(ProtocolError, match="LEASE_EXPIRED"):
        service.submit(make_command(first, "old", 2, 62), 62*MS)
    service.release(second.session_id, second.lease_epoch, 65*MS)
    assert service.neutralized == [second.lease_epoch]


def test_freshness_monotonic_sequence_retry_and_request_echo():
    service = MockControl()
    session = service.reserve(0)
    service.arm(session.session_id, session.lease_epoch, 0)
    first = make_command(session, "first", 3, 5)
    applied = service.submit(first, 9*MS)
    retry = {**first, "request_id": "retry"}
    repeated = service.submit(retry, 230*MS)
    assert decode_envelope(envelope("retry", data=repeated), "retry") == applied
    assert len(service.dispatches) == 1
    with pytest.raises(ProtocolError, match="COMMAND_ID_CONFLICT"):
        service.submit({**retry, "intent": "different"}, 230*MS)
    with pytest.raises(ProtocolError, match="STALE_COMMAND"):
        service.submit(make_command(session, "late", 4, 5), 230*MS)
    with pytest.raises(ProtocolError, match="STALE_COMMAND"):
        service.submit(make_command(session, "obsolete", 2, 230), 230*MS)
    with pytest.raises(ValueError, match="decimal strings"):
        read_ns(230*MS)


def test_backdated_privileged_and_future_frames_are_inaccessible(tmp_path):
    import shutil

    root = tmp_path / "fixture"
    shutil.copytree(FIXTURE, root)
    replay = Replay(root, "so101-1")
    before = replay.packets()
    private = deepcopy(before[0])
    private["hidden_answer"] = "injected test"
    private["validity"]["true_contents"] = "privileged"
    evaluator = rows(root / "evaluator_truth/reveal.jsonl")[0]["camera"]
    private["cameras"].append(evaluator)
    future = deepcopy(private)
    future["sequence"] = 2
    future["cameras"][0]["capture_interval"]["end_ns"] = "1100000000"
    (root / POLICY / "so101-1.jsonl").write_bytes(json_bytes(private)+b"\n"+json_bytes(future)+b"\n")
    assert replay.packets() == before
    assert replay.packets(970*MS) == []
    with pytest.raises(KeyError):
        replay.image(evaluator["frame_id"])
