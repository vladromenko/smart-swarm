"""Generate the small B-owned wire-compatible fixture without MuJoCo."""

from pathlib import Path

import numpy as np

from smart_swarm.contract import CLOCK, VERSION, ProtocolError, envelope, interval, time_at
from smart_swarm.episode import LABELS, POLICY, TRUTH, Recorder, json_bytes
from smart_swarm.mock import ACTION_SCHEMA, MockControl

AGENT = "so101-1"
ROOT = Path(__file__).resolve().parents[1] / "contracts/fixture"
SENSORS = ["so101-1/wrist", "so101-1/top"]
WORLD = "world-so101-fixture-1"
MS = 1_000_000


def observations(recorder):
    frame = np.full((16, 20, 3), [28, 63, 79], dtype=np.uint8)
    packet = recorder.capture(AGENT, {SENSORS[0]: frame, SENSORS[1]: frame[::-1, ::-1]},
                              980*MS, [0, 1, 0, 0, 0, 49], ["deg"]*5 + ["calibrated_percent"])
    packet["clock_uncertainty_ms"] = .35
    packet["validity"]["state_unavailable_reason"] = "previous interval motor application not reported"
    path = recorder.root / POLICY / f"{AGENT}.jsonl"
    path.write_bytes(json_bytes(packet) + b"\n")
    image = recorder.frame(TRUTH, "evaluator/overhead", 1, frame, 1_100*MS)
    recorder.append(TRUTH, "reveal", {"camera": image, "ground_truth": "surface",
                                     "target_state_time_or_interval": interval(1_000*MS, 1_100*MS),
                                     "relations_preserved_until_reveal": True,
                                     "observed_at": time_at(1_100*MS)})
    (recorder.root / "communication/deliveries.jsonl").touch()


def command(stream, session, command_id, sequence, issued_ms, values):
    return {"protocol_version": VERSION, "request_id": stream, "session_id": session.session_id,
            "lease_epoch": session.lease_epoch, "command_id": command_id, "agent_id": AGENT,
            "action_schema_id": ACTION_SCHEMA, "sequence": sequence, "action_values": values,
            "source_sample_ids": [WORLD + "-0-0001"], "issued_time": time_at(issued_ms*MS),
            "max_age_ms": 100, "receiver_ttl_ms": 50, "freshness_token": session.freshness_token,
            "intent": "fixture_absolute_joint_target"}


def lifecycle(recorder):
    service = MockControl()
    first = service.reserve(0, ttl_ms=500)
    recorder.append("communication", "lifecycle", {"type": "reserved", "state": first.state,
                                                    "lease_epoch": first.lease_epoch})
    recorder.append("communication", "lifecycle", {"type": "gap", "sequence_from": 2,
                                                    "sequence_through": 3, "reason": "interrupted"})
    recorder.append("communication", "lifecycle", {"type": "lease_expired", "lease_epoch": first.lease_epoch})
    latest = service.reserve(900*MS, ttl_ms=500)
    recorder.append("communication", "lifecycle", {"type": "recovered", "lease_epoch": latest.lease_epoch,
                                                    "state": latest.state})
    service.arm(latest.session_id, latest.lease_epoch, 990*MS)
    okay = command("fixture-accepted", latest, "command-applied", 1, 990, [0, 1, 0, 0, 0, 108])
    old = command("fixture-stale", latest, "command-stale", 2, 800, [0, 1, 0, 0, 0, 50])
    recorder.append(LABELS, "commands", {"request": okay, "response": envelope(
        okay["request_id"], data=service.submit(okay, 1_000*MS))})
    try:
        service.submit(old, 1_000*MS)
    except ProtocolError as error:
        recorder.append(LABELS, "commands", {"request": old, "response": envelope(old["request_id"], error=error)})


def build():
    meta = {"run_mode": "fixture", "source_type": "software_fixture", "clock": CLOCK,
            "rehearsal": "scope, lease recovery, explicit unknowns and stale rejection"}
    recorder = Recorder(ROOT, WORLD, "fixture-source-group", {AGENT: SENSORS}, meta)
    observations(recorder)
    lifecycle(recorder)
    recorder.close(1_000*MS, 1_000*MS, 1_100*MS)


if __name__ == "__main__":
    build()
