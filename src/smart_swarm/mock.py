"""Contract rehearsal for one simulated writer with caller-supplied clock ticks."""

from copy import deepcopy
from dataclasses import dataclass, field
import math
import uuid

from .contract import CLOCK, VERSION, ProtocolError, interval, json_bytes, read_ns

ACTION_SCHEMA = "sim-so101-absolute-v1"
LIMITS = ((-110, 110), (-100, 100), (-97, 97), (-95, 95), (-157, 162), (0, 100))


@dataclass
class Session:
    session_id: str
    run_mode: str
    lease_epoch: int
    lease_deadline_ns: int
    freshness_token: str
    state: str = "DISARMED"
    last_sequence: int = -1
    cache: dict = field(default_factory=dict)


class MockControl:
    def __init__(self):
        self.current = None
        self.epoch = 0
        self.dispatches = []
        self.neutralized = []

    def reserve(self, now, *, mode="fixture", ttl_ms=1000):
        if mode not in ("fixture", "mujoco"):
            raise ProtocolError("NOT_ARMED", "Mock capabilities grant simulated actuation only")
        if self.current and now < self.current.lease_deadline_ns:
            raise ProtocolError("RESOURCE_BUSY", "A live simulated writer holds the resource", True)
        if not 0 < ttl_ms <= 10_000:
            raise ValueError("Invalid fixture lease duration")
        self.epoch += 1
        self.current = Session(uuid.uuid4().hex, mode, self.epoch, now + ttl_ms * 1_000_000,
                               uuid.uuid4().hex)
        return deepcopy(self.current)

    def verify_lease(self, session_id, epoch, now):
        session = self.current
        if (session is None or session.session_id != session_id or session.lease_epoch != epoch
                or now >= session.lease_deadline_ns):
            raise ProtocolError("LEASE_EXPIRED", "No live simulated lease for this epoch")
        return session

    def arm(self, session_id, epoch, now):
        session = self.verify_lease(session_id, epoch, now)
        session.state = "ARMED"
        return {"session_id": session_id, "lease_epoch": epoch, "state": "ARMED",
                "run_mode": session.run_mode}

    def release(self, session_id, epoch, now):
        session = self.current
        if (session and session.session_id == session_id and session.lease_epoch == epoch
                and now < session.lease_deadline_ns):
            if session.state == "ARMED":
                self.neutralized.append(epoch)
            self.current = None

    def submit(self, command, now):
        session = self.verify_lease(command["session_id"], command["lease_epoch"], now)
        key = command["command_id"]
        semantic = json_bytes({k: v for k, v in command.items() if k != "request_id"})
        cached = session.cache.get(key)
        if cached is not None:
            payload, result = cached
            if payload != semantic:
                raise ProtocolError("COMMAND_ID_CONFLICT", "Command ID already has a different payload")
            if isinstance(result, ProtocolError):
                raise deepcopy(result)
            return deepcopy(result)
        try:
            adapter = self.validate_command(command, session, now)
        except ProtocolError as error:
            session.cache[key] = (semantic, deepcopy(error))
            raise
        session.last_sequence = command["sequence"]
        result = acknowledge(command, adapter, now)
        session.cache[key] = (semantic, deepcopy(result))
        self.dispatches.append(deepcopy(result))
        return result

    def validate_command(self, command, session, now):
        if session.state != "ARMED":
            raise ProtocolError("NOT_ARMED", "Simulation must be activated")
        if (command["protocol_version"] != VERSION or command["action_schema_id"] != ACTION_SCHEMA
                or command["agent_id"] != "so101-1"):
            raise ProtocolError("INVALID_ACTION_SCHEMA", "Unexpected simulated agent or action schema")
        stamp = command["issued_time"]
        age = now - read_ns(stamp["monotonic_ns"])
        budgets = [command["max_age_ms"], command["receiver_ttl_ms"]]
        valid_budget = all(type(x) in (float, int) and math.isfinite(x) and x > 0 for x in budgets)
        if (not valid_budget or stamp["clock_id"] != CLOCK or age < 0
                or age > budgets[0] * 1_000_000 or command["freshness_token"] != session.freshness_token
                or type(command["sequence"]) is not int or command["sequence"] <= session.last_sequence):
            raise ProtocolError("STALE_COMMAND", "Command age, clock, sequence or freshness is invalid")
        values = command["action_values"]
        if len(values) != len(LIMITS) or any(type(x) not in (int, float) or not math.isfinite(x) for x in values):
            raise ProtocolError("INVALID_ACTION_SCHEMA", "Expected six finite simulator joint values")
        return [min(max(x, lo), hi) for x, (lo, hi) in zip(values, LIMITS, strict=True)]


def acknowledge(command, adapter, now):
    return {"command_id": command["command_id"], "status": "dispatched",
            "requested_values": command["action_values"], "adapter_values": adapter,
            "native_applied_values": None, "native_applied_valid": False,
            "native_applied_reason": "mock_has_no_actuator_feedback",
            "dispatch_interval": interval(now, now), "hardware_acknowledgment": None,
            "measured_response_ref": None, "task_outcome": "unknown"}
