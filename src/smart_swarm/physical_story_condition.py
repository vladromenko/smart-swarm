"""Audit the observed condition separately from the assigned physical pilot card."""

from hashlib import sha256

from .contract import json_bytes

PHASE_DIGEST = "attended_phase_observations_sha256"
OBSERVED_CHANGE = "observed_change"
CHANGE_VALUES = ("confirmed", "different", "unknown")
REMOVAL = "the cube leave the carrier and rest on the marked surface"
CARRIER_MOVE = "the carrier move from the stationary start to a marked station"
RELOCATION_CHECKS = {
    "F2": "the empty carrier move to a marked station while the cube stayed on the marked surface",
    "F4": "the same-looking unmarked carriers cross or pass behind the shield",
    "F5": "the small carrier nest inside the outer tray and move with it",
}


def phase_digest(markers):
    return sha256(json_bytes(markers)).hexdigest()


def verify_phase_digest(markers, sealed_digest):
    if sealed_digest is not None and sealed_digest != phase_digest(markers):
        raise ValueError("The attended phase observations changed after the story was sealed")


def observation(value, expected):
    if value in (None, "unknown"):
        return "unknown"
    return "confirmed" if value in expected else "deviation"


def audit_condition(record, markers, sealed_digest):
    verify_phase_digest(markers, sealed_digest)
    family = record["family"]
    labels = record["labels"]
    checks = {"carrier_transition": observation(markers["relocation_closed"].get(OBSERVED_CHANGE), ("confirmed",))}
    if family != "F6":
        phase = "insertion_closed" if family == "F3" else "placement_closed"
        outcome = record["attended_placement_outcomes"].get(phase)
        expected = "placement_failed" if family == "F2" else "placement_succeeded"
        checks["initial_placement"] = observation(outcome, (expected,))
        terminal = ("surface",) if family in ("F2", "F3") else ("left", "right")
        checks["terminal_relation"] = observation(labels["terminal"], terminal)
        if labels.get("settled_placement") is not None:
            expected_settled = ("surface",) if family in ("F2", "F3") else ("cyan", "violet")
            checks["reviewed_settled_placement"] = observation(labels["settled_placement"], expected_settled)
    if family == "F3":
        checks["removal"] = observation(markers["placement_closed"].get(OBSERVED_CHANGE), ("confirmed",))
        inserted = (labels.get("intermediate_events") or {}).get("first_insertion")
        if inserted is not None:
            checks["reviewed_insertion"] = observation(inserted, ("cyan", "violet"))
    if family == "F6":
        status = "withholding_only"
    elif sealed_digest is None:
        status = "unverified"
    elif "deviation" in checks.values():
        status = "deviation"
    elif "unknown" in checks.values():
        status = "unverified"
    else:
        status = "observed"
    return {"assigned_family": family, "status": status, "checks": checks, "attended_markers_sealed": sealed_digest is not None}
