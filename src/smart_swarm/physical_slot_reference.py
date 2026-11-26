"""Verify the physical Tank-to-ROSMASTER side convention from stationary RGB landmarks."""

from dataclasses import replace

import numpy as np
from PIL import Image

from .contract import json_bytes, project_packet
from .episode import POLICY, available, listed_file, rows, sha
from .evidence_inputs import AGENTS
from .memory import PERMUTATIONS

SCOPE_FLAG = "verify_physical_slot_reference"
REVERSED = "reversed"
SAME = "same"
UNKNOWN = "unverified"
MIN_HORIZONTAL_FRACTION = .125


def compact_center(mask):
    y, x = np.where(mask)
    if len(x) < max(12, mask.size // 1_000):
        return None
    width, height = int(x.max() - x.min() + 1), int(y.max() - y.min() + 1)
    if min(width, height) >= 4 and len(x) / (width * height) >= .55 and len(x) <= mask.size / 8:
        return float(np.median(x))
    return None


def landmark_masks(frame):
    r, g, b = frame.astype(np.int16).transpose(2, 0, 1)
    return ((g >= 130) & (r <= 65) & (b <= 65) & (g - r >= 65) & (g - b >= 65),
            (b >= 145) & (r <= 65) & (g <= 65) & (b - r >= 75) & (b - g >= 75))


def landmark_order(frame):
    green, blue = (compact_center(mask) for mask in landmark_masks(frame))
    enough = green is not None and blue is not None and abs(green - blue) >= frame.shape[1] * MIN_HORIZONTAL_FRACTION
    return {"green_x": green, "blue_x": blue, "green_on_image_left": green < blue if enough else None}


def association_image(frame):
    """Keep calibration-card colors out of the learned identity classifier."""
    if landmark_order(frame)["green_on_image_left"] is None:
        return frame
    trim = np.zeros(frame.shape[:2], dtype=bool)
    height, width = trim.shape
    for mask in landmark_masks(frame):
        y, x = np.where(mask)
        trim[max(0, y.min() - 1):min(height, y.max() + 2), max(0, x.min() - 1):min(width, x.max() + 2)] = True
    neutral = np.rint(np.median(frame[~trim], axis=0)).astype(np.uint8)
    return np.where(trim[..., None], neutral, frame)


def frame_evidence(root, manifest, agent):
    grants = manifest["policy_cameras"][agent]
    query = manifest["query_boundary"]
    stamp = int(query["monotonic_ns"])
    result = []
    for row in rows(root / POLICY / f"{agent}.jsonl"):
        packet = project_packet(row, agent, grants)
        if available(packet, stamp, query["clock_id"]):
            cameras = [camera for camera in packet["cameras"] if camera["sensor_id"] == agent + "/front"]
            for camera in cameras:
                path = listed_file(root, camera["frame_ref"], POLICY)
                if sha(path) != camera["sha256"]:
                    raise ValueError("A physical slot reference image changed")
                with Image.open(path) as image:
                    vote = landmark_order(np.asarray(image.convert("RGB")))
                result.append({"frame_id": camera["frame_id"], "sha256": camera["sha256"], **vote})
    return result


def slot_reference(root, manifest):
    if not manifest.get(SCOPE_FLAG):
        return None
    tank, receiver = AGENTS[1:]
    first, last = frame_evidence(root, manifest, tank), frame_evidence(root, manifest, receiver)
    order = [row["green_on_image_left"] for row in first]
    resolved = {vote for vote in order if vote is not None}
    stable = len(order) >= 2 and order[0] is not None and order[-1] is not None and len(resolved) == 1
    tank_order = order[0] if stable else None
    wire = {"sender": tank, "recipient": receiver, "green_on_image_left": tank_order}
    receipt = {"tank": first, "rosmaster": last, "wire": wire, "serialized_bytes": len(json_bytes(wire))}
    if len(last) != 1 or last[0]["green_on_image_left"] is not True:
        why = "ROSMASTER must see the green left landmark and blue right landmark"
    elif not stable:
        why = "Tank must see stable left and right landmarks at both ends of relocation"
    else:
        return {**receipt, "mapping": SAME if tank_order else REVERSED, "reason": None}
    return {**receipt, "mapping": UNKNOWN, "reason": why}


def mapped_assignment(values, mapping):
    if mapping == SAME:
        return dict(zip(PERMUTATIONS, (values[key] for key in reversed(PERMUTATIONS)), strict=True))
    return values


def adapt_rule(report, reference):
    if report is None or reference is None:
        return report
    if reference["mapping"] == UNKNOWN:
        return None
    return replace(report, hypotheses=mapped_assignment(report.hypotheses, reference["mapping"]))


def verify_decision_reference(root, manifest, decision):
    reference = slot_reference(root, manifest)
    if decision.get("slot_reference") != reference:
        raise ValueError("The saved physical camera reference differs from its original scoped RGB")
    learned = decision.get("learned_evidence", {})
    if reference is not None and learned.get("status") == "scored":
        receipt = learned["receipt"]
        if (receipt.get("slot_mapping") != reference["mapping"] or receipt.get("slot_reference_wire") != reference["wire"]
                or receipt.get("slot_reference_bytes") != reference["serialized_bytes"]):
            raise ValueError("The learned and rule forecasts used different physical slot references")
    return reference
