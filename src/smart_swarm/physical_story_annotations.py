"""Review observed event labels after sealing a physical story and its forecast."""

from hashlib import sha256
from pathlib import Path

import numpy as np

from .contract import interval, json_bytes, read_ns
from .episode import LABELS, TRUTH, Replay, atomic, load, rows, sha, validate_reveal
from .evidence_inputs import AGENTS
from .memory import CUBE_SUPPORT, PERMUTATIONS
from .physical_reveal import validate_sealed_clip
from .physical_slot_reference import adapt_rule, slot_reference
from .physical_story_replay import OUTCOMES
from .physical_story_source import card_for, story_workspace
from .vision import insertion, relocation

SCHEMA = "smart-swarm-physical-event-review-v1"
TERMINAL_SCHEMA = "smart-swarm-physical-event-review-v2"
TERMINAL = "terminal"
FIELDS = {"first_insertion": CUBE_SUPPORT, "settled_placement": CUBE_SUPPORT, "relocation_assignment": PERMUTATIONS}
ROLE_BY_EVENT = {"first_insertion": (AGENTS[0], AGENTS[0] + "/top"),
                 "settled_placement": (AGENTS[0], AGENTS[0] + "/top"),
                 "relocation_assignment": (AGENTS[1], AGENTS[1] + "/front")}
MAX_REVIEW_IMAGES = 3


def frame_windows(root, state):
    family = state["card"]["family"]
    start = state["start"]["monotonic_ns"]
    markers = {name: marker["monotonic_ns"] for name, marker in state["markers"].items()}
    placed = markers["placement_closed"]
    result = {"settled_placement": (start, placed), "relocation_assignment": (placed, markers["relocation_closed"])}
    if "insertion_closed" in markers and family == "F3":
        inserted = markers["insertion_closed"]
        result.update(first_insertion=(start, inserted), settled_placement=(inserted + 1, placed))
    return {name: scoped_frames(root, name, limits) for name, limits in result.items()
            if family != "F4" and (family != "F6" or name == "relocation_assignment")}


def scoped_frames(root, event, limits):
    agent, sensor = ROLE_BY_EVENT[event]
    client = Replay(root, agent)
    low, high = limits
    result = []
    for packet in client.packets():
        for camera in packet["cameras"]:
            boundary = camera["capture_interval"]
            if (camera["sensor_id"] == sensor and low <= int(boundary["start_ns"]) <= int(boundary["end_ns"])
                    <= int(packet["receive_time"]["monotonic_ns"]) <= high):
                result.append((packet, camera))
    return result


def visual_vote(root, event, frames):
    if len(frames) < 2:
        return None
    client = Replay(root, ROLE_BY_EVENT[event][0])
    recent = frames[-MAX_REVIEW_IMAGES:]
    decoded = [(packet, np.asarray(client.image(camera["frame_id"]))) for packet, camera in recent]
    if event == "relocation_assignment":
        first, camera = frames[0]
        initial = np.asarray(client.image(camera["frame_id"]))
        reference = slot_reference(root, load(root / "manifest.json"))
        estimates = [adapt_rule(relocation([first, packet], initial, image), reference)
                     for packet, image in decoded if first != packet]
    else:
        estimates = [insertion(packet, image) for packet, image in decoded]
    proposed = [max(value.hypotheses, key=value.hypotheses.get) for value in estimates if value is not None]
    if len(proposed) >= 2 and len(proposed) == len(estimates) and len(set(proposed)) == 1:
        return {"value": proposed[0], "agreeing_rgb_checks": len(proposed),
                "confidence": "uncalibrated visual agreement",
                "provenance": "existing scoped RGB rule; candidates require direct visual review"}
    return None


def catalog(root, state):
    available = frame_windows(root, state)
    result = {}
    for event, vocabulary in FIELDS.items():
        stream = available.get(event, [])
        frames = stream[-MAX_REVIEW_IMAGES:]
        images = [review_image(camera) for _, camera in frames]
        status = "needs_confirmation" if images else "unknown_no_scoped_frame"
        if event not in available:
            status = "not_applicable" if event == "first_insertion" and state["card"]["family"] != "F3" else "withheld_by_family"
        result[event] = {"allowed_labels": list(vocabulary), "candidate": visual_vote(root, event, stream),
                         "review_images": images, "status": status}
    if state.get("status") == "complete" and "observed_terminal_outcome" in state:
        result[TERMINAL] = terminal_catalog(root, state)
    return result


def review_image(camera):
    return {key: camera[key] for key in ("frame_id", "frame_ref", "sha256", "sensor_id", "capture_interval")}


def terminal_catalog(root, state):
    manifest = load(root / "manifest.json")
    reference = f"{TRUTH}/reveal.jsonl"
    if (manifest["run_mode"] != "hardware" or state["decision"]["at"] != manifest["decision_boundary"]
            or manifest["sidecar_sha256"].get(reference) != sha(root / reference)):
        raise ValueError("The evaluator review must use the sealed native recording after its saved decision")
    row, = rows(root / reference)
    outcome = state["observed_terminal_outcome"]
    reveal = manifest["reveal_boundary"]
    clock = reveal["clock_id"]
    if (row["ground_truth"] != outcome or outcome not in OUTCOMES or row["observed_at"] != reveal
            or row["relations_preserved_until_reveal"] is not True):
        raise ValueError("The evaluator review and attended outcome do not match")
    decision_ns = read_ns(state["decision"]["at"]["monotonic_ns"])
    validate_reveal(root, decision_ns, read_ns(reveal["monotonic_ns"]), clock)
    validate_sealed_clip(root, row, manifest)
    clip = row.get("reveal_clip")
    frames = clip["frames"] if clip else [{"camera": row["camera"]}]
    images = []
    for frame in frames:
        receipt = frame.get("native_preview_receipt")
        proof = interval(read_ns(receipt["earliest_possible_capture_ns"]), read_ns(receipt["received_ns"]), clock) if receipt else None
        images.append({**review_image(frame["camera"]), "access_role": "evaluator_only", "conservative_capture_interval": proof})
    return {"allowed_labels": list(OUTCOMES[:-1]), "candidate": None, "review_images": images,
            "status": "needs_confirmation" if outcome == "unknown" else "confirmed_at_capture",
            "attended_outcome": outcome, "target_state_time_or_interval": row["target_state_time_or_interval"]}


def validate_annotation(event, submitted, available):
    if event == TERMINAL and available["status"] != "needs_confirmation":
        raise ValueError("The original attended terminal outcome is already confirmed")
    required = {"value", "frame_ids", "directly_observed", "basis"}
    if event == "relocation_assignment":
        required.add("rosmaster_left_reference_verified")
    if not isinstance(submitted, dict) or set(submitted) != required:
        raise ValueError(f"{event} requires a label, allowed RGB evidence and a direct observation statement")
    basis = submitted["basis"]
    vocabulary = OUTCOMES[:-1] if event == TERMINAL else FIELDS[event]
    if (submitted["value"] not in vocabulary or submitted["directly_observed"] is not True
            or not isinstance(basis, str) or not basis.strip() or len(basis) > 400):
        raise ValueError(f"{event} must be directly observed with a brief explanation")
    chosen = submitted["frame_ids"]
    permitted = {image["frame_id"]: image for image in available["review_images"]}
    if (not isinstance(chosen, list) or not chosen or any(type(item) is not str for item in chosen)
            or len(set(chosen)) != len(chosen) or not set(chosen) <= set(permitted)):
        raise ValueError(f"{event} refers to an unavailable, duplicate or wrong-phase frame")
    record = {"value": submitted["value"], "basis": basis.strip(), "directly_observed": True,
              "frames": [permitted[identifier] for identifier in chosen]}
    if event == "relocation_assignment":
        if submitted["rosmaster_left_reference_verified"] is not True:
            raise ValueError("Confirm the observed ROSMASTER left reference before labeling Tank's assignment")
        record["rosmaster_left_reference_verified"] = True
    return record


def build_receipt(state, manifest_digest, fields, available):
    if state.get("status") != "complete":
        raise ValueError("Complete the saved decision and evaluator reveal before confirming any annotation")
    reviewable = (FIELDS.keys() | {TERMINAL}) & available.keys()
    if not isinstance(fields, dict) or not fields or not fields.keys() <= reviewable:
        raise ValueError("Confirm one or more listed event labels and omit unknown events")
    labels = {event: validate_annotation(event, row, available[event]) for event, row in fields.items()}
    schema = TERMINAL_SCHEMA if TERMINAL in labels else SCHEMA
    return {"schema": schema, "scope": "training_or_evaluation_only", "story_id": state["card"]["story_id"],
            "manifest_sha256": manifest_digest, "decision_boundary": state["decision"]["at"],
            "source": "explicit review of cited scoped RGB", "observations": labels}


def read_receipt(path, state, manifest_digest, available):
    if path.stem != sha(path):
        raise ValueError("The saved annotation content changed")
    actual = load(path)
    submitted = {event: {key: value for key, value in row.items() if key != "frames"}
                 | {"frame_ids": [frame["frame_id"] for frame in row["frames"]]} for event, row in actual["observations"].items()}
    expected = build_receipt(state, manifest_digest, submitted, available)
    if actual != expected:
        raise ValueError("The saved annotation changed its story, scope, decision or original evidence")
    return actual


def saved_reviews(trial, state, manifest_digest, available):
    directory = trial / LABELS / "reviewed-events"
    accepted, refused = {}, []
    for path in sorted(directory.glob("*.json")):
        try:
            accepted[path.stem] = read_receipt(path, state, manifest_digest, available)
        except (ValueError, KeyError, OSError, TypeError) as error:
            refused.append({"receipt": path.name, "reason": str(error)})
    result = {}
    for event in available:
        votes = {key: receipt["observations"][event]["value"] for key, receipt in accepted.items()
                 if event in receipt["observations"]}
        values = set(votes.values())
        status = "confirmed" if len(values) == 1 else "conflicting_confirmations" if votes else available[event]["status"]
        value = next(iter(values)) if len(values) == 1 else None
        if status == "confirmed_at_capture":
            value = available[event]["attended_outcome"]
        result[event] = {"value": value, "status": status, "receipts": votes}
    return {"events": result, "refused": refused}


def apply_reviews(trial, root, state, record):
    available = catalog(root, state)
    reviewed = saved_reviews(trial, state, record["manifest_sha256"], available)
    events = reviewed["events"]
    labels = dict(record["labels"])
    labels.update({name: events[name]["value"] for name in ("settled_placement", "relocation_assignment")})
    if TERMINAL in events and labels[TERMINAL] is None:
        labels[TERMINAL] = events[TERMINAL]["value"]
    first = events["first_insertion"]["value"]
    labels["intermediate_events"] = {"first_insertion": first} if first is not None else None
    proposals = {name: {"candidate": item["candidate"], "reviewable_frames": len(item["review_images"])}
                 for name, item in available.items()}
    return {**record, "labels": labels, "event_annotations": reviewed, "event_annotation_proposals": proposals}


def annotation_action(config, identifier, submitted_path=None, *, interactive=False):
    from .physical_evidence_dataset import load_admitted

    card = card_for(config, identifier)
    trial = story_workspace(config) / identifier
    if not (trial / "trial.json").is_file():
        raise ValueError("This prepared card has no saved physical story")
    state = load(trial / "trial.json")
    record = load_admitted(config, trial, card, state)
    root = Path(record["path"])
    available = catalog(root, state)
    result = {"scope": "post-decision supervision review", "story_id": identifier, "image_root": str(root),
              "events": available, "stored": record["event_annotations"]}
    fields = load(submitted_path) if submitted_path is not None else None
    if interactive:
        from .physical_annotation_review import review

        fields = review(root, available, result["stored"]["events"])
        if not fields:
            return result
    if fields is not None or submitted_path is not None:
        result.update(save_confirmation(config, trial, state, record["manifest_sha256"], fields, available))
    return result


def save_confirmation(config, trial, state, manifest_digest, fields, available):
    from .physical_evidence_dataset import publish

    receipt = build_receipt(state, manifest_digest, fields, available)
    content = sha256(json_bytes(receipt) + b"\n").hexdigest()
    directory = trial / LABELS / "reviewed-events"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{content}.json"
    if not path.exists():
        atomic(path, receipt)
    if sha(path) != content:
        raise ValueError("The immutable source annotation changed")
    return {"saved_receipt": str(path), "stored": saved_reviews(trial, state, manifest_digest, available),
            "evidence_corpus_manifest": publish(config)["manifest"]}
