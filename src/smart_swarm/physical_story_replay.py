"""Project native hardware recordings into separate timed witness scopes."""

from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image

from .contract import ProtocolError, time_at
from .episode import DELIVERIES, LABELS, POLICY, TRUTH, Recorder, atomic, load, rows, sha, validate, validate_agent
from .fleet_adapter import FleetArchive
from .memory import QUERY, deliver, predict
from .physical_fleet_timing import (HOME_MARGIN_NS, MAX_CLOCK_MS, FleetClock, PhysicalStoryRecorder,
                                    eligible_frames, wall_offset)
from .physical_learned import inference as learned_inference, offline_seal, verify_optional as verify_learned_prediction
from .physical_motor_prefix import CAMERA_ORDER as ARM_CAMERAS, episode_directory, placement_boundary, segment_choice
from .physical_reveal import FIELD as REVEAL_FIELD, PROFILE as REVEAL_PROFILE, capture_clip, store_clip, validate_sealed_clip
from .physical_slot_reference import SCOPE_FLAG, adapt_rule, slot_reference, verify_decision_reference
from .physical_story_condition import PHASE_DIGEST, phase_digest, verify_phase_digest
from .physical_story_source import (ACTIVE_STORY_STATES, FLEET_TERMINAL, StudioEpisodeUnlinked, anchor, card_for, corresponding_native_episode,
                                    fleet_manifest, homebrain_boot, mobile_phase_bounds, restore_start, same_recording_boot,
                                    save_incomplete, story_phases, story_update, story_workspace)
from .studio import native_controls_idle, native_status, validate_story_episode
from .vision import insertion, relocation

AGENTS = ("so101-1", "tank-1", "rosmaster-1")
OUTCOMES = ("left", "right", "surface", "held", "outside", "unknown")


def boot_clock():
    return {"host_id": "homebrain", "boot_id": homebrain_boot(), "kind": "monotonic"}


def selected(rows, count):
    if len(rows) < count:
        raise ValueError(f"Native camera window requires at least {count} distinct valid samples")
    return [rows[round(index * (len(rows) - 1) / (count - 1))] for index in range(count)] if count > 1 else rows[-1:]


def rgb(stream):
    with Image.open(stream) as image:
        return np.asarray(image.convert("RGB").resize((160, 120), Image.Resampling.BILINEAR))


def native_picture(source, camera):
    target = (source / camera["path"]).resolve()
    if not target.is_relative_to(source) or not target.is_file() or sha(target) != camera["sha256"]:
        raise ValueError("Studio native camera frame failed scope or byte provenance")
    return rgb(target)


def studio_clips(recorder, source, state):
    upper = state["markers"]["placement_closed"]["monotonic_ns"]
    lower = state["start"]["monotonic_ns"]
    manifest = source / "frames.jsonl"
    proof = state.get("native_story_validation")
    if proof is not None and sha(manifest) != proof["frames_sha256"]:
        raise ValueError("The native manifest changed after episode validation")
    source_rows = [item for item in rows(manifest)
                   if lower <= item["observation_timestamp_ns"] <= upper]
    if not source_rows:
        raise ValueError("No valid native Studio observations in the declared placement window")
    if state["card"]["family"] == "F6":
        return [(recorder.capture(AGENTS[0], {}, lower, [], []), {})]
    clips = []
    for item in placement_samples(source_rows, state):
        frames = {f"so101-1/{key}": item["cameras"][key] for key in ARM_CAMERAS}
        views = {name: native_picture(source, frame) for name, frame in frames.items()}
        times = {name: frame["timestamp_ns"] for name, frame in frames.items()}
        measured = item["observation.state"]
        units = ["degrees"] * 5 + ["percent_calibrated_range"]
        packet = recorder.capture(AGENTS[0], views, item["observation_timestamp_ns"], measured, units, exposures=times)
        clips.append((packet, views))
    return clips


def placement_samples(samples, state):
    if state["card"]["family"] == "F3":
        split = state["markers"]["insertion_closed"]["monotonic_ns"]
        first = [row for row in samples if row["observation_timestamp_ns"] <= split]
        second = [row for row in samples if row["observation_timestamp_ns"] > split]
        return [*selected(first, 4), *selected(second, 4)]
    return selected(samples, 8)


def fleet_clips(config, recorder, state, agent, through, fleet, audit, *, current_anchor=None):
    archive = FleetArchive(config, state["fleet_capture_id"], agent)
    start = state["start"]
    offset = wall_offset([start, *state["markers"].values(), current_anchor or anchor()])
    tank, receiver = mobile_phase_bounds(state, through)
    bounds = tank if agent == AGENTS[1] else receiver
    clock = FleetClock(archive.role, fleet["initial_hub_status"], archive.clock_history())
    eligible, audit[agent] = eligible_frames(archive.packets(through_s=60), clock, offset, bounds)
    required = 8 if agent == AGENTS[1] else 1
    audit[agent].update(required=required, permitted_interval_ns=list(bounds))
    if len(eligible) < required:
        raise ValueError(f"{agent} retained {len(eligible)} of {required} required RGB samples within its native phase")
    result = []
    for item in selected(eligible, required):
        picture = rgb(BytesIO(archive.image(item["frame_ref"], through_s=item["after_s"])))
        name = f"{agent}/front"
        packet = recorder.fleet_packet(agent, name, picture, item)
        result.append((packet, picture))
    audit[agent]["selected"] = len(result)
    return result


def persist_decision(config, recorder, arm, tank, query, *, evaluation_mode="online"):
    if evaluation_mode == "blind_offline":
        manifest = {**recorder.meta, "sample_counts": recorder.counts, "query_boundary": time_at(query, recorder.clock)}
        learned = offline_seal(recorder.root, manifest)
    else:
        learned = learned_inference(config, recorder, query)
    return write_decision(recorder, arm, tank, query, anchor()["monotonic_ns"], learned)


def write_decision(recorder, arm, tank, query, decision_ns, learned):
    placement = insertion(arm[-1][0], arm[-1][1]["so101-1/top"]) if arm[-1][1] else None
    manifest = {**recorder.meta, "query_boundary": time_at(query, recorder.clock)}
    reference = slot_reference(recorder.root, manifest)
    motion = adapt_rule(relocation([tank[0][0], tank[-1][0]], tank[0][1], tank[-1][1]), reference)
    request = {"origin_agent": AGENTS[2], "recipients": list(AGENTS[:2]), "text": QUERY,
               "issued_time": time_at(query, recorder.clock)}
    recorder.append("communication", "queries", request)
    messages = [deliver(item, decision_ns, 0) for item in (placement, motion) if item is not None]
    (recorder.root / "communication/deliveries.jsonl").touch()
    for message in messages:
        recorder.append("communication", "deliveries", message)
    result = predict(messages, decision_ns)
    record = {"at": time_at(decision_ns, recorder.clock), "query": QUERY, "agent_id": AGENTS[2], "query_all": result,
              "prediction_source": "deterministic local RGB rules and finite joint filter", "learned_evidence": learned}
    if reference is not None:
        record["slot_reference"] = reference
        verify_decision_reference(recorder.root, manifest, record)
    recorder.append("decisions", "prediction", record)
    return record


def admit_sources(config, trial, state, through, fleet):
    clock = boot_clock()
    cameras = {AGENTS[0]: [f"so101-1/{key}" for key in ARM_CAMERAS], **{agent: [agent + "/front"] for agent in AGENTS[1:]}}
    if state["card"]["family"] == "F6":
        cameras[AGENTS[0]] = []
    provenance = {"run_mode": "hardware", SCOPE_FLAG: True, REVEAL_FIELD: REVEAL_PROFILE,
                  "family": state["card"]["family"], "native_recording_clock": clock,
                  "clock_mapping_id": "native-studio-and-bracketed-explorer-hub-v1",
                  "clock_mapping_uncertainty_ms": MAX_CLOCK_MS + HOME_MARGIN_NS / 1e6,
                  "frame_calibration_revision": "native-studio-and-hub-provenance",
                  "operator_intervention": state["markers"]["relocation_closed"]["assistance"],
                  "operator_intervention_by_phase": {phase: mark["assistance"] for phase, mark in state["markers"].items()},
                  PHASE_DIGEST: phase_digest(state["markers"]),
                  "source_type": "native_hardware", "studio_episode": state["native_episode_id"],
                  "fleet_capture": state["fleet_capture_id"],
                  "native_validation": state["native_story_validation"]}
    recorder = PhysicalStoryRecorder(trial / "private-replay", state["card"]["story_id"], state["card"]["source_group_id"],
                                    cameras, provenance, clock=clock)
    audit = {}
    try:
        arm = studio_clips(recorder, episode_directory(config, state), state)
        tank = fleet_clips(config, recorder, state, AGENTS[1], through, fleet, audit)
        fleet_clips(config, recorder, state, AGENTS[2], through, fleet, audit)
        return persist_decision(config, recorder, arm, tank, through,
                                evaluation_mode=state.get("evaluation_mode", "online"))
    except (ValueError, OSError, RuntimeError, KeyError, ProtocolError) as error:
        recorder.quarantine(str(error))
        raise
    finally:
        if audit:
            atomic(trial / "fleet-timing-audit.json", audit)


def prepare_native_arm(config, trial, state, *, motor_only=False):
    boundary = placement_boundary(state["card"]["family"]) if motor_only else "placement_closed"
    row = corresponding_native_episode(config, state, boundary=boundary)
    location = episode_directory(config, {**state, "native_episode_id": row["id"]})
    proof = validate_story_episode(config, location)
    if (proof["episode_id"] != row["id"] or proof["dataset_id"] != state["studio_dataset_id"]
            or proof["frames_sha256"] != row["metadata"]["frames_sha256"]):
        raise ValueError("The Studio episode inventory and saved native bytes disagree")
    state.update(native_episode_id=row["id"], native_story_validation=proof)
    queue_motor_review(config, trial, state)


def queue_motor_review(config, trial, state):
    path = trial / "motor-intake.json"
    try:
        choice, reason = segment_choice(config, state)
        if choice and choice["native_frame_manifest_sha256"] != state["native_story_validation"]["frames_sha256"]:
            raise ValueError("The motor prefix differs from the validated native story manifest")
    except (OSError, ValueError, KeyError) as error:
        choice, reason = None, str(error)
    if choice is None:
        atomic(path, {"status": "excluded", "reason": reason})
    else:
        atomic(path, {"status": "pending_review", "reason": "Waiting for the block's official Studio export"})


def incomplete_story(trial, state, reason):
    return incomplete_result(save_incomplete(trial, state, reason))


def incomplete_result(state):
    return {"status": "incomplete", "reason": state["reason"], "motor_segment_admitted": state.get("motor_segment_admitted", False)}


def capture_failure(fleet, missing, *, withheld=None):
    reasons = []
    if fleet["phase"] != "complete" or not fleet.get("readback", {}).get("passed"):
        reasons.append(fleet.get("error") or "Explorer native observation readback did not pass")
    if missing:
        reasons.append(f"Native capture ended before these phases were recorded: {', '.join(missing)}")
    if withheld and "relocation_closed" in missing:
        reasons.append(withheld)
    return ". ".join(reasons)


def preserve_finished_prefix(config, trial, state, reason):
    marker = placement_boundary(state["card"]["family"])
    if marker not in state["markers"]:
        rejection = "no independently observed placement boundary"
        atomic(trial / "motor-intake.json", {"status": "excluded", "reason": rejection})
        return incomplete_story(trial, state, reason)
    try:
        prepare_native_arm(config, trial, state, motor_only=True)
    except (OSError, ValueError, RuntimeError, KeyError, ProtocolError) as error:
        record_arm_error(trial, error)
        reason = f"{reason}. Studio intake: {error}"
    return incomplete_story(trial, state, reason)


def record_arm_error(trial, error):
    retriable = isinstance(error, (StudioEpisodeUnlinked, OSError, ProtocolError))
    atomic(trial / "motor-intake.json", {"status": "pending_review" if retriable else "excluded", "reason": str(error)})


def project_ready(config, trial, state, fleet):
    through = anchor()["monotonic_ns"]
    try:
        prepare_native_arm(config, trial, state)
    except (OSError, ValueError, KeyError) as error:
        record_arm_error(trial, error)
        return incomplete_story(trial, state, f"Native Studio placement input could not be validated: {error}")
    try:
        result = admit_sources(config, trial, state, through, fleet)
    except (OSError, ValueError, RuntimeError, KeyError, ProtocolError) as error:
        return incomplete_story(trial, state, str(error))
    state.update(status="pending_reveal", decision=result, query_ns=through)
    atomic(trial / "trial.json", state)
    return result


def validate_pending_decision(recorder, trial, state):
    root = recorder.root
    metadata = {**recorder.meta, "sample_counts": recorder.counts}
    if (recorder.world_id != trial.name or metadata["run_mode"] != "hardware" or not metadata.get("studio_episode")
            or metadata["fleet_capture"] != state["fleet_capture_id"]):
        raise ValueError("The saved prediction belongs to a different story or native capture")
    verify_phase_digest(state["markers"], metadata.get(PHASE_DIGEST))
    decision, = rows(root / "decisions/prediction.jsonl")
    query, = rows(root / "communication/queries.jsonl")
    start = int(query["issued_time"]["monotonic_ns"])
    stop = int(decision["at"]["monotonic_ns"])
    if (query["issued_time"]["clock_id"] != recorder.clock or decision["at"]["clock_id"] != recorder.clock
            or not state["markers"]["relocation_closed"]["monotonic_ns"] <= start <= stop
            or query["text"] != QUERY or decision["agent_id"] != AGENTS[2]):
        raise ValueError("The saved query or decision does not fit the native story interval")
    messages = rows(root / DELIVERIES)
    for agent, cameras in recorder.cameras.items():
        validate_agent(root, metadata, agent, cameras, start, recorder.clock)
    witnessed = {agent: {camera["frame_id"] for packet in rows(root / POLICY / f"{agent}.jsonl")
                         for camera in packet["cameras"]} for agent in AGENTS[:2]}
    if any(not set(item["evidence"]["frame_ids"]) <= witnessed[item["evidence"]["origin_agent"]] for item in messages):
        raise ValueError("The recovered prediction cites evidence outside the witness's recorded frames")
    if predict(messages, stop) != decision["query_all"]:
        raise ValueError("The recorded messages do not reproduce the saved decision")
    policy_manifest = {**metadata, "query_boundary": query["issued_time"]}
    verify_learned_prediction(root, policy_manifest, decision)
    verify_decision_reference(root, policy_manifest, decision)
    return decision, start


def recover_pending_prediction(trial, state):
    root = trial / "private-replay"
    if not (root / "decisions/prediction.jsonl").is_file():
        return None
    try:
        recorder = Recorder.reopen_pending(root)
        decision, query = validate_pending_decision(recorder, trial, state)
    except (OSError, ValueError, KeyError, ProtocolError) as error:
        return incomplete_story(trial, state, f"Stored prediction could not be validated: {error}")
    intake = trial / "motor-intake.json"
    accepted = intake.is_file() and load(intake)["status"] == "reviewable_prefix"
    state.update(status="pending_reveal", native_episode_id=recorder.meta["studio_episode"],
                 decision=decision, query_ns=query, motor_segment_admitted=accepted)
    atomic(trial / "trial.json", state)
    return decision


@story_update
def predict_selected(config, identifier):
    card_for(config, identifier)
    trial = story_workspace(config) / identifier
    state = restore_start(config, trial, load(trial / "trial.json"))
    if state["status"] in ("placement_recording", "recording", "pending_reveal") and not same_recording_boot(state):
        if state["status"] == "placement_recording":
            return preserve_finished_prefix(config, trial, state, "Homebrain rebooted before the mobile observation phase")
        if state["status"] == "recording":
            recover_pending_prediction(trial, state)
        return incomplete_story(trial, state, "Homebrain rebooted before recording and reveal completed")
    if state["status"] in ("pending_reveal", "complete"):
        return state["decision"]
    if state["status"] == "incomplete":
        return incomplete_result(state)
    if state["status"] == "placement_recording":
        raise ValueError("Save and stop Studio, then resume this card in the launcher to start Explorer")
    if state["status"] != "recording":
        raise ValueError("Wait for the existing native capture to acknowledge start")
    saved = recover_pending_prediction(trial, state)
    if saved is not None:
        return saved
    native_controls_idle(native_status(config))
    fleet = fleet_manifest(config, state["fleet_capture_id"])
    if fleet["phase"] not in FLEET_TERMINAL:
        raise ValueError("Wait for the existing Explorer capture and readback to finish")
    missing = [phase for phase in story_phases(state) if phase not in state["markers"]]
    reason = capture_failure(fleet, missing, withheld=state.get("relocation_prompt_withheld"))
    if reason:
        return preserve_finished_prefix(config, trial, state, reason)
    return project_ready(config, trial, state, fleet)


def recover_sealed_reveal(trial, state):
    root = trial / "private-replay"
    sealed = root / "manifest.json"
    if state["status"] != "pending_reveal" or not sealed.is_file():
        return state
    manifest = load(sealed)
    decision = state["decision"]
    if (manifest["run_mode"] != "hardware" or manifest["world_episode_id"] != trial.name
            or manifest["decision_boundary"] != decision["at"]
            or int(manifest["query_boundary"]["monotonic_ns"]) != state["query_ns"]
            or rows(root / "decisions/prediction.jsonl") != [decision]):
        raise ValueError("The sealed reveal does not belong to this story or decision")
    verify_phase_digest(state.get("markers", {}), manifest.get(PHASE_DIGEST))
    validate(root, manifest)
    reveal, = rows(root / TRUTH / "reveal.jsonl")
    validate_sealed_clip(root, reveal, manifest)
    if reveal["ground_truth"] not in OUTCOMES:
        raise ValueError("The sealed reveal contains an invalid outcome")
    state.update(status="complete", observed_terminal_outcome=reveal["ground_truth"],
                 recorded_at_ns=int(reveal["observed_at"]["monotonic_ns"]))
    atomic(trial / "trial.json", state)
    return state


@story_update
def terminal_reveal(config, identifier, outcome, preserved):
    card_for(config, identifier)
    trial = story_workspace(config) / identifier
    state = recover_sealed_reveal(trial, load(trial / "trial.json"))
    if state["status"] == "complete":
        if outcome != state["observed_terminal_outcome"]:
            raise ValueError("This story already has a sealed operator outcome")
        return state
    if state["status"] == "pending_reveal" and outcome == "relation_unverified" and not preserved:
        return unverified_reveal(trial, state)
    if state["status"] != "pending_reveal" or outcome not in OUTCOMES or not preserved:
        raise ValueError("A saved prediction and confirmation of unchanged object relation are required")
    native_controls_idle(native_status(config))
    recorder = Recorder.reopen_pending(trial / "private-replay")
    if recorder.clock != boot_clock():
        raise ValueError("Terminal reveal must use the same Homebrain boot clock")
    if "learned_evidence" in state["decision"]:
        saved, query = validate_pending_decision(recorder, trial, state)
        if saved != state["decision"] or query != state["query_ns"]:
            raise ValueError("The journal and the stored pre-reveal decision disagree")
    decision = int(state["decision"]["at"]["monotonic_ns"])
    samples = capture_clip(config, decision)
    native_controls_idle(native_status(config))
    ts = store_clip(recorder, samples, decision, outcome)
    atomic(recorder.root / LABELS / "settled-event.jsonl", {"settled_placement": None, "relocation_assignment": None,
                    "intermediate_events": None, "terminal": outcome if outcome != "unknown" else None,
                    "source": "attended terminal observation", "scope": "training_or_evaluation_only"})
    recorder.close(state["query_ns"], decision, ts)
    state.update(status="complete", observed_terminal_outcome=outcome, recorded_at_ns=ts)
    atomic(trial / "trial.json", state)
    return state


def unverified_reveal(trial, state, reason="Operator could not confirm the queried object relation was unchanged until reveal"):
    recorder = Recorder.reopen_pending(trial / "private-replay")
    recorder.quarantine(reason)
    incomplete_story(trial, state, reason)
    return state


@story_update
def recover_story(config, trial):
    state = restore_start(config, trial, load(trial / "trial.json"))
    if state["status"] == "recording":
        recover_pending_prediction(trial, state)
    return recover_sealed_reveal(trial, state)


@story_update
def close_interrupted_story(config, identifier, reason):
    reason = reason.strip()
    if not reason or len(reason) > 300 or not reason.isprintable():
        raise ValueError("Give an observed interruption reason within 300 characters")
    card_for(config, identifier)
    trial = story_workspace(config) / identifier
    journal = trial / "trial.json"
    if not journal.is_file():
        raise ValueError("Start a story before closing an interrupted attempt")
    state = recover_sealed_reveal(trial, load(journal))
    if state["status"] == "incomplete":
        return incomplete_result(state)
    if state["status"] == "complete":
        raise ValueError("The story already has a sealed outcome")
    if state["status"] not in ACTIVE_STORY_STATES:
        raise ValueError("Only an interrupted active story may be closed as incomplete")
    studio = native_status(config)
    if any(studio.get(key) for key in ("recording_active", "teleoperation_active", "policy_active", "native_arm_hardware_open")):
        raise ProtocolError("RESOURCE_BUSY", "Finish the current Studio recording and stop native control before closing this story")
    state = restore_start(config, trial, state)
    if state["status"] == "preparing":
        raise ProtocolError("RESOURCE_BUSY", "The existing Explorer launch is still pending. Wait for its readback")
    if state["status"] == "recording":
        if fleet_manifest(config, state["fleet_capture_id"])["phase"] not in FLEET_TERMINAL:
            raise ProtocolError("RESOURCE_BUSY", "Wait for the existing Explorer capture and readback to finish")
        recover_pending_prediction(trial, state)
        state = recover_sealed_reveal(trial, state)
    if state["status"] == "complete":
        raise ValueError("The story already has a sealed outcome")
    if state["status"] == "incomplete":
        return incomplete_result(state)
    reason = f"Operator closed interrupted attempt. {reason}"
    if state["status"] == "pending_reveal":
        return incomplete_result(unverified_reveal(trial, state, reason))
    return preserve_finished_prefix(config, trial, state, reason)
