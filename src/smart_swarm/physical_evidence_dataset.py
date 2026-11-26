"""Publish native physical stories and provide a label-free policy replay view."""

from collections import Counter
from contextlib import contextmanager
from hashlib import sha256
from pathlib import Path
import re

from .contract import ProtocolError, json_bytes
from .datasets import audit_lineage
from .episode import DELIVERIES, LABELS, POLICY, TRUTH, Replay, atomic, load, rows, sha, validate
from .evidence_inputs import runtime_signature
from .memory import QUERY
from .physical_learned import DEFERRED, verify_optional as verify_learned_prediction
from .physical_motor_prefix import episode_directory
from .physical_offline import read_replay
from .physical_reveal import validate_sealed_clip
from .physical_slot_reference import SCOPE_FLAG, verify_decision_reference
from .physical_story_annotations import apply_reviews
from .physical_story_condition import PHASE_DIGEST, audit_condition
from .physical_story_plan import prepare
from .physical_story_replay import AGENTS, OUTCOMES
from .physical_story_source import mobile_phase_bounds
from .primary_inputs import primary_policy_view

SCHEMA = "smart-swarm-physical-evidence-corpus-v1"
QUERY_FILE = "communication/queries.jsonl"
DECISION_FILE = "decisions/prediction.jsonl"
LABEL_FILE = f"{LABELS}/settled-event.jsonl"
REVEAL_FILE = f"{TRUTH}/reveal.jsonl"
REQUIRED_SIDECARS = {QUERY_FILE, DELIVERIES, DECISION_FILE, LABEL_FILE, REVEAL_FILE,
                     *(f"{POLICY}/{agent}.jsonl" for agent in AGENTS)}
BASE_CAMERAS = {AGENTS[0]: [f"{AGENTS[0]}/wrist", f"{AGENTS[0]}/top"],
                **{agent: [f"{agent}/front"] for agent in AGENTS[1:]}}


def verify_studio_source(config, state, manifest):
    proof = state["native_story_validation"]
    if (proof["source"] != "hardware" or proof["technical_only"] is not False
            or proof != manifest["native_validation"]):
        raise ValueError("The saved native source is technical or its validation receipt changed")
    if (proof["dataset_id"] != state["studio_dataset_id"]
            or not proof["episode_id"] == state["native_episode_id"] == manifest["studio_episode"]):
        raise ValueError("The original Studio episode does not match the recorded story")
    source = episode_directory(config, state)
    hashes = {"frames.jsonl": proof["frames_sha256"], "episode.json": proof["episode_metadata_sha256"]}
    if any(sha(source / name) != checksum for name, checksum in hashes.items()):
        raise ValueError("Validated Studio source records have changed")
    dataset = load(source.parent.parent / "dataset.json")
    if dataset["source"] != "hardware" or dataset["dataset_id"] != state["studio_dataset_id"]:
        raise ValueError("The native Studio dataset identity is no longer correct")
    return proof["frames_sha256"]


def verify_role_scopes(root, manifest):
    hidden = manifest["family"] == "F6"
    expected = {**BASE_CAMERAS, AGENTS[0]: [] if hidden else BASE_CAMERAS[AGENTS[0]]}
    counts = dict(zip(AGENTS, (1 if hidden else 8, 8, 1), strict=True))
    if manifest["policy_cameras"] != expected or manifest["sample_counts"] != counts:
        raise ValueError("The sealed story differs from the declared private RGB windows")
    if hidden:
        arm, = rows(root / POLICY / f"{AGENTS[0]}.jsonl")
        if arm["measured_state"] or arm["state_units"] or arm["applied_command_ref"] is not None:
            raise ValueError("An F6 arm packet leaked measurements or a command reference")
    if not REQUIRED_SIDECARS <= set(manifest["sidecar_sha256"]):
        raise ValueError("The sealed physical story is missing a required sidecar digest")
    validate(root, manifest)


def verify_fleet_windows(trial, root, journal):
    audit = load(trial / "fleet-timing-audit.json")
    tank, receiver = mobile_phase_bounds(journal, journal["query_ns"])
    scopes = ((AGENTS[1], *tank, 8), (AGENTS[2], *receiver, 1))
    for agent, early, late, required in scopes:
        proof = audit[agent]
        if (proof["permitted_interval_ns"] != [early, late] or proof["selected"] != required
                or not proof["clock_status_samples"] or proof["eligible"] < required):
            raise ValueError("The sealed mobile capture differs from its clock admission record")
        for packet in rows(root / POLICY / f"{agent}.jsonl"):
            camera, = packet["cameras"]
            bounds = camera["capture_interval"]
            begin, end = [int(bounds[key]) for key in ("start_ns", "end_ns")]
            received = int(packet["receive_time"]["monotonic_ns"])
            if not early <= begin <= end <= received <= late or camera["exposure_time"] is not None:
                raise ValueError("A mobile exposure or receipt falls outside its private phase")


def verify_decision(root, journal):
    prediction, = rows(root / DECISION_FILE)
    query, = rows(root / QUERY_FILE)
    receiver = Replay(root, AGENTS[-1])
    if (prediction != journal["decision"] or prediction["agent_id"] != AGENTS[-1]
            or query["origin_agent"] != AGENTS[-1] or query["recipients"] != list(AGENTS[:2])
            or prediction["query"] != QUERY or query["text"] != QUERY
            or query["issued_time"]["clock_id"] != receiver.clock
            or int(query["issued_time"]["monotonic_ns"]) != receiver.query
            or prediction["at"]["clock_id"] != receiver.clock or receiver.predict() != prediction["query_all"]
            or receiver.query != journal["query_ns"] or receiver.decision != int(prediction["at"]["monotonic_ns"])):
        raise ValueError("The saved decision does not reproduce from the scoped received messages")
    for row in receiver.deliveries():
        report = row["evidence"]
        agent = report["origin_agent"]
        if agent not in AGENTS[:2]:
            raise ValueError("A physical story message does not originate from a permitted witness")
        source = Replay(root, agent)
        witnesses = {view["frame_id"] for packet in source.packets() for view in packet["cameras"]}
        if not set(report["frame_ids"]) <= witnesses:
            raise ValueError("A message cites frames outside its own witness scope")
    return prediction["query_all"]["decision"]


def verify_terminal(root, journal, manifest):
    terminal, = rows(root / REVEAL_FILE)
    sample_count = validate_sealed_clip(root, terminal, manifest)
    labels, = rows(root / LABEL_FILE)
    outcome = terminal["ground_truth"]
    if (outcome not in OUTCOMES or outcome != journal["observed_terminal_outcome"]
            or labels["terminal"] != (None if outcome == "unknown" else outcome)
            or labels["scope"] != "training_or_evaluation_only" or labels["source"] != "attended terminal observation"):
        raise ValueError("The sealed operator label and evaluator observation disagree")
    for event in ("settled_placement", "relocation_assignment", "intermediate_events"):
        if labels[event] is not None:
            raise ValueError("A physical story filled unobserved event labels")
    return {key: labels[key] for key in ("terminal", "settled_placement", "relocation_assignment", "intermediate_events")}, sample_count


def check_link(journal, manifest, card):
    if (journal["card"] != card or manifest["world_episode_id"] != card["story_id"]
            or manifest["scene_group_id"] != card["source_group_id"]
            or not manifest["lineage"]["split"] == card["split"] == "development"):
        raise ValueError("The sealed story identity or development split differs from its original card")
    capture = journal["fleet_capture_id"]
    if (manifest["run_mode"] != "hardware" or manifest["source_type"] != "native_hardware"
            or manifest["family"] != card["family"] or manifest["fleet_capture"] != capture
            or re.fullmatch(r"[a-f0-9]{32}", capture) is None):
        raise ValueError("The physical story lacks its original native observation source")
    if (manifest["lineage"]["source_world_ids"]
            or manifest["native_recording_clock"]["host_id"] != "homebrain"
            or journal.get("homebrain_boot_id") != manifest["native_recording_clock"]["boot_id"]):
        raise ValueError("The native story has different split, source lineage or boot identity")


def load_admitted(config, trial, card, journal):
    root = trial / "private-replay"
    sealed = root / "manifest.json"
    if root.resolve().parent != trial.resolve() or journal["status"] != "complete":
        raise ValueError("Only a sealed, local, completed story can enter the corpus")
    manifest = load(sealed)
    check_link(journal, manifest, card)
    digest = verify_studio_source(config, journal, manifest)
    verify_fleet_windows(trial, root, journal)
    choice = verify_decision(root, journal)
    verify_decision_reference(root, manifest, journal["decision"])
    labels, reveal_samples = verify_terminal(root, journal, manifest)
    assistance = manifest["operator_intervention_by_phase"]
    if assistance != {phase: mark["assistance"] for phase, mark in journal["markers"].items()}:
        raise ValueError("The recorded human or robot assistance changed")
    outcomes = {phase: mark["observed_placement_outcome"] for phase, mark in journal["markers"].items()
                if mark.get("observed_placement_outcome") is not None}
    record = {"story_id": card["story_id"], "family": card["family"], "scene_group_id": card["source_group_id"],
              "split": card["split"], "native_studio_episode": journal["native_episode_id"], "native_studio_frames_sha256": digest,
              "native_explorer_capture": journal["fleet_capture_id"], "path": str(root), "manifest_sha256": sha(sealed),
              "camera_packets": manifest["sample_counts"], "labels": labels, "pre_inspection_decision": choice,
              "evaluator_reveal_samples": 1 if reveal_samples is None else reveal_samples,
              "assistance_by_phase": assistance, "attended_placement_outcomes": outcomes}
    if "learned_evidence" in journal["decision"]:
        record["learned_evidence"] = verify_learned_prediction(root, manifest, journal["decision"])
        if record["learned_evidence"]["status"] == DEFERRED:
            try:
                record["blind_offline_forecast"] = read_replay(root)
            except (OSError, ValueError, KeyError) as error:
                record["blind_offline_forecast_error"] = str(error)
    with policy_view(record) as scoped:
        record["permitted_input_sha256"] = sha256(json_bytes(physical_policy_signature(scoped))).hexdigest()
    reviewed = apply_reviews(trial, root, journal, record)
    audit = audit_condition(reviewed, journal["markers"], manifest.get(PHASE_DIGEST))
    return {**reviewed, "condition_audit": audit}


def inventory(config, workspace, cards):
    complete, excluded = [], []
    for card in cards:
        trial = workspace / card["story_id"]
        path = trial / "trial.json"
        if path.is_file():
            try:
                state = load(path)
                if state["status"] == "complete":
                    complete.append(load_admitted(config, trial, card, state))
                else:
                    reason = state.get("reason") or "Native story capture or its reveal is incomplete"
                    excluded.append(exclusion(card["story_id"], state["status"], reason))
            except (OSError, ValueError, KeyError, TypeError, ProtocolError) as error:
                excluded.append(exclusion(card["story_id"], "rejected", str(error)))
    identities = ("native_studio_episode", "native_studio_frames_sha256", "native_explorer_capture")
    counts = {key: Counter(row[key] for row in complete) for key in identities}
    admitted = []
    for row in complete:
        repeated = [key for key, uses in counts.items() if uses[row[key]] > 1]
        if repeated:
            excluded.append(exclusion(row["story_id"], "rejected", "Repeated native source " + ", ".join(repeated)))
        else:
            admitted.append(row)
    return admitted, excluded


def exclusion(identifier, status, reason):
    return {"story_id": identifier, "status": status, "reason": reason}


def describe(records):
    manifests = [{"world_episode_id": row["story_id"], "scene_group_id": row["scene_group_id"],
                  "lineage": {"split": row["split"], "source_world_ids": []}} for row in records]
    lineages = audit_lineage(manifests)
    fields = ("terminal", "settled_placement", "relocation_assignment")
    counts = {key: dict(Counter(row["labels"][key] or "unknown" for row in records)) for key in fields}
    counts["intermediate_events"] = dict(Counter("confirmed" if row["labels"]["intermediate_events"] else "unknown"
                                                for row in records))
    counts["first_insertion"] = dict(Counter((row["labels"]["intermediate_events"] or {}).get("first_insertion", "unknown")
                                            for row in records if row["family"] == "F3"))
    observed = [row for row in records if row.get("condition_audit", {}).get("status") == "observed"]
    return {"schema": SCHEMA, "scope": "native physical development data", "admitted_histories": lineages["worlds"],
            "independent_source_groups": lineages["scene_groups"], "histories_by_split": lineages["worlds_by_split"],
            "histories_by_family": dict(Counter(row["family"] for row in records)), "label_coverage": counts,
            "observed_conditions_by_family": dict(Counter(row["family"] for row in observed)),
            "condition_status_counts": dict(Counter(row.get("condition_audit", {}).get("status", "unverified") for row in records)),
            "terminal_label_count": sum(row["labels"]["terminal"] is not None for row in records),
            "records": records}


def publish(config):
    plan = prepare(config)
    root = config["data_root"] / "physical-stories"
    records, disallowed = inventory(config, root, plan["cards"])
    data = describe(records)
    data["plan_sha256"] = sha(Path(plan["plan_path"]))
    path = None
    if records:
        digest = sha256(json_bytes(data) + b"\n").hexdigest()
        directory = root / "evidence-datasets"
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{digest}.json"
        if not path.exists():
            atomic(path, data)
        if sha(path) != digest:
            raise ValueError("Immutable physical corpus content no longer matches its file identity")
    return {**data, "manifest": str(path) if path else None, "planned": len(plan["cards"]),
            "attempted": len(records) + len(disallowed), "exclusions": disallowed,
            "exclusion_counts": dict(Counter(row["status"] for row in disallowed))}


def physical_policy_signature(root):
    signature = runtime_signature(root)
    if load(root / "manifest.json").get(SCOPE_FLAG):
        signature[SCOPE_FLAG] = True
    return signature


@contextmanager
def policy_view(record):
    root = Path(record["path"])
    path = root / "manifest.json"
    if sha(path) != record["manifest_sha256"]:
        raise ValueError("The sealed physical recording manifest changed after corpus admission")
    manifest = load(path)
    proof = manifest["native_validation"]
    if (manifest["world_episode_id"] != record["story_id"] or manifest["lineage"]["split"] != record["split"]
            or manifest["scene_group_id"] != record["scene_group_id"]
            or manifest["source_type"] != "native_hardware" or proof["source"] != "hardware"
            or proof["technical_only"] is not False):
        raise ValueError("The immutable corpus points to another recording")
    verify_role_scopes(root, manifest)
    with primary_policy_view(root) as scoped:
        target = scoped / "manifest.json"
        if manifest.get(SCOPE_FLAG):
            atomic(target, {**load(target), SCOPE_FLAG: True})
        serial = json_bytes(physical_policy_signature(scoped))
        digest = sha256(serial).hexdigest()
        if serial != json_bytes(physical_policy_signature(root)) or record.get("permitted_input_sha256", digest) != digest:
            raise ValueError("The physical runtime view differs from the original permitted record")
        yield scoped
