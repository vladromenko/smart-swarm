"""Review completed story prefixes in one official Studio export per native source."""

from collections import defaultdict
import re
from subprocess import SubprocessError

from .contract import ProtocolError
from .episode import atomic, load, sha
from .physical_motor_prefix import export_storage_digest, save_prefix
from .physical_story_replay import prepare_native_arm, record_arm_error
from .physical_story_source import story_update, story_workspace
from .research_jobs import admitted
from .studio import DATASET_PATTERN, accepted_export, native_controls_idle, native_status, rejected_native_exports

REVIEW_ERRORS = (OSError, ValueError, RuntimeError, KeyError, ProtocolError, SubprocessError)


def pending_trials(root):
    for intake in sorted(root.glob("*/motor-intake.json")):
        record = load(intake)
        trial = intake.parent
        if record["status"] in ("pending_review", "reviewable_prefix"):
            state = load(trial / "trial.json")
            if state["status"] in ("complete", "incomplete"):
                if record["status"] == "pending_review" or rejected_prefix(trial, state, record):
                    yield trial, state


def rejected_prefix(trial, state, intake):
    path = trial / "motor-segment.json"
    dataset = state.get("studio_dataset_id", "")
    if intake.get("segment") == path.name and path.is_file() and re.fullmatch(DATASET_PATTERN, dataset):
        export = load(path)["native_export"]["path"]
        return rejected_native_exports(trial.parent / "_native-exports" / dataset).get(export)
    return None


def closed_block_trials(config, root):
    from .physical_story_plan import prepare
    from .physical_story_review import closed_card, ready_scopes

    pending = list(pending_trials(root))
    if not pending or any(not closed_card(path) for path in root.glob("*/trial.json")):
        return []
    scopes, _ = ready_scopes(prepare(config), root)
    closed = {card["story_id"] for block in scopes.values() for card in block}
    return [(trial, state) for trial, state in pending if trial.name in closed]


def motor_review_generation(config):
    pending = closed_block_trials(config, story_workspace(config))
    if not pending:
        return None
    datasets = {}
    for _, state in pending:
        dataset = state["studio_dataset_id"]
        if dataset not in datasets and re.fullmatch(DATASET_PATTERN, dataset):
            path = config["studio_project"] / "data" / dataset / "dataset.json"
            datasets[dataset] = sha(path) if path.is_file() else None
    return {"trials": {trial.name: sha(trial / "trial.json") for trial, _ in pending}, "catalogues": datasets}


def linked_sources(config, pending):
    sources = defaultdict(list)
    for trial, state in pending:
        intake = trial / "motor-intake.json"
        reason = rejected_prefix(trial, state, load(intake))
        if reason:
            atomic(intake, {"status": "pending_review", "reason": reason})
            state["motor_segment_admitted"] = False
            atomic(trial / "trial.json", state)
        if not state.get("native_story_validation"):
            try:
                prepare_native_arm(config, trial, state, motor_only=True)
            except REVIEW_ERRORS as error:
                record_arm_error(trial, error)
            atomic(trial / "trial.json", state)
        if load(trial / "motor-intake.json")["status"] == "pending_review" and state.get("native_story_validation"):
            sources[state["studio_dataset_id"]].append((trial, state))
    return sources


def review_dataset(config, root, dataset, trials):
    if not re.fullmatch(DATASET_PATTERN, dataset):
        raise ValueError("The motor review source is not a native Studio dataset ID")
    source = config["studio_project"] / "data" / dataset
    cache = root / "_native-exports" / dataset
    cache.mkdir(parents=True, exist_ok=True)
    destination, validation = accepted_export(config, source, cache)
    digest = export_storage_digest(destination)
    for trial, state in trials:
        state["motor_segment_admitted"] = save_prefix(config, trial, state, destination, storage_digest=digest)
        atomic(trial / "trial.json", state)
    return str(destination), validation.get("archive_storage")


def review_sources(config, root, pending, result):
    with admitted(config, "smart-swarm-physical-motor-intake", archive_only=True) as (_, heartbeat):
        result["reservation_acquired"] = True
        for dataset, trials in linked_sources(config, pending).items():
            try:
                heartbeat()
                export, storage = review_dataset(config, root, dataset, trials)
                result["exports"][dataset] = export
                if storage is not None:
                    result["export_storage"][dataset] = storage
            except REVIEW_ERRORS as error:
                result["errors"][dataset] = str(error)
                for trial, _ in trials:
                    path = trial / "motor-intake.json"
                    if load(path)["status"] == "pending_review":
                        atomic(path, {"status": "pending_review", "reason": str(error)})


def ready_for_archive_review(config):
    native = native_status(config)
    native_controls_idle(native)
    if native["native_arm_hardware_open"] or native["native_stop_check_required"]:
        raise ProtocolError("RESOURCE_BUSY", "Studio must finish native control and confirm STOP before motor review")


@story_update
def review_pending_motor(config, *, closed_blocks_only=False):
    root = story_workspace(config)
    pending = closed_block_trials(config, root) if closed_blocks_only else list(pending_trials(root))
    result = {"attempted": len(pending), "exports": {}, "export_storage": {}, "errors": {}, "states": {},
              "reservation_acquired": False}
    if not pending:
        return result
    reports = root / "_native-exports"
    reports.mkdir(exist_ok=True)
    if closed_blocks_only:
        try:
            ready_for_archive_review(config)
        except ProtocolError as error:
            if error.code == "RESOURCE_BUSY":
                return {**result, "attempted": 0, "deferred_reason": error.message}
            raise
    try:
        review_sources(config, root, pending, result)
    except REVIEW_ERRORS as error:
        result["errors"]["shared_reservation"] = str(error)
    result["states"] = {trial.name: load(trial / "motor-intake.json") for trial, _ in pending}
    atomic(reports / "last-review.json", result)
    return result
