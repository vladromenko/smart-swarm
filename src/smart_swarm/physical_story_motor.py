"""Review completed story prefixes in one official Studio export per native source."""

from collections import defaultdict
import re
from subprocess import SubprocessError

from .contract import ProtocolError
from .episode import atomic, load
from .physical_motor_prefix import export_storage_digest, save_prefix
from .physical_story_replay import prepare_native_arm, record_arm_error
from .physical_story_source import story_workspace
from .research_jobs import admitted
from .studio import DATASET_PATTERN, accepted_export

REVIEW_ERRORS = (OSError, ValueError, RuntimeError, KeyError, ProtocolError, SubprocessError)


def pending_trials(root):
    for intake in sorted(root.glob("*/motor-intake.json")):
        if load(intake)["status"] == "pending_review":
            trial = intake.parent
            state = load(trial / "trial.json")
            if state["status"] in ("complete", "incomplete"):
                yield trial, state


def linked_sources(config, pending):
    sources = defaultdict(list)
    for trial, state in pending:
        if not state.get("native_story_validation"):
            try:
                prepare_native_arm(config, trial, state)
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
    destination, _ = accepted_export(config, source, cache)
    digest = export_storage_digest(destination)
    for trial, state in trials:
        state["motor_segment_admitted"] = save_prefix(config, trial, state, destination, storage_digest=digest)
        atomic(trial / "trial.json", state)
    return str(destination)


def review_sources(config, root, pending, result):
    with admitted(config, "smart-swarm-physical-motor-intake") as (_, heartbeat):
        for dataset, trials in linked_sources(config, pending).items():
            try:
                heartbeat()
                result["exports"][dataset] = review_dataset(config, root, dataset, trials)
            except REVIEW_ERRORS as error:
                result["errors"][dataset] = str(error)
                for trial, _ in trials:
                    path = trial / "motor-intake.json"
                    if load(path)["status"] == "pending_review":
                        atomic(path, {"status": "pending_review", "reason": str(error)})


def review_pending_motor(config):
    root = story_workspace(config)
    pending = list(pending_trials(root))
    result = {"attempted": len(pending), "exports": {}, "errors": {}, "states": {}}
    if not pending:
        return result
    reports = root / "_native-exports"
    reports.mkdir(exist_ok=True)
    try:
        review_sources(config, root, pending, result)
    except REVIEW_ERRORS as error:
        result["errors"]["shared_reservation"] = str(error)
    result["states"] = {trial.name: load(trial / "motor-intake.json") for trial, _ in pending}
    atomic(reports / "last-review.json", result)
    return result
