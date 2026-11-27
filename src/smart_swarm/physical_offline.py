"""Score a pre-reveal private input seal and retain a separate replay receipt."""

from fcntl import LOCK_EX, LOCK_NB, flock
from pathlib import Path
from subprocess import TimeoutExpired
from tempfile import TemporaryDirectory
import time

from .contract import ProtocolError
from .episode import atomic, load, rows, sha
from .physical_learned import DEFERRED, OfflineWorkerTimeout, blind_inference, verify_optional
from .research_jobs import NativeComputeBusy

FILE = "blind-offline-forecast.json"
RETRY = "blind-offline-retry.json"
PREDICTION = "decisions/prediction.jsonl"
MAX_TIMEOUTS = 3
RETRY_SECONDS = 60
RETRYABLE = ("waiting_for_compute", "retry_pending")


def input_seal(root):
    manifest_path = root / "manifest.json"
    manifest = load(manifest_path)
    bound = manifest["sidecar_sha256"][PREDICTION]
    path = root / PREDICTION
    if sha(path) != bound:
        raise ValueError("The archived prediction record changed after sealing")
    decision, = rows(path)
    seal = verify_optional(root, manifest, decision)
    if seal["status"] != DEFERRED:
        raise ValueError("This story has no pre-reveal blind offline input seal")
    identity = {"decision_sha256": bound, "permitted_input_sha256": seal["permitted_input_sha256"]}
    return manifest, seal, identity


def verified_replay(root, path, manifest, identity):
    saved = load(path)
    if saved["input_seal"] != identity or saved["mode"] != "blind_offline" or saved["status"] != "scored":
        raise ValueError("The offline forecast is bound to a different sealed input")
    verify_optional(root, manifest, {"learned_evidence": saved})
    return saved


def read_replay(root):
    path = root.parent / FILE
    if path.is_file():
        manifest, _, identity = input_seal(root)
        return verified_replay(root, path, manifest, identity)
    return None


def score_trial(config, trial):
    root = trial / "private-replay"
    manifest, seal, identity = input_seal(root)
    path = trial / FILE
    if path.is_file():
        return verified_replay(root, path, manifest, identity)
    forecast = blind_inference(config, root, manifest, seal)
    return publish_replay(root, path, forecast, identity)


def publish_replay(root, path, forecast, identity):
    manifest, _, current = input_seal(root)
    if current != identity:
        raise ValueError("The pre-reveal input seal changed during offline scoring")
    with TemporaryDirectory(prefix=".forecast-", dir=path.parent) as directory:
        candidate = Path(directory) / FILE
        atomic(candidate, {**forecast, "input_seal": identity})
        verified_replay(root, candidate, manifest, identity)
        try:
            path.hardlink_to(candidate)
        except FileExistsError:
            pass
    return verified_replay(root, path, manifest, identity)


def retry_state(trial, seal):
    path = trial / RETRY
    identity = {key: seal.get(key) for key in ("permitted_input_sha256", "profiles_sha256")}
    prior = load(path) if path.is_file() else {}
    return prior if prior.get("seal") == identity else {"seal": identity, "worker_timeouts": 0}


def record_failure(trial, prior, error):
    state = {**prior, "status": "failed", "reason": str(error) or type(error).__name__}
    state.pop("next_retry_unix", None)
    if isinstance(error, OfflineWorkerTimeout):
        state["worker_timeouts"] += 1
        state["status"] = "retry_pending" if state["worker_timeouts"] < MAX_TIMEOUTS else "retry_exhausted"
    if isinstance(error, NativeComputeBusy) or (isinstance(error, ProtocolError) and error.code == "RESOURCE_BUSY"):
        state["status"] = "waiting_for_compute"
    if state["status"] in RETRYABLE:
        state["next_retry_unix"] = time.time() + RETRY_SECONDS
    atomic(trial / RETRY, state)


def try_pending(config, trial, seal):
    with (trial / ".blind-offline-dispatch.lock").open("a") as lock:
        try:
            flock(lock, LOCK_EX | LOCK_NB)
        except BlockingIOError:
            return False, None
        if (trial / FILE).is_file():
            return False, None
        prior = retry_state(trial, seal)
        if "status" in prior and (prior["status"] not in RETRYABLE or time.time() < prior["next_retry_unix"]):
            return False, None
        try:
            score_trial(config, trial)
        except (OSError, ValueError, RuntimeError, KeyError, TypeError, TimeoutExpired, ProtocolError) as error:
            record_failure(trial, prior, error)
            return True, None
        return True, trial.name


def observe_pending(config):
    root = config["data_root"] / "physical-stories"
    for path in sorted(root.glob("*/trial.json")):
        trial = path.parent
        if not (trial / FILE).is_file():
            state = load(path)
            learned = state.get("decision", {}).get("learned_evidence", {})
            if state["status"] == "complete" and learned.get("status") == DEFERRED:
                attempted, scored = try_pending(config, trial, learned)
                if attempted:
                    return scored
    return None
