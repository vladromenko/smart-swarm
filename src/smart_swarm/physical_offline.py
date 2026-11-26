"""Score a pre-reveal private input seal and retain a separate replay receipt."""

import time

from .episode import atomic, load, rows, sha
from .physical_learned import DEFERRED, blind_inference, verify_optional

FILE = "blind-offline-forecast.json"
PREDICTION = "decisions/prediction.jsonl"


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
    atomic(path, {**forecast, "input_seal": identity})
    return verified_replay(root, path, manifest, identity)


def observe_pending(config, attempted):
    root = config["data_root"] / "physical-stories"
    now = time.monotonic()
    for path in sorted(root.glob("*/trial.json")):
        trial = path.parent
        if attempted.get(trial.name, 0.) <= now and not (trial / FILE).is_file():
            state = load(path)
            learned = state.get("decision", {}).get("learned_evidence", {})
            if state["status"] == "complete" and learned.get("status") == DEFERRED:
                attempted[trial.name] = now + 60
                score_trial(config, trial)
                return trial.name
    return None
