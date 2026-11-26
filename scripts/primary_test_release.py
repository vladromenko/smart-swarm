"""Bind the reserved test recordings to one frozen profile before capture."""

from hashlib import sha256
import json

from primary_test_budget import TEST_LIMITS
from smart_swarm.episode import atomic, sha

RELEASE = "primary-test-release.json"


def release_identity(profile_path, profile):
    content = profile_path.read_bytes()
    if json.loads(content) != profile:
        raise ValueError("The frozen profile changed after validation")
    if profile["status"] != "frozen_for_primary_test":
        raise ValueError("Only released profiles can bind primary test recordings")
    return {"protocol": "smart-swarm-primary-test-release-v1", "profile_sha256": sha256(content).hexdigest(),
            "allocation_sha256": profile["source_allocation_sha256"], "config_sha256": profile["config_sha256"],
            "validation_sources": profile["validation_count"], "capture_limits": TEST_LIMITS,
            "checkpoints": {kind: fit["checkpoint_sha256"] for kind, fit in profile["fits"].items()}}


def bound_test_release(root, profile_path, profile, *, publish=False):
    identity = release_identity(profile_path, profile)
    receipt = root / RELEASE
    if receipt.is_file():
        if json.loads(receipt.read_bytes()) != identity:
            raise ValueError("This primary test has already been bound to a different frozen profile")
    elif publish:
        if root.exists() and any(root.iterdir()):
            raise ValueError("Existing test files lack their original release receipt")
        root.mkdir(parents=True, exist_ok=True)
        atomic(receipt, identity)
    else:
        raise ValueError("The primary test release receipt is absent")
    return sha(receipt)
