"""Preserve closed Smart Swarm Explorer archives on the mounted secondary volume."""

import ctypes
from fcntl import LOCK_EX, LOCK_NB, flock
from hashlib import file_digest, sha256
import os
from pathlib import Path
import re
import shutil
from tempfile import TemporaryDirectory

from .contract import json_bytes, ProtocolError
from .episode import atomic, load
from .physical_story_plan import PLAN_VERSION
from .physical_story_source import fleet_capacity, fleet_manifest, worker_identity

RECEIPT = "archive-storage.json"


def archive_members(root, *, durable=False, required="manifest.json"):
    files = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink() or not (path.is_dir() or path.is_file()):
            raise ValueError("A native archive member is outside the original file layout")
        if path.is_file():
            with path.open("rb") as stream:
                settings = os.fstat(stream.fileno())
                if durable:
                    os.fsync(stream.fileno())
                files[path.relative_to(root).as_posix()] = {"bytes": settings.st_size, "mtime_ns": settings.st_mtime_ns,
                    "sha256": file_digest(stream, "sha256").hexdigest()}
    if required not in files:
        raise ValueError("The closed native archive lacks its manifest")
    return files


def sync_directory(path):
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def exchange(path, other):
    rename = ctypes.CDLL(None, use_errno=True).renameat2
    rename.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
    rename.restype = ctypes.c_int
    if rename(-100, os.fsencode(path), -100, os.fsencode(other), 2):
        raise OSError(ctypes.get_errno(), "The native archive path could not be exchanged atomically")


def prepare_archive_parent(path):
    if path.resolve() != path:
        raise ValueError("The configured Smart Swarm archive directory contains a symbolic link")
    if not path.is_dir():
        prepare_archive_parent(path.parent)
        path.mkdir(exist_ok=True)
        sync_directory(path.parent)


def verified_copy(source, destination, expected, required):
    prepare_archive_parent(destination.parent)
    if destination.exists():
        if destination.is_symlink() or archive_members(destination, required=required) != expected:
            raise ValueError("The secondary archive path has different contents")
        return
    with TemporaryDirectory(prefix=".copying-", dir=destination.parent) as stage:
        copied = Path(stage) / "archive"
        shutil.copytree(source, copied)
        if (archive_members(copied, durable=True, required=required) != expected
                or archive_members(source, required=required) != expected):
            raise ValueError("The original native archive changed during verified copying")
        for directory in sorted((path for path in copied.rglob("*") if path.is_dir()), reverse=True):
            sync_directory(directory)
        sync_directory(copied)
        copied.rename(destination)
        sync_directory(destination.parent)


def retain_directory(source, destination, verify_native, *, required="manifest.json"):
    pending = source.parent / ".smart-swarm-archive-transfer"
    if pending.is_symlink():
        raise ValueError("The local archive transfer directory contains a symbolic link")
    backup = pending / source.name
    if source.is_symlink():
        if source.resolve(strict=True) != destination:
            raise ValueError("The original capture path points to an unexpected archive")
        expected = archive_members(destination, required=required)
        verify_native()
    else:
        expected = archive_members(source, required=required)
        verified_copy(source, destination, expected, required)
        pending.mkdir(exist_ok=True)
        if not backup.is_symlink():
            backup.symlink_to(destination, target_is_directory=True)
        if backup.resolve(strict=True) != destination or archive_members(source, required=required) != expected:
            raise ValueError("The original capture or prepared archive path changed before publication")
        exchange(source, backup)
        try:
            verify_native()
            if archive_members(backup, required=required) != expected:
                raise ValueError("The original native archive changed before the path was published")
            sync_directory(source.parent)
            sync_directory(pending)
        except BaseException:
            exchange(source, backup)
            raise
    if backup.exists():
        if (backup.is_symlink() or archive_members(backup, required=required) != expected
                or archive_members(destination, required=required) != expected):
            raise ValueError("The retained native archive differs from the local copy")
        shutil.rmtree(backup)
        sync_directory(pending)
    return {"file_count": len(expected), "bytes": sum(item["bytes"] for item in expected.values()),
            "archive_sha256": sha256(json_bytes(expected)).hexdigest()}


def owned_capture(trial, state):
    identifier = state.get("fleet_capture_id")
    start = trial / "fleet-start.json"
    worker = trial / "fleet-launch.json"
    if not identifier or not re.fullmatch(r"[a-f0-9]{32}", identifier) or not start.is_file() or not worker.is_file():
        return None
    acknowledgement = load(start)
    launch = load(worker)
    acknowledged = acknowledgement.get("acknowledged")
    pid, started = launch.get("pid"), launch.get("started_ticks")
    if (acknowledgement.get("id") != identifier or not acknowledged or acknowledged != state.get("fleet_acknowledgement")
            or type(pid) is not int or pid < 1 or not isinstance(started, str) or not started.isdigit()
            or worker_identity(pid) == launch):
        return None
    return identifier


def retain_capture(config, trial, identifier, volume):
    with (trial / ".archive-storage.lock").open("a") as lock:
        try:
            flock(lock, LOCK_EX | LOCK_NB)
        except BlockingIOError:
            return None
        manifest = fleet_manifest(config, identifier)
        if manifest["id"] != identifier or manifest["phase"] != "complete" or not manifest.get("readback", {}).get("passed"):
            return None
        storage = fleet_capacity(config)
        if not storage or "readback_error" in storage:
            raise ValueError("The installed Explorer capture directory is unavailable")
        source = Path(storage["directory"]).resolve(strict=True) / identifier
        if not volume.is_mount() or volume.stat().st_dev == source.parent.stat().st_dev:
            raise OSError("The configured Smart Swarm secondary archive volume is not mounted")
        destination = volume / "smart-swarm-preserved/physical-native/explorer" / identifier
        if load(source / "manifest.json") != manifest:
            raise ValueError("The native capture path and Explorer readback disagree")

        def verify_native():
            if fleet_manifest(config, identifier) != manifest:
                raise ValueError("Explorer did not reopen the preserved native capture by its original ID")

        result = retain_directory(source, destination, verify_native)
        return {"state": "retained", "capture_id": identifier, "native_path": str(source),
                "storage_path": str(destination), **result}


def retain_closed_captures(config):
    location = config.get("physical_archive_volume")
    if not location:
        return []
    root = config["data_root"]
    prepared = root / "collection-plans" / f"{PLAN_VERSION}.json"
    if not prepared.is_file():
        return []
    results = []
    for card in load(prepared)["cards"]:
        trial = root / "physical-stories" / card["story_id"]
        journal, receipt = trial / "trial.json", trial / RECEIPT
        if journal.is_file():
            state = load(journal)
            prior = load(receipt) if receipt.is_file() else {}
            if state.get("card") == card and state.get("status") in ("complete", "incomplete") and prior.get("state") != "retained":
                identifier = owned_capture(trial, state)
                if identifier:
                    try:
                        result = retain_capture(config, trial, identifier, Path(location).resolve())
                    except (OSError, ValueError, KeyError, ProtocolError) as error:
                        result = {"state": "pending", "capture_id": identifier, "reason": str(error)}
                    if result is not None and prior != result:
                        atomic(receipt, result)
                        results.append(result)
    return results
