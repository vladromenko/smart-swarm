"""Preserve verified Smart Swarm Studio exports at their original reader paths."""

from fcntl import LOCK_EX, LOCK_NB, flock
from pathlib import Path
import re
from subprocess import SubprocessError

from .contract import ProtocolError
from .episode import atomic, load, sha
from .physical_archive_storage import retain_directory


def secondary_export_path(config, original):
    location = config.get("physical_archive_volume")
    volume = Path(location).resolve() if location else None
    if volume is None or not volume.is_mount() or volume.stat().st_dev == original.parent.stat().st_dev:
        raise OSError("The configured Smart Swarm secondary export volume is not mounted")
    return volume / "smart-swarm-preserved/physical-native/studio-exports" / original.name


def checked_export_path(config, original):
    base = config["studio_project"].resolve() / "runtime/smart-swarm-pilot-exports"
    if original.parent != base or base.resolve() != base:
        raise ValueError("The referenced export is outside the native Smart Swarm export directory")
    physical = original.resolve(strict=True)
    if physical != original:
        if (not original.is_symlink() or not re.fullmatch(r"[a-f0-9]{12}-[a-f0-9]{8}", original.name)
                or physical != secondary_export_path(config, original)):
            raise ValueError("The referenced export leaves its original or configured retained path")
    return physical


def preserve_export(config, source, original, validation, verify, prior):
    physical = checked_export_path(config, original)
    destination = secondary_export_path(config, original)
    lineage = physical / "provenance/export.json"
    metadata = load(lineage)
    metadata_sha = sha(lineage)
    if (not re.fullmatch(r"[a-f0-9]{12}-[a-f0-9]{8}", original.name)
            or metadata.get("status") != "complete" or metadata.get("source_path") != str(source.resolve())
            or metadata.get("dataset_sha256") != validation["dataset_sha256"]
            or metadata.get("official_verification", {}).get("rgb_pixels_bit_exact") is not True):
        raise ValueError("Secondary retention requires this source's verified native Studio export")
    if physical == destination and prior.get("state") == "retained" and prior.get("metadata_sha256") == metadata_sha:
        return prior

    def reopen():
        if verify() != validation:
            raise ValueError("The original Studio reader did not reopen the same official export")

    retained = retain_directory(original, destination, reopen, required="provenance/export.json")
    return {"state": "retained", "original_path": str(original), "storage_path": str(destination),
            "metadata_sha256": metadata_sha, **retained}


def archive_accepted_export(config, source, original, scope, validation, verify):
    if not config.get("physical_archive_volume"):
        return None
    path = scope / "archived-native-exports" / f"{original.name}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.with_suffix(".lock").open("a") as lock:
        try:
            flock(lock, LOCK_EX | LOCK_NB)
        except BlockingIOError:
            return {"state": "pending", "reason": "Another intake is preserving this native export"}
        prior = load(path) if path.is_file() else {}
        try:
            result = preserve_export(config, source, original, validation, verify, prior)
        except (OSError, ValueError, KeyError, ProtocolError, SubprocessError) as error:
            result = {"state": "pending", "original_path": str(original), "reason": str(error)}
        if result != prior:
            atomic(path, result)
        return result
