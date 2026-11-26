"""Read-only Smart Swarm bindings to Explorer sensor status and native archives."""

import hashlib
import json
from urllib.error import HTTPError
from urllib.request import ProxyHandler, Request, build_opener
import re

from .contract import ProtocolError
from .platform_client import NoRedirect
from .studio import native_status

ROLES = {"tank-1": "tank", "rosmaster-1": "rosmaster"}
MAX_LIVE_RGB_AGE_S = 1.0
MAX_LIVE_CLOCK_UNCERTAINTY_MS = 50.0


def fetch(config, endpoint):
    origin = config["explorer_origin"]
    if origin != "http://127.0.0.1:8030":
        raise ValueError("This adapter expects Explorer on Homebrain loopback")
    req = Request(origin + endpoint)
    with build_opener(ProxyHandler({}), NoRedirect()).open(req, timeout=5) as response:
        return response.read(16 * 1024**2)


def data(config, endpoint):
    return json.loads(fetch(config, endpoint))


def rgb_readiness(fleet):
    issues = []
    roles = {}
    if fleet.get("running") is not True or fleet.get("read_only") is not True:
        issues.append("The native read-only observation hub is unavailable")
    for role in ROLES.values():
        row = fleet.get("robots", {}).get(role, {})
        age = row.get("ages_s", {}).get("rgb")
        uncertainty = row.get("uncertainty_ms")
        camera_ok = type(age) in (float, int) and 0 <= age <= MAX_LIVE_RGB_AGE_S
        clock_ok = type(uncertainty) in (float, int) and 0 <= uncertainty <= MAX_LIVE_CLOCK_UNCERTAINTY_MS
        if row.get("error"):
            issues.append(f"{role}: {row['error']}")
        if not camera_ok:
            issues.append(f"{role}: no fresh native RGB sample")
        if not clock_ok:
            issues.append(f"{role}: no native clock estimate within {MAX_LIVE_CLOCK_UNCERTAINTY_MS:g} ms")
        roles[role] = {"rgb_age_s": age, "clock_uncertainty_ms": uncertainty,
                       "rgb_fresh": camera_ok, "clock_within_limit": clock_ok, "source_error": row.get("error")}
    return {"ready": not issues, "issues": issues, "roles": roles}


def rosmaster_status(config):
    try:
        data(config, "/api/rosmaster/observation")
        return 200
    except HTTPError as error:
        return error.code


def native_team_status(config, *, probe_rosmaster=True):
    fleet = data(config, "/api/fleet/live")
    captures = data(config, "/api/fleet/captures")
    rgb = rgb_readiness(fleet)
    return {"so101_studio": native_status(config), "explorer_fleet_read_only": fleet.get("read_only"),
            "tank_and_rosmaster_clocks_and_views_ready": rgb["ready"], "story_observation_readiness": rgb,
            "sensor_join_issues": fleet["readiness"]["problems"],
            "rosmaster_observation_http_status": rosmaster_status(config) if probe_rosmaster else None,
            "active_native_capture": captures["active"],
            "available_native_captures": [{"id": row["id"], "phase": row["phase"]}
                                          for row in captures["records"]],
            "physical_team_verified": False, "native_motion_requested": False}


class FleetArchive:
    """Expose one agent's scoped, already recorded Explorer RGB stream."""

    def __init__(self, config, identifier, agent):
        if re.fullmatch(r"[a-f0-9]{32}", identifier) is None or agent not in ROLES:
            raise ValueError("Invalid native capture or agent ID")
        self.config = config
        self.agent = agent
        self.root = f"/api/fleet/captures/{identifier}/"
        self.role = ROLES[agent]
        manifest = data(config, self.root + "manifest.json")
        if manifest["phase"] != "complete":
            raise ValueError("Native capture is not finalized")
        self.status_count = manifest.get("status_samples")
        replay = data(config, self.root + "replay.json")
        self.frames = [self.project(frame) for frame in replay["frames"]
                       if f"{self.role}/rgb" in frame["views"]]

    def project(self, frame):
        sample = frame["views"][f"{self.role}/rgb"]
        name = sample["file"]
        if re.fullmatch(rf"preview-\d{{3}}-{self.role}-rgb\.jpg", name) is None:
            raise ValueError("Capture frame reference is outside the agent's native RGB role")
        return {"frame_ref": name, "after_s": frame["after_s"], "age_s": sample["age_s"],
                "source_unix_ns": sample.get("source_timestamp_ns"), "received_unix_ns": sample.get("received_ns")}

    def packets(self, *, through_s):
        return [row.copy() for row in self.frames if row["after_s"] <= through_s]

    def clock_history(self):
        history = data(self.config, self.root + "clock-status.json")
        if not isinstance(history, list) or not history or len(history) != self.status_count:
            raise ValueError("Native clock history differs from the original capture manifest")
        return history

    def image(self, frame_ref, *, through_s):
        allowed = {row["frame_ref"] for row in self.packets(through_s=through_s)}
        if frame_ref not in allowed:
            raise ProtocolError("FORBIDDEN", "Native camera frame is outside this agent's role or decision time")
        return fetch(self.config, self.root + frame_ref)


def archive_rehearsal(config, identifier):
    result = {}
    for agent in ROLES:
        client = FleetArchive(config, identifier, agent)
        allowed = client.packets(through_s=30.0)
        if not allowed:
            raise ValueError(f"No recorded private frames for {agent}")
        frame = allowed[0]
        private = client.image(frame["frame_ref"], through_s=frame["after_s"])
        result[agent] = {"available_frames": len(allowed), "first_rgb_sha256": hashlib.sha256(private).hexdigest(),
                         "first_rgb_file": frame["frame_ref"]}
    tank = FleetArchive(config, identifier, "tank-1")
    try:
        tank.image(result["rosmaster-1"]["first_rgb_file"], through_s=30.)
        raise RuntimeError("Cross-robot archive frame was incorrectly reachable")
    except ProtocolError as error:
        result["cross_role_guard"] = error.code
    return {"native_capture_id": identifier, "private_replay": result,
            "evidence_scope": "native observation archive compatibility; no Smart Swarm story labels"}
