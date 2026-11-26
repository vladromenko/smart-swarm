"""Save an attended setup aid from the existing idle camera-only previews."""

from datetime import datetime, timezone
from hashlib import sha256
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
import time
from uuid import uuid4

from PIL import Image

from .contract import ProtocolError
from .episode import atomic
from .physical_reveal import MAX_NATIVE_AGE_NS, STREAM_FIELDS, read_preview_jpeg
from .so101_pilot import CAMERAS
from .studio import STOP_UNCONFIRMED, native_request, preview_status

ROLES = tuple(sensor.rsplit("/", 1)[-1] for sensor in CAMERAS)


def snapshot(config):
    native = native_request(config, "/api/status")
    preview = native["preview"]
    controls = (native["recording"], native["teleoperation"])
    policy = any(job.get("kind") == "policy" and job.get("state") in ("running", "queued", "waiting_resources")
                 for job in native["jobs"])
    if policy or any(row.get("active") or row.get("hardware_open") or row.get("hardware_stop_state") == STOP_UNCONFIRMED
                     for row in controls):
        raise ProtocolError("RESOURCE_BUSY", "Use framing review while Studio control and recording are idle and STOP is confirmed")
    status = preview_status(preview)
    if not all(status[role]["fresh"] for role in ROLES):
        raise ProtocolError("OFFLINE", "Connect the wrist and top camera-only previews in Studio before framing review")
    cameras = {}
    for role in ROLES:
        entry = preview["cameras"][role].get("stream") or {}
        if (not all(isinstance(entry.get(key), str) and entry[key] for key in ("device", "device_resolved", "timestamp_basis"))
                or not all(type(entry.get(key)) is int and entry[key] > 0 for key in ("width", "height"))):
            raise ValueError(f"The native {role} preview lacks a device identity or image geometry")
        cameras[role] = {**status[role], "stream": {key: entry[key] for key in STREAM_FIELDS}}
    generation = native.get("hardware_stop_generation")
    if (len({row["stream"]["device_resolved"] for row in cameras.values()}) != len(ROLES)
            or any(type(row["frames_received"]) is not int or row["frames_received"] < 1 for row in cameras.values())
            or type(generation) is not int or generation < 0):
        raise ValueError("The wrist and top previews lack distinct native device identities and valid frame counters")
    return {"stop_generation": generation, "cameras": cameras}


def sampled_jpeg(config, role, stream):
    started = time.monotonic_ns()
    contents = read_preview_jpeg(config, role)
    received = time.monotonic_ns()
    with Image.open(BytesIO(contents)) as image:
        if image.format != "JPEG" or image.size != (stream["width"], stream["height"]):
            raise ValueError(f"The {role} preview differs from the native JPEG geometry")
        image.load()
    return contents, {"role": role, "file": f"{role}.jpg", "native_jpeg_sha256": sha256(contents).hexdigest(),
                      "requested_ns": str(started), "received_ns": str(received),
                      "conservative_earliest_capture_ns": str(started - MAX_NATIVE_AGE_NS)}


def verify_bracket(before, after):
    if before["stop_generation"] != after["stop_generation"]:
        raise ProtocolError("OFFLINE", "Studio STOP changed during the framing check")
    for role in ROLES:
        first, second = before["cameras"][role], after["cameras"][role]
        if first["stream"] != second["stream"] or second["frames_received"] < first["frames_received"]:
            raise ProtocolError("OFFLINE", f"The native {role} stream restarted during the framing check")


def review_page(captured):
    moment = captured.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    return f"""<!doctype html>
<html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>SO101 camera framing</title>
<style>
body{{font:16px/1.5 system-ui,sans-serif;color:#142635;background:#f2f6f7;margin:32px auto;padding:0 20px;max-width:1180px}}
h1{{font-size:30px;margin:0}} p{{margin:8px 0 20px}}
section{{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,360px),1fr));gap:18px}}
figure{{margin:0;background:#fff;border:1px solid #d5e0e5}} img{{display:block;width:100%;height:auto}}
figcaption{{padding:8px 12px;font-weight:600}} small{{color:#405a67}}
</style>
<h1>SO101 camera framing</h1><p><small>Camera-only preview · {moment}</small></p>
<section><figure><img src="wrist.jpg" alt="Native SO101 wrist camera preview"><figcaption>Wrist camera</figcaption></figure>
<figure><img src="top.jpg" alt="Native SO101 top camera preview"><figcaption>Top camera</figcaption></figure></section>
<p>Check that the top view covers the pickup and receiving areas. During the attended native control check, verify the wrist view at pickup and release.</p>
</html>"""


def capture_framing(config):
    before = snapshot(config)
    captured = datetime.now(timezone.utc)
    frames = [sampled_jpeg(config, role, before["cameras"][role]["stream"]) for role in ROLES]
    after = snapshot(config)
    verify_bracket(before, after)
    root = config["data_root"] / "so101-pilot/framing-previews"
    root.mkdir(mode=0o700, parents=True, exist_ok=True)
    destination = root / f"{captured:%Y%m%dT%H%M%SZ}-{uuid4().hex[:8]}"
    report = {"scope": "setup-only native camera previews", "captured_at": captured.isoformat(),
              "ordered_camera_roles": list(ROLES), "native_before": before, "native_after": after,
              "frames": [record for _, record in frames], "physical_preflight_verified": False,
              "motion_requested": False, "recording_started": False}
    with TemporaryDirectory(prefix="framing-", dir=root) as temporary:
        stage = Path(temporary)
        for contents, record in frames:
            (stage / record["file"]).write_bytes(contents)
        atomic(stage / "receipt.json", report)
        (stage / "framing.html").write_text(review_page(captured), encoding="utf-8")
        stage.rename(destination)
    return {**report, "review_page": str(destination / "framing.html"), "receipt": str(destination / "receipt.json")}
