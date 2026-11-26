"""Use the existing camera-only Studio preview for a timed evaluator reveal clip."""

from hashlib import sha256
from io import BytesIO
import re
import time
from urllib.request import ProxyHandler, Request, build_opener

import numpy as np
from PIL import Image

from .contract import ProtocolError, interval, read_ns, time_at
from .episode import TRUTH, atomic, audited_frame
from .platform_client import NoRedirect
from .studio import STOP_UNCONFIRMED, native_request

PROFILE = "sampled-native-top-preview-v1"
FIELD = "evaluator_reveal_profile"
SAMPLE_COUNT = 3
MAX_NATIVE_AGE_NS = 1_000_000_000
MAX_FRAME_BYTES = 4 * 1024**2
STREAM_FIELDS = ("device", "device_resolved", "width", "height", "timestamp_basis")


def preview_summary(status):
    preview = status["preview"]
    top = preview.get("cameras", {}).get("top", {})
    age = top.get("last_frame_age_ms")
    sequence = top.get("frames_received")
    controls = (status["recording"], status["teleoperation"])
    if (any(row.get("active") or row.get("hardware_open") or row.get("hardware_stop_state") == STOP_UNCONFIRMED
            for row in controls)
            or preview.get("active") is not True or preview.get("camera_only") is not True
            or preview.get("torque_enabled") is not False or top.get("connected") is not True
            or type(age) not in (int, float) or not 0 <= age < 500 or type(sequence) is not int or sequence < 1):
        raise ProtocolError("OFFLINE", "Start or check the native camera-only top preview before recording the reveal")
    return {"top_frame_sequence": sequence, "status_frame_age_ms": age}


def preview_snapshot(config):
    status = native_request(config, "/api/status")
    sample = preview_summary(status)
    stream = status["preview"]["cameras"]["top"]["stream"]
    sample.update(stream={key: stream[key] for key in STREAM_FIELDS},
                  native_stop_generation=status["hardware_stop_generation"])
    if not valid_snapshot(sample):
        raise ValueError("The native preview lacks a stable device, shape or STOP-generation readback")
    return sample


def valid_snapshot(snapshot):
    stream, generation = snapshot["stream"], snapshot["native_stop_generation"]
    sequence, age = snapshot["top_frame_sequence"], snapshot["status_frame_age_ms"]
    return (set(stream) == set(STREAM_FIELDS)
            and all(isinstance(stream[k], str) and stream[k] for k in ("device", "device_resolved", "timestamp_basis"))
            and all(type(stream[k]) is int and stream[k] > 0 for k in ("width", "height"))
            and type(generation) is int and generation >= 0 and type(sequence) is int and sequence > 0
            and type(age) in (int, float) and 0 <= age < 500)


def read_preview(config):
    if config["studio_origin"] != "http://127.0.0.1:8020":
        raise ValueError("Expected the native Homebrain preview on loopback")
    request = Request(config["studio_origin"] + "/api/frame/top")
    opener = build_opener(ProxyHandler({}), NoRedirect())
    with opener.open(request, timeout=5) as response:
        contents = response.read(MAX_FRAME_BYTES + 1)
        if response.headers.get_content_type() != "image/jpeg" or not 0 < len(contents) <= MAX_FRAME_BYTES:
            raise ValueError("Studio returned an invalid bounded JPEG preview")
    with Image.open(BytesIO(contents)) as image:
        if image.format != "JPEG":
            raise ValueError("Studio returned a preview with an invalid image format")
        picture = np.asarray(image.convert("RGB"))
    return picture, sha256(contents).hexdigest()


def same_stream(first, second):
    return all(first[key] == second[key] for key in ("stream", "native_stop_generation"))


def read_sample(config, previous):
    # Studio rejects JPEGs older than one second. Waiting past that bound
    # rules out any exposure within the previous sample's conservative interval.
    delay = previous + MAX_NATIVE_AGE_NS + 1 - time.monotonic_ns()
    if delay > 0:
        time.sleep(delay / 1e9)
    before = preview_snapshot(config)
    started = time.monotonic_ns()
    image, source_digest = read_preview(config)
    finished = time.monotonic_ns()
    after = preview_snapshot(config)
    proof = {"request_started_ns": str(started), "received_ns": str(finished),
             "earliest_possible_capture_ns": str(started - MAX_NATIVE_AGE_NS),
             "preview_status_before": before, "preview_status_after": after, "native_jpeg_sha256": source_digest}
    expected = before["stream"]["height"], before["stream"]["width"], 3
    if image.shape != expected:
        raise ValueError("The reveal RGB shape differs from the native camera readback")
    return image, proof


def validate_sample(proof, previous_proof, decision):
    low, high, started = [read_ns(proof[name]) for name in ("earliest_possible_capture_ns", "received_ns", "request_started_ns")]
    before, after = proof["preview_status_before"], proof["preview_status_after"]
    if not (decision < low == started - MAX_NATIVE_AGE_NS < started <= high
            and valid_snapshot(before) and valid_snapshot(after) and same_stream(before, after)
            and before["top_frame_sequence"] <= after["top_frame_sequence"]
            and re.fullmatch(r"[a-f0-9]{64}", proof["native_jpeg_sha256"])):
        raise ValueError("The native preview cannot prove a frame after the saved decision on the same stream")
    if previous_proof is not None:
        prior = previous_proof["preview_status_after"]
        if (low <= int(previous_proof["received_ns"]) or not same_stream(prior, before)
                or prior["top_frame_sequence"] > before["top_frame_sequence"]
                or prior["top_frame_sequence"] >= after["top_frame_sequence"]):
            raise ValueError("The reveal samples overlap or the preview stream changed")
    return high


def capture_clip(config, decision):
    if not 0 <= decision <= time.monotonic_ns():
        raise ValueError("The saved decision is ahead of the current native clock")
    previous, proof = decision, None
    result = []
    for _ in range(SAMPLE_COUNT):
        image, sample = read_sample(config, previous)
        previous = validate_sample(sample, proof, decision)
        result.append((image, sample))
        proof = sample
    return result


def store_clip(recorder, samples, decision, outcome):
    if len(samples) != SAMPLE_COUNT:
        raise ValueError("The evaluator preview capture is incomplete")
    frames = []
    for image, proof in samples:
        stamp = int(proof["received_ns"])
        camera = recorder.frame(TRUTH, "evaluator/top", stamp, image, stamp)
        frames.append({"camera": camera, "native_preview_receipt": proof})
    last = frames[-1]
    stamp = int(last["native_preview_receipt"]["received_ns"])
    row = {"camera": last["camera"], "ground_truth": outcome, "relations_preserved_until_reveal": True,
           "target_state_time_or_interval": interval(decision, stamp, recorder.clock), "observed_at": time_at(stamp, recorder.clock),
           "preview_clock_scope": last["native_preview_receipt"], "reveal_clip": {"profile": PROFILE, "frames": frames}}
    validate_clip(recorder.root, row, decision, stamp, recorder.clock, required=True)
    atomic(recorder.root / TRUTH / "reveal.jsonl", row)
    return stamp


def validate_clip(root, row, decision, observed, clock, *, required=False):
    clip = row.get("reveal_clip")
    if clip is None and not required:
        return None
    if not clip or clip.get("profile") != PROFILE or len(clip["frames"]) != SAMPLE_COUNT:
        raise ValueError("The physical evaluator reveal lacks its complete three-sample clip")
    if (row["observed_at"] != time_at(observed, clock)
            or row["relations_preserved_until_reveal"] is not True
            or row["target_state_time_or_interval"] != interval(decision, observed, clock)):
        raise ValueError("The evaluator reveal does not cover the saved decision and receipt")
    previous = None
    identities = set()
    for frame in clip["frames"]:
        proof, camera = frame["native_preview_receipt"], frame["camera"]
        stamp = validate_sample(proof, previous, decision)
        stream = proof["preview_status_before"]["stream"]
        if (camera["sensor_id"] != "evaluator/top" or camera["access_role"] != "evaluator_only"
                or camera["exposure_time"] is not None or camera["capture_interval"] != interval(stamp, stamp, clock)
                or [camera["width"], camera["height"]] != [stream["width"], stream["height"]]):
            raise ValueError("A clip frame disagrees with its evaluator access, ordering or native camera readback")
        audited_frame(root, camera, TRUTH)
        identities.add(camera["frame_id"])
        previous = proof
    last = clip["frames"][-1]
    if (len(identities) != SAMPLE_COUNT or stamp != observed or last["camera"] != row["camera"]
            or last["native_preview_receipt"] != row["preview_clock_scope"]):
        raise ValueError("The terminal reveal does not point to the last verified clip frame")
    return SAMPLE_COUNT


def validate_sealed_clip(root, row, manifest):
    decision, reveal = manifest["decision_boundary"], manifest["reveal_boundary"]
    profile = manifest.get(FIELD)
    if profile is not None and profile != PROFILE:
        raise ValueError("Unsupported physical evaluator reveal profile")
    return validate_clip(root, row, read_ns(decision["monotonic_ns"]), read_ns(reveal["monotonic_ns"]),
                         decision["clock_id"], required=profile is not None)
