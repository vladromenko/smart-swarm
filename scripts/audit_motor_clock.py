"""Check the motor clock transform against recorded and deployment input paths."""

import argparse
from base64 import b64decode
from hashlib import sha256
from pathlib import Path

import numpy as np

from export_lerobot import KEYS, load_episode, make_frame
from smart_swarm.arm_executor import ArmExecutor
from smart_swarm.episode import atomic, load
from smart_swarm.evaluate_motor import action_request
from smart_swarm.motor import SENSORS
from smart_swarm.motor_input import CLOCKED_PROFILE, ISSUE_CLOCK_BITS, PROFILE
from smart_swarm.physics import STEP


class RecordedWorld:
    def __init__(self, data):
        self.data = data
        self.ns = 100_000_000
        self.index = -1

    def step(self, count):
        self.ns += round(count * STEP * 1e9)
        self.index += 1

    def views(self, agent):
        if agent != "so101-1":
            raise ValueError("Unexpected camera grant")
        return {key: self.data[name][self.index] for key, name in zip(SENSORS, KEYS, strict=True)}

    def state(self, agent):
        if agent != "so101-1":
            raise ValueError("Unexpected state grant")
        return [self.data["state"][self.index].tolist()]


def payload_matches(world, data, meta, index):
    marker = meta.get("target_marker", "cyan")
    infer = action_request(world, CLOCKED_PROFILE, marker, issue_index=index)
    expected = make_frame(data, meta, index, CLOCKED_PROFILE)
    if infer["task"] != expected["task"] or not np.array_equal(infer["state"], expected["observation.state"]):
        raise ValueError("Source and inference disagree on target or proprioception")
    if any(b64decode(value) != expected[key].tobytes() for value, key in zip(infer["images"], KEYS.values(), strict=True)):
        raise ValueError("Source and inference disagree on camera bytes")
    return expected[KEYS["top"]][4:9, 18:58].tobytes()


def audit_source(raw, row):
    meta = load(raw / "manifest.json")
    if meta["sha256"] != row["source_sha256"]:
        raise ValueError("Preserved source manifest disagrees with fitting lineage")
    data = load_episode(raw, meta, aligned=True)
    with np.load(raw / "motor.npz", allow_pickle=False) as archive:
        source_times = archive["timestamps_ns"]
    world = RecordedWorld(data)
    executor = ArmExecutor(world, "aligned-sample-v2")
    observed_codes = set()
    for index, timestamp in enumerate(source_times.tolist()):
        executor.advance()
        if executor.observation_index() != index or world.ns != timestamp:
            raise ValueError("Deployment issue clock disagrees with the verified source boundary")
        code = payload_matches(world, data, meta, executor.observation_index())
        observed_codes.add(sha256(code).hexdigest())
    if len(observed_codes) != len(source_times):
        raise ValueError("The local issue cue does not separate every source observation")
    legacy, updated = dwell_pixels(data, meta)
    return {"source": str(raw), "source_sha256": row["source_sha256"], "frames": len(source_times),
            "verified_camera_state_and_issue_time": True, "unique_issue_codes": len(observed_codes),
            "historical_hold_top_delta": legacy, "clocked_hold_top_delta": updated}


def dwell_pixels(data, meta):
    def diff(profile):
        a, b = [make_frame(data, meta, index, profile)[KEYS["top"]].astype(np.int16) for index in (41, 58)]
        return {"changed_rgb_values": int(np.count_nonzero(a - b)), "maximum_pixel_delta": int(np.max(np.abs(a - b)))}

    return diff(PROFILE), diff(CLOCKED_PROFILE)


def audit(fit, output):
    result = {}
    for split in ("train", "validation"):
        rows = load(fit / split / "smart-swarm-lineage.json")["episodes"]
        result[split] = [audit_source(Path(row["raw_path"]), row) for row in rows]
    report = {"input_profile": CLOCKED_PROFILE, "parent_profile": PROFILE, "clock_bits": ISSUE_CLOCK_BITS,
              "timing_source": "local simulated control issues elapsed since reset, available at inference",
              "scope": "bit-exact exporter and request parity on archived frames, no optimizer or rendering",
              "source_dataset": str(fit), "source_counts": {key: len(rows) for key, rows in result.items()}, "checks": result}
    atomic(output, report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fit_dataset", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    audit(args.fit_dataset, args.output)


if __name__ == "__main__":
    main()
