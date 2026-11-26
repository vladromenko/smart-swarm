"""Admit Explorer frames by their full mapped capture interval and phase."""

from bisect import bisect_left
from collections import Counter
import math

from .contract import interval
from .episode import POLICY, Recorder

HOME_MARGIN_NS = 25_000_000
MAX_BRACKET_NS = 1_500_000_000
MAX_CLOCK_MS = 50.
MAX_RGB_AGE_S = 1.5
CLOCK_KIND = "midpoint_estimate_not_hardware_sync"


def wall_offset(anchors):
    offsets = [row["unix_ns"] - row["monotonic_ns"] for row in anchors]
    if max(offsets) - min(offsets) > HOME_MARGIN_NS:
        raise ValueError("Homebrain wall-to-monotonic clock mapping changed during capture")
    return offsets[0]


def clock_radius(role, native):
    entry = native.get("robots", {}).get(role, {})
    value = entry.get("uncertainty_ms")
    if (native.get("read_only") is not True or native.get("motion_commands_sent") != 0
            or native.get("running") is not True or entry.get("error")
            or entry.get("clock_kind") != CLOCK_KIND or type(value) not in (int, float)
            or not math.isfinite(value) or not 0 <= value <= MAX_CLOCK_MS):
        raise ValueError("native archived clock is invalid or outside the uncertainty limit")
    return math.ceil(value * 1_000_000)


class FleetClock:
    """Use nearby native status rows as well as the recording's initial boot identity."""

    def __init__(self, role, start, history):
        self.role = role
        self.boot = start.get("robots", {}).get(role, {}).get("boot_id")
        if not isinstance(self.boot, str) or not self.boot:
            raise ValueError(f"{role} lacks the initial native clock boot identity")
        self.initial_radius_ns = clock_radius(role, start)
        self.rows = {}
        for row in history:
            stamp = row.get("updated_at")
            if type(stamp) not in (float, int) or not math.isfinite(stamp) or stamp <= 0:
                raise ValueError("native archived clock status has no valid Homebrain timestamp")
            instant = round(stamp * 1e9)
            if instant in self.rows and self.rows[instant] != row:
                raise ValueError("native archived clock status has contradictory equal-time rows")
            self.rows[instant] = row
        self.times = sorted(self.rows)

    def radius(self, received):
        index = bisect_left(self.times, received)
        if index <= 0 or index >= len(self.times):
            raise ValueError("native frame receipt lacks before-and-after clock status")
        close = self.times[index - 1:index + 1]
        if max(abs(received - instant) for instant in close) > MAX_BRACKET_NS:
            raise ValueError("native frame receipt is too far from its clock status")
        values = [self.initial_radius_ns]
        for instant in close:
            row = self.rows[instant]
            if row.get("robots", {}).get(self.role, {}).get("boot_id") != self.boot:
                raise ValueError("native clock boot changed during the recording")
            values.append(clock_radius(self.role, row))
        return HOME_MARGIN_NS + max(values)


def in_phase(row, clock, offset, bounds):
    nominal, received, age = row["source_unix_ns"], row["received_unix_ns"], row["age_s"]
    if type(nominal) is not int or type(received) is not int or not 0 < nominal <= received:
        raise ValueError("native RGB is missing an ordered source and receipt timestamp")
    if type(age) not in (float, int) or not 0 <= age <= MAX_RGB_AGE_S:
        raise ValueError("native RGB exceeded the archive freshness limit")
    uncertainty = clock.radius(received)
    sampled, arrived = nominal - offset, received - offset
    begin, end = sampled - uncertainty, min(sampled + uncertainty, arrived)
    lower, upper = bounds
    if not lower <= begin <= end <= arrived <= upper:
        raise ValueError("full RGB interval and receipt must fit the permitted phase")
    return {**row, "center_ns": sampled, "received_ns": arrived, "exposure_bounds_ns": (begin, end),
            "mapping_radius_ns": uncertainty}


def eligible_frames(candidates, clock, offset, bounds):
    admitted, rejected = {}, Counter()
    for row in sorted(candidates, key=lambda item: (item.get("received_unix_ns") or 0, item["after_s"])):
        try:
            verified = in_phase(row, clock, offset, bounds)
            source = verified["center_ns"]
            if source in admitted:
                raise ValueError("repeated native camera source timestamp")
        except ValueError as error:
            rejected[str(error)] += 1
        else:
            admitted[source] = verified
    audit = {"candidates": len(candidates), "eligible": len(admitted), "clock_status_samples": len(clock.times),
             "excluded_reasons": dict(rejected)}
    return list(admitted.values()), audit


class PhysicalStoryRecorder(Recorder):
    """Use the shared packet writer with a conservative interval on fleet RGB."""

    fleet_exposure = None

    def fleet_packet(self, agent, sensor, image, timing):
        self.fleet_exposure = (sensor, timing["exposure_bounds_ns"])
        try:
            return self.capture(agent, {sensor: image}, timing["received_ns"], [], [],
                                exposures={sensor: timing["center_ns"]})
        finally:
            self.fleet_exposure = None

    def frame(self, namespace, sensor, sequence, image, timestamp):
        result = super().frame(namespace, sensor, sequence, image, timestamp)
        scope = self.fleet_exposure
        if namespace == POLICY and scope is not None and sensor == scope[0]:
            result["capture_interval"] = interval(*scope[1], self.clock)
        return result
