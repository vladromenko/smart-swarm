"""Read RGB receipt headers in the installed Explorer recorder's uncompressed MCAP files."""

import os
import re
from struct import Struct

START_GATE = "native-mcap-rgb-receipts-v1"
START_RECEIPT = "fleet-rgb-ready.json"
REQUIRED = {"tank": 8, "rosmaster": 1}
TOPICS = {f"/explorer3/robots/{role}/rgb".encode(): role for role in REQUIRED}
MAGIC = b"\x89MCAP0\r\n"
RECORD = Struct("<BQ")
CHANNEL_PREFIX = Struct("<HHI")
MESSAGE_PREFIX = Struct("<HIQQ")
CHUNK_PREFIX = Struct("<QQQIIQ")
CHANNEL, MESSAGE, CHUNK, DATA_END = 4, 5, 6, 15


def complete_records(stream, limit):
    while stream.tell() + RECORD.size <= limit:
        header = stream.read(RECORD.size)
        if len(header) != RECORD.size:
            return
        opcode, size = RECORD.unpack(header)
        end = stream.tell() + size
        if end > limit:
            return
        yield opcode, end
        stream.seek(end)


def observe_record(stream, opcode, end, channels, times):
    available = end - stream.tell()
    if opcode == CHANNEL and available >= CHANNEL_PREFIX.size:
        identity, _, topic_size = CHANNEL_PREFIX.unpack(stream.read(CHANNEL_PREFIX.size))
        if topic_size <= min(256, end - stream.tell()):
            role = TOPICS.get(stream.read(topic_size))
            if role is not None:
                channels[identity] = role
    if opcode == MESSAGE and available >= MESSAGE_PREFIX.size:
        identity, _, logged, _ = MESSAGE_PREFIX.unpack(stream.read(MESSAGE_PREFIX.size))
        if identity in channels:
            times[channels[identity]].add(logged)


def uncompressed_records_end(stream, end):
    if end - stream.tell() < CHUNK_PREFIX.size:
        raise ValueError("An Explorer MCAP chunk lacks its complete header")
    _, _, raw_size, _, compression_length, stored_size = CHUNK_PREFIX.unpack(stream.read(CHUNK_PREFIX.size))
    if compression_length or raw_size != stored_size or stream.tell() + stored_size > end:
        raise ValueError("This Explorer MCAP capture does not use the installed uncompressed profile")
    return stream.tell() + stored_size


def bag_rgb_receipts(path, times):
    channels = {}
    with path.open("rb") as stream:
        limit = os.fstat(stream.fileno()).st_size
        if limit < len(MAGIC):
            return
        if stream.read(len(MAGIC)) != MAGIC:
            raise ValueError("Explorer observation file lacks its native MCAP signature")
        for opcode, end in complete_records(stream, limit):
            if opcode == DATA_END:
                return
            if opcode == CHUNK:
                records_end = uncompressed_records_end(stream, end)
                for nested, nested_end in complete_records(stream, records_end):
                    observe_record(stream, nested, nested_end, channels, times)
            else:
                observe_record(stream, opcode, end, channels, times)


def archive_rgb_receipts(archive):
    """Return log times only. The official archive reader still validates observations for replay."""
    times = {key: set() for key in REQUIRED}
    root = archive / "bag"
    bags = [(int(path.stem.removeprefix("bag_")), path) for path in root.glob("bag_*.mcap")
            if re.fullmatch(r"bag_[0-9]+", path.stem) and path.resolve().parent == root.resolve()]
    for _, path in sorted(bags):
        bag_rgb_receipts(path, times)
    return times


def first_recorded_rgb(archive, acknowledged_ns):
    counts = {}
    for role, stamps in archive_rgb_receipts(archive).items():
        values = sorted(value for value in stamps if value >= acknowledged_ns)
        if len(values) >= REQUIRED[role]:
            counts[role] = {"count": len(values), "first_received_ns": values[0],
                            "required_received_ns": values[REQUIRED[role] - 1]}
    return counts if counts.keys() == REQUIRED.keys() else None


def valid_start_receipt(record, capture_id, acknowledged):
    if (record.get("capture_id") != capture_id or record.get("gate") != START_GATE
            or record.get("acknowledged") != acknowledged or record.get("seen_monotonic_ns", -1) < acknowledged["monotonic_ns"]):
        return False
    roles = record.get("recorded_rgb", {})
    return roles.keys() == REQUIRED.keys() and all(
        type(roles[role].get("count")) is int and roles[role]["count"] >= minimum
        and type(roles[role].get("first_received_ns")) is int and type(roles[role].get("required_received_ns")) is int
        and acknowledged["unix_ns"] <= roles[role]["first_received_ns"] <= roles[role]["required_received_ns"]
        <= record.get("seen_unix_ns", -1) for role, minimum in REQUIRED.items())
