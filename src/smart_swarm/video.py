"""Compose a trusted playback film from recorded views, with the reveal last."""

from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from .episode import POLICY, TRUTH, load, rows

SENSORS = ("so101-1/top", "so101-1/wrist", "tank-1/front", "rosmaster-1/front")
TITLES = ("SO101 / station", "SO101 / wrist", "Tank / exterior view", "ROSMASTER / query view")


def timeline(root):
    updates = defaultdict(dict)
    meta = load(root / "manifest.json")
    for agent in meta["episodes"]:
        for packet in rows(root / POLICY / f"{agent}.jsonl"):
            stamp = int(packet["source_time"]["monotonic_ns"])
            updates[stamp].update({row["sensor_id"]: row for row in packet["cameras"]})
    decision = rows(root / "decisions" / "prediction.jsonl")[0]
    reveal = rows(root / TRUTH / "reveal.jsonl")[0]
    end = int(reveal["observed_at"]["monotonic_ns"])
    for stamp in range(0, end, 333_333_333):
        updates[stamp].update({})
    updates[end]["reveal"] = reveal["camera"]
    return sorted(updates.items()), decision, reveal


def panel(root, images, stamp, decision, reveal):
    output = Image.new("RGB", (640, 572), "#142031")
    draw = ImageDraw.Draw(output)
    draw.text((10, 7), f"Smart Swarm   /   MuJoCo replay   /   {stamp / 1e9:05.2f} s", fill="white")
    for index, (sensor, title) in enumerate(zip(SENSORS, TITLES, strict=True)):
        x, y = 320 * (index % 2), 27 + 261 * (index // 2)
        draw.text((x + 8, y), title, fill="#d3e4ee")
        row = images.get(sensor)
        if row:
            with Image.open(root / row["frame_ref"]) as image:
                output.paste(image, (x, y + 18))
        else:
            draw.text((x+87, y+134), "No frames received", fill="#9bb0c6")
    footer = "Acquisition windows differ by agent. Last recorded frame is held in each panel."
    if stamp >= int(decision["at"]["monotonic_ns"]):
        name = decision["query_all"]["decision"].replace("_", " ")
        footer = f"ROSMASTER query decision   /   {name}"
    if "reveal" in images:
        with Image.open(root / images["reveal"]["frame_ref"]) as image:
            output.paste(image, (320, 306))
        draw.rectangle((320, 288, 640, 305), fill="#142031")
        draw.text((328, 288), "Evaluator / post-decision reveal", fill="#eef59f")
        footer += f"   /   withheld ground truth: {reveal['ground_truth']}"
    draw.text((10, 551), footer, fill="white")
    return np.asarray(output)


def encode_replay(root: Path, destination: Path):
    import av

    events, decision, reveal = timeline(root)
    with av.open(str(destination), "w", options={"movflags": "+faststart"}) as file:
        stream = file.add_stream("libx264", rate=6)
        stream.width, stream.height, stream.pix_fmt = 640, 572, "yuv420p"
        stream.options = {"crf": "23", "preset": "ultrafast"}
        visible = {}
        index = 0
        for stamp in playback_clock(decision, reveal):
            while index < len(events) and events[index][0] <= stamp:
                visible.update(events[index][1])
                index += 1
            view = panel(root, visible, stamp, decision, reveal)
            frame = av.VideoFrame.from_ndarray(view, format="rgb24")
            for packet in stream.encode(frame):
                file.mux(packet)
        for packet in stream.encode():
            file.mux(packet)

def playback_clock(decision, reveal):
    decision_ns = int(decision["at"]["monotonic_ns"])
    reveal_ns = int(reveal["observed_at"]["monotonic_ns"])
    ticks = [round(index * 1e9 / 6) for index in range(round(decision_ns * 6 / 1e9) + 1)]
    return [stamp for stamp in ticks if stamp < decision_ns] + [decision_ns] * 9 + [reveal_ns] * 22
