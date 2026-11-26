#!/usr/bin/env python3
"""Plot observed wrist frames and the frozen ACT pixel-swap response."""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from smart_swarm.episode import load, sha

CUBE = "cube_without_jaw_overlap"
JAW = "jaw_without_cube_overlap"
OCCLUSION = "cube_and_jaw_overlap"
REMAINDER = "remaining_pixels"
CONDITIONS = (
    ("Original early wrist frame", ()),
    ("Late jaw and remaining pixels", (JAW, OCCLUSION, REMAINDER)),
    ("Late cube pixels, away from jaw", (CUBE,)),
    ("Original late wrist frame", (CUBE, JAW, OCCLUSION, REMAINDER)),
)


def original_frames(report, fit):
    source = min((row for row in report["sources"] if row["split"] == "validation"), key=lambda row: row["seed"])
    lineage = load(Path(fit["dataset"]) / "validation" / "smart-swarm-lineage.json")
    entry = next(row for row in lineage["episodes"] if row["source_sha256"] == source["source_sha256"])
    path = Path(entry["raw_path"]) / "motor.npz"
    if source["source_sha256"] != sha(path):
        raise ValueError("The original validation source changed")
    issues = [report["held_fixed_issue"], report["late_pixel_source_issue"]]
    with np.load(path, allow_pickle=False) as archive:
        return source["seed"], issues, archive["wrist"][issues]


def responses(axis, sources):
    axis.axvspan(0, 25, color="#e1efe8", label="Below 25%")
    ordered = sorted(sources, key=lambda row: (row["split"], row["seed"]))
    offsets = np.linspace(-.19, .19, len(ordered))
    labeled = set()
    for row_index, (_, regions) in enumerate(CONDITIONS):
        for offset, source in zip(offsets, ordered, strict=True):
            point = next(row for row in source["region_swaps"] if set(row["later_regions"]) == set(regions))
            split = source["split"]
            fit = split == "train"
            axis.scatter(point["first_gripper_target_percent"], row_index + offset,
                         marker="o" if fit else "^", color="#174a72" if fit else "#99491f", s=29, zorder=3,
                         label=split.capitalize() if split not in labeled else None)
            labeled.add(split)
    axis.set_yticks(range(len(CONDITIONS)), [label for label, _ in CONDITIONS])
    axis.set_ylim(3.55, -.55)
    axis.set_xlim(0, 60)
    axis.set_xlabel("First requested gripper opening (% of calibrated range)")
    axis.set_axisbelow(True)
    axis.grid(axis="x", color="#cdd4db", linewidth=.7)
    axis.legend(loc="lower center", bbox_to_anchor=(.5, 1.02), ncol=3, frameon=False, fontsize=9)


def draw(report, fit, destination):
    seed, issues, frames = original_frames(report, fit)
    style = {"font.family": "DejaVu Sans", "axes.spines.top": False, "axes.spines.right": False}
    with plt.rc_context(style):
        fig = plt.figure(figsize=(11.5, 5.2), layout="constrained")
        grid = fig.add_gridspec(2, 2, width_ratios=[1, 3.6])
        for row, (issue, frame) in enumerate(zip(issues, frames, strict=True)):
            image = fig.add_subplot(grid[row, 0])
            image.imshow(frame)
            image.set_title(f"Recorded issue {issue}", fontsize=11, loc="left")
            image.axis("off")
        axis = fig.add_subplot(grid[:, 1])
        responses(axis, report["sources"])
        fig.suptitle("Changing the cube region in wrist RGB switches the frozen ACT forecast", x=.015, ha="left", fontsize=15)
        fig.supxlabel(f"Original wrist frames from validation seed {seed}. Overhead RGB, joints and clock remain at issue {issues[0]} in the replay.",
                      x=.015, ha="left", fontsize=9)
        fig.savefig(destination, metadata={"CreationDate": None, "ModDate": None})
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--fit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    draw(load(args.audit), load(args.fit), args.output)


if __name__ == "__main__":
    main()
