#!/usr/bin/env python3
"""Plot the measured SO101 target support and matched failure replay."""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from smart_swarm.episode import load

FAMILIES = {"F1": ("F1  Placement", "#386CB0"), "F2": ("F2  Failure", "#EF8A32"),
            "F3": ("F3  Removal", "#359864"), "F5": ("F5  Nesting", "#8773B2")}
MODELS = {"events_only": "Event-only",
          "events_plus_reveal": "Event and reveal"}


def pixel_support(axis, report):
    observations = report["pilot_inputs"]["observed"]
    for family, (name, color) in FAMILIES.items():
        points = [row["last_cube_xy_pixels"] for row in observations
                  if row["split"] == "train" and row["family"] == family]
        x, y = zip(*(point for point in points if point is not None), strict=True)
        axis.scatter(x, y, label=name, s=24, color=color, alpha=.55, edgecolors="none")
    failed = next(row for row in report["diagnostic"]["events_only"]["rows"]
                  if row["group"] == "station_70_degrees" and row["truth"] == "surface")
    x, y = failed["support"]["last_cube_xy_pixels"]
    axis.scatter(x, y, s=62, marker="D", color="#C83840", zorder=5)
    axis.annotate("Matched failure", (x, y), xytext=(14, 2), textcoords="offset points", color="#921E2D",
                  arrowprops={"arrowstyle": "-", "color": "#921E2D"})
    axis.set(xlim=(0, 159), ylim=(119, 0), xlabel="Horizontal pixel", ylabel="Vertical pixel",
             title="Target location in the last SO101 frame")
    axis.legend(loc="upper right", frameon=False, handletextpad=.1)
    axis.grid(alpha=.22)


def wrong_answer_support(axis, report):
    for index, (kind, model) in enumerate(MODELS.items()):
        failure = next(row for row in report["diagnostic"][kind]["rows"]
                       if row["group"] == "station_70_degrees" and row["truth"] == "surface")
        for offset, key, label, color in ((-.18, "fixed", "Frozen learner", "#C83840"),
                                         (.18, "visibility_and_conflict", "Agreement check", "#567389")):
            row = failure["outcomes"][key]
            mass = row["belief"]["right"]
            axis.bar(index + offset, mass, .32, color=color, label=label if index == 0 else None)
            decision = "Right" if row["decision"] == "right" else "Inspect"
            axis.text(index + offset, mass + .04, f"{mass:.2f}\n{decision}", ha="center", va="bottom",
                      color="#273543", fontsize=9)
    axis.set(xticks=range(2), xticklabels=MODELS.values(), ylim=(0, 1.18),
             ylabel="Probability of the right container", title="Prediction on the matched failure")
    axis.legend(loc="upper center", bbox_to_anchor=(.5, -.11), ncol=2, frameon=False)
    axis.grid(axis="y", alpha=.22)
    axis.set_axisbelow(True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Choose a new figure destination to preserve the published plot")
    report = load(args.report)
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    figure, axes = plt.subplots(1, 2, figsize=(9.4, 4.0), layout="constrained")
    pixel_support(axes[0], report)
    wrong_answer_support(axes[1], report)
    figure.savefig(args.output)
    plt.close(figure)


if __name__ == "__main__":
    main()
