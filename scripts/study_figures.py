"""Generate printable figures directly from saved research evidence."""

import argparse
import json
from pathlib import Path
from statistics import median

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def paired_supervision(path, output):
    report = json.loads(path.read_text())
    comparisons = (("learned_selective_joint_memory", "Registered method", "#235a78", "o", -.10),
                   ("selective_with_observed_marker_support", "Visible marker check", "#a24a24", "s", .10))
    fig, axes = plt.subplots(1, 2, figsize=(7.8, 3.55), sharey=True, layout="constrained")
    teams = ("3", "4", "8")
    for method, name, color, marker, offset in comparisons:
        for line, size in enumerate(teams):
            estimate = report["methods"][method][size]
            primary = estimate["paired_container_effect"]
            nll = estimate["paired_group_differences"]["group_mean_nll"]
            intervals = ((primary, "conservative_paired_95pct", 100), (nll, "paired_group_bootstrap_95pct", 1))
            for axis, (effect, key, scale) in zip(axes, intervals, strict=True):
                position = line + offset
                edges = [value * scale for value in effect[key]]
                axis.hlines(position, *edges, color=color, linewidth=1.2)
                axis.scatter(effect["difference"] * scale, position, s=28, color=color, marker=marker,
                             label=name if line == 0 else None, zorder=3)
    axes[0].set_title("First correct container choice")
    axes[0].set_xlabel("Difference (percentage points)")
    axes[1].set_title("Group-weighted log loss")
    axes[1].set_xlabel("Difference (nats)")
    axes[0].set_yticks(range(len(teams)), [f"N = {size}" for size in teams])
    axes[0].invert_yaxis()
    axes[0].set_ylim(2.45, -.45)
    for axis in axes:
        axis.axvline(0, color="#454545", linewidth=.8, linestyle="--")
        axis.grid(axis="x", alpha=.17)
        axis.spines[["top", "right", "left"]].set_visible(False)
    axes[0].legend(loc="lower left", bbox_to_anchor=(-.05, -.34), ncol=2, frameon=False, fontsize=8)
    title = "Independent primary test" if report["primary_test_release_sha256"] else "Fixed-profile validation replay"
    fig.suptitle(f"{title} · {report['sources_per_team']} shared source groups\nEvents + reveal minus events only", fontsize=11)
    metadata = {"CreationDate": None, "ModDate": None} if output.suffix == ".pdf" else {}
    fig.savefig(output, dpi=220, metadata=metadata)
    plt.close(fig)


def evidence_figure(path, output):
    report = json.loads(path.read_text())
    xs = [0, 1]
    models = report["models"]
    failure = [next(row for row in model["outcomes"] if row["variant"] == "F2") for model in models]
    visual = [row["visual_placement"]["surface"] for row in failure]
    fused = [row["belief"]["surface"] for row in failure]
    fig, axis = plt.subplots(figsize=(6.3, 3.9), layout="constrained")
    axis.bar([x - .17 for x in xs], visual, .3, color="#326583", label="Visual placement probability")
    axis.bar([x + .17 for x in xs], fused, .3, color="#be6741", label="Fused surface probability")
    axis.set_xticks(xs, ["Events only", "Events + reveal"])
    axis.set_ylabel("Probability of cube on surface")
    axis.set_ylim(0, .8)
    axis.axhline(.75, linestyle="--", linewidth=.8, color="#303030", label="Inspection threshold")
    axis.set_title("Failed placement in one development history")
    axis.legend(loc="upper right", frameon=False, fontsize=8)
    axis.spines[["top", "right"]].set_visible(False)
    fig.savefig(output, dpi=220)
    plt.close(fig)


def learning_curve(path, output):
    title = "ACT five-trajectory diagnostic"
    if path.suffix == ".json":
        report = json.loads(path.read_text())
        points = report["training_loss"]
        if report["input_profile"] == "aligned-rgb-target-clock-v4":
            title = "ACT clock-conditioned diagnostic"
    else:
        from smart_swarm.motor_pilot_run import optimizer_rows

        points = optimizer_rows(path)
        receipt = path.with_name("run.json")
        if receipt.is_file():
            run = json.loads(receipt.read_text())
            if run["recipe"].startswith("smolvla"):
                title = f"SmolVLA CPU training · {run['train_worlds']} simulation sources"
    if len(points) < 2:
        raise ValueError("Training log contains fewer than two optimizer updates")
    fig, axis = plt.subplots(figsize=(6.3, 3.7), layout="constrained")
    axis.plot([row["update"] for row in points], [row["loss"] for row in points],
              color="#326583", linewidth=1.2)
    axis.set_yscale("log")
    axis.set_title(title)
    axis.set_xlabel("Optimizer update")
    axis.set_ylabel("LeRobot training loss")
    axis.spines[["top", "right"]].set_visible(False)
    fig.savefig(output, dpi=220)
    plt.close(fig)


def query_tradeoff(path, output):
    report = json.loads(path.read_text())
    models = report["models"]
    archived = "query_policies" in models[0]
    marks = {"broadcast": ("Broadcast", "o", "#326583"),
             "latest_witness": ("Latest witness", "s", "#727984"),
             "expected_information_gain": ("Expected information gain", "D", "#be6741")}
    fig, axes = plt.subplots(1, len(models), figsize=(8.0, 3.9), sharey=True, squeeze=False, layout="constrained")
    for axis, model in zip(axes[0], models, strict=True):
        policies = model["query_policies"] if archived else model["communication"]["policies"]
        for key, (title, marker, color) in marks.items():
            row = policies[key]
            scores = row["test"]
            rate = scores["resolvable_first_correct_container"] / scores["resolvable_container_count"]
            axis.scatter(row["mean_serialized_bytes"], rate, label=title, marker=marker, c=color, s=48, clip_on=False)
        axis.set_ylim(-.03, 1.03)
        axis.set_xlabel("Mean serialized bytes per decision")
        axis.set_title(model.get("kind", model.get("model")).replace("_", " ").capitalize())
        axis.spines[["top", "right"]].set_visible(False)
    axes[0][0].set_ylabel("First correct container fraction")
    axes[0][-1].legend(loc="upper right", frameon=False, fontsize=8)
    fig.suptitle("Preserved development replay" if archived else "Frozen pilot evaluation")
    fig.savefig(output, dpi=220)
    plt.close(fig)


def evidence_fit(path, output):
    report = json.loads(path.read_text())
    models = report["models"]
    archived = "train_loss" in models[0]
    fig, axes = plt.subplots(1, len(models), figsize=(8.0, 3.65), squeeze=False, layout="constrained")
    for axis, model in zip(axes[0], models, strict=True):
        warmup = model.get("fit_loss", [])
        values = model["train_loss"] if archived else warmup + model["full_loss"]
        indices = range(1, len(values) + 1)
        window = min(21, max(3, len(values) // 5))
        smooth = [median(values[max(0, end - window):end]) for end in indices]
        axis.plot(indices, values, color="#326583", linewidth=.8, alpha=.28)
        axis.plot(indices, smooth, color="#326583", linewidth=1.35, label=f"Trailing median ({window})")
        if warmup:
            axis.axvspan(1, len(warmup), color="#ba9064", alpha=.15, label="Balanced fitting subset")
        if model.get("selected_update") is not None:
            axis.axvline(model["selected_update"], color="#be6741", linestyle="--", linewidth=.85,
                         label="Validation-selected update")
        name = model["model"] if archived else model["kind"]
        axis.set_title(name.replace("_", " ").capitalize())
        axis.set_xlabel("Optimizer update")
        axis.set_ylabel("Training objective")
        axis.legend(frameon=False, fontsize=8)
        axis.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Archived development pilot" if archived else "Pilot evidence-model fitting")
    fig.savefig(output, dpi=220)
    plt.close(fig)


def matched_budget(path, output):
    transport = json.loads(path.read_text())["transport"]
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.8), sharey=True, layout="constrained")
    for axis, horizon in zip(axes, (90, 120), strict=True):
        schedules = (("parallel", "Compact parallel", "#326583", -.19),
                     ("sequential", "Sequential", "#be6741", .19))
        for schedule, name, color, offset in schedules:
            for idx, cap in enumerate((2048, 3072)):
                row = transport[f"{schedule}_{cap}_{horizon}"]
                winners = row["test"]["resolvable_first_correct_container"]
                total = row["test"]["resolvable_container_count"]
                position = idx + offset
                axis.bar(position, winners, width=.32, color=color, label=name if idx == 0 else None)
                axis.text(position, winners + .7, f"{winners}/{total}", color=color, ha="center", fontsize=8)
        axis.set_ylim(0, 25.5)
        axis.set_xticks([0, 1], ["2,048", "3,072"])
        axis.set_xlabel("Serialized byte cap")
        axis.set_title(f"{horizon} ms reply window")
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("First correct container choices")
    axes[-1].legend(frameon=False, loc="upper left", fontsize=8)
    fig.suptitle("Post-freeze replay on 60 previously scored simulation sources")
    fig.savefig(output, dpi=220)
    plt.close(fig)


def scaling_windows(path, output):
    report = json.loads(path.read_text())
    teams = report["teams"]
    heights = [len(team["traces"][0]["camera_samples_by_relocation_witness"]) + 1 for team in teams]
    fig, axes = plt.subplots(len(teams), 1, figsize=(7.1, 6.0), sharex=True, squeeze=False, layout="constrained",
                             gridspec_kw={"height_ratios": heights})
    for axis, team in zip(axes[:, 0], teams, strict=True):
        trace = team["traces"][0]
        budget = trace["observation_budget"]
        before, after = budget["before_relocation"], budget["after_relocation"]
        origin = before[0]["source_time_ns"]
        actors = sorted(trace["camera_samples_by_relocation_witness"])
        for sample, color, label in ((before, "#326583", "Before relocation"), (after, "#be6741", "After relocation")):
            elapsed = [(row["source_time_ns"] - origin) / 1e9 for row in sample]
            actor_index = [actors.index(row["agent_id"]) for row in sample]
            axis.scatter(elapsed, actor_index, s=12, alpha=.9, c=color, label=label)
        axis.set_yticks(range(len(actors)), actors, fontsize=8)
        axis.set_ylim(len(actors) - .5, -.5)
        axis.set_title(f"{team['actor_count']} simulated agents", loc="left", fontsize=10)
        axis.spines[["top", "right", "left"]].set_visible(False)
    axes[0, 0].legend(loc="center", frameon=False, fontsize=9)
    axes[-1, 0].set_xlabel("Simulation time since the first witness sample, seconds")
    fig.suptitle("48 matched camera opportunities in the F1 development scene")
    fig.savefig(output, dpi=220)
    plt.close(fig)


def primary_cost(path, output):
    events = json.loads(path.read_text())
    reveal = json.loads(path.with_name("primary-test-events_plus_reveal.json").read_text())
    if (events["checkpoint_kind"] != "events_only" or reveal["checkpoint_kind"] != "events_plus_reveal"
            or not events["primary_test_release_sha256"]
            or events["primary_test_release_sha256"] != reveal["primary_test_release_sha256"]
            or events["source_groups"] != reveal["source_groups"]):
        raise ValueError("The figure requires two paired independent primary score receipts")
    labels = (("receiver_current_rgb", "Receiver current RGB"), ("receiver_private_temporal", "Receiver private history"),
              ("deterministic_broadcast_ledger", "Deterministic broadcast"),
              ("calibrated_broadcast_joint_particle_filter", "Calibrated particle broadcast"),
              ("central_full_rgb_joint_filter", "Central full RGB"), ("central_compact_budget_matched", "Central compact"),
              ("learned_selective_joint_memory", "Learned selective"))
    first, second = [report["metrics_by_team_size"]["8"] for report in (events, reveal)]
    positions = list(range(len(labels)))
    total = [first[key]["group_mean_bytes"] for key, _ in labels]
    if total != [second[key]["group_mean_bytes"] for key, _ in labels]:
        raise ValueError("The two supervision methods no longer share the communication counts")
    fig, (left, right) = plt.subplots(1, 2, figsize=(8.8, 3.9), sharey=True, layout="constrained")
    left.scatter(total, positions, color="#434343", s=29, zorder=3)
    left.set_xscale("symlog", linthresh=500)
    left.set_xticks([0, 1000, 10_000, 100_000], ["0", "1k", "10k", "100k"])
    left.set_xlabel("Group-mean charged bytes")
    left.set_yticks(positions, [title for _, title in labels])
    left.invert_yaxis()
    for data, legend, color, marker, offset in ((first, "Events only", "#235a78", "o", -.15),
                                                (second, "Events + reveal", "#a24a24", "s", .15)):
        right.scatter([data[key]["group_mean_decision_cost"] for key, _ in labels], [i + offset for i in positions],
                      label=legend, color=color, marker=marker, s=29, zorder=3)
    right.set_xlabel("Group-mean decision cost")
    right.legend(loc="lower center", bbox_to_anchor=(.5, -.34), frameon=False, ncol=2, fontsize=8)
    for axis in (left, right):
        axis.spines[["top", "right", "left"]].set_visible(False)
        axis.grid(axis="y", alpha=.18)
    source_count = len(events["source_groups"])
    fig.suptitle(f"Independent F1–F6 simulation · {source_count} shared source groups · eight agents", fontsize=12)
    meta = {"CreationDate": None, "ModDate": None} if output.suffix == ".pdf" else {}
    fig.savefig(output, dpi=220, metadata=meta)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    figures = {"evidence": evidence_figure, "motor": learning_curve, "queries": query_tradeoff,
               "evidence-fit": evidence_fit, "scaling-windows": scaling_windows, "matched-budget": matched_budget,
               "primary-cost": primary_cost}
    parser.add_argument("kind", choices=figures)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    figures[args.kind](args.input, args.output)


if __name__ == "__main__":
    main()
