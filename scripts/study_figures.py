"""Generate printable figures directly from saved research evidence."""

import argparse
import json
from pathlib import Path
from statistics import median

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PRIMARY_METHOD_LABELS = (("receiver_current_rgb", "Receiver current RGB"),
                         ("receiver_private_temporal", "Receiver private history"),
                         ("deterministic_broadcast_ledger", "Deterministic broadcast"),
                         ("calibrated_broadcast_joint_particle_filter", "Calibrated particle broadcast"),
                         ("central_full_rgb_joint_filter", "Central full RGB"),
                         ("central_compact_budget_matched", "Central compact"),
                         ("learned_selective_joint_memory", "Learned selective"))


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
    labels = PRIMARY_METHOD_LABELS
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


def inspection_points(path):
    audit = json.loads(path.read_text())
    if (audit["schema"] != "smart-swarm-frozen-f6-audit-v1" or audit["scope"] != "independent primary test"
            or not audit["full_saved_prediction_payload_identical_per_pair"]
            or not audit["evaluator_and_training_sidecars_substituted"] or not audit["primary_test_release_sha256"]):
        raise ValueError("The chart requires the sealed independent F6 audit")
    methods = [key for key, _ in PRIMARY_METHOD_LABELS]
    if methods != audit["qualified_policy_methods"]:
        raise ValueError("The independent audit differs from the declared comparator set")
    pairs = audit["independent_source_groups"]
    checks = audit["team_checks"]["8"]
    n_controls = checks["events_only"]["resolvable_controls"]["independent_groups"]
    if pairs != len(audit["groups"]) or pairs < 1 or n_controls < 1:
        raise ValueError("The audit has incomplete independent source coverage")
    points = {}
    for kind in ("events_only", "events_plus_reveal"):
        row = checks[kind]
        decisions = row["f6_pair_decisions_by_method"]
        counts = row["resolvable_controls"]
        if (counts["independent_groups"] != n_controls or counts["always_inspect_correct"] != 0
                or any(sum(decisions[key].values()) != pairs for key in methods)):
            raise ValueError("The audit has inconsistent F6 or resolving-control counts")
        controls = [counts["correct_by_method"][key] for key in methods]
        if any(not 0 <= count <= n_controls for count in controls):
            raise ValueError("A method has an invalid resolving-control count")
        points[kind] = ([decisions[key].get("request_inspection", 0) for key in methods], controls)
    return pairs, n_controls, points


def primary_inspection(path, output):
    pairs, controls, points = inspection_points(path)
    titles = [label for _, label in PRIMARY_METHOD_LABELS] + ["Always inspect"]
    positions = range(len(titles) - 1)
    fig, axes = plt.subplots(1, 2, figsize=(7.8, 3.85), sharey=True, layout="constrained")
    variants = (("events_only", "Events only", "#235a78", "o", -.14),
                ("events_plus_reveal", "Events + reveal", "#a24a24", "s", .14))
    for kind, label, color, marker, offset in variants:
        for axis, counts in zip(axes, points[kind], strict=True):
            axis.scatter(counts, [index + offset for index in positions], color=color, marker=marker,
                         label=label, s=24, zorder=3)
    for axis, count, baseline, title in zip(axes, (pairs, controls), (pairs, 0),
                                           ("F6 pairs sent to inspection", "Resolving controls answered correctly"),
                                           strict=True):
        axis.scatter(baseline, len(positions), color="#4d4d4d", marker="D", s=27, zorder=3)
        axis.set_title(title, fontsize=9)
        axis.set_xlim(-.055 * count, 1.055 * count)
        axis.set_xticks([round(count * part / 4) for part in range(5)])
        axis.tick_params(axis="y", length=0)
        axis.grid(axis="x", alpha=.2)
        axis.spines[["top", "right", "left"]].set_visible(False)
    axes[0].set_yticks(range(len(titles)), titles, fontsize=8)
    axes[0].set_ylim(len(titles) - .45, -.55)
    axes[1].legend(loc="lower center", bbox_to_anchor=(.5, -.28), frameon=False, ncol=2, fontsize=8)
    fig.suptitle("Pre-inspection decisions in the independent eight-agent simulation", fontsize=11)
    meta = {"CreationDate": None, "ModDate": None} if output.suffix == ".pdf" else {}
    fig.savefig(output, dpi=220, metadata=meta)
    plt.close(fig)


def smolvla_teacher_response(path, output):
    report = json.loads(path.read_text())
    cases = report["teacher_windows"]
    if (report["inputs_only"] or report["optimizer_updates"] != 0 or len(cases) != 2
            or {case["marker"] for case in cases} != {"cyan", "violet"}
            or not all(case["matched_original_initial_chunk"] for case in cases)):
        raise ValueError("The diagnostic figure requires both matching frozen startup traces")
    fig, axes = plt.subplots(1, 2, figsize=(7.85, 3.2), sharey=True, layout="constrained")
    series = (("teacher_action", "Teacher target", "#333333", "s", 26, False),
              ("policy_with_feedback", "Recorded closed loop", "#a24a24", "o", 75, True),
              ("policy_with_teacher_input", "Policy on teacher RGB and state", "#235a78", "o", 22, False))
    for axis, case in zip(axes, cases, strict=True):
        windows = case["chunks"]
        rows = [sample for chunk in windows for sample in chunk["sampled_actions"]]
        if [chunk["refresh_issue"] for chunk in windows] != [0, 50] or [row["issue"] for row in rows] != [14, 29, 44, 59, 74, 89]:
            raise ValueError("The samples do not align to the two declared chunk refreshes")
        for field, name, color, marker, size, hollow in series:
            axis.scatter([row["issue"] for row in rows], [row[field][0] for row in rows],
                         label=name, s=size, marker=marker, edgecolor=color, facecolor="none" if hollow else color,
                         linewidth=.9, zorder=3)
        axis.axhline(0, color="#555555", linewidth=.7)
        axis.axvline(50, color="#888888", linestyle="--", linewidth=1)
        axis.set_title(f"{case['marker'].capitalize()} target", fontsize=9)
        axis.set_xlabel("Control issue")
        axis.set_xticks([0, 25, 50, 75, 100])
        axis.grid(axis="y", alpha=.18)
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Shoulder-pan target (degrees)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", frameon=False, ncol=3, fontsize=8)
    fig.suptitle("SmolVLA approach commands under a matched teacher-input replay", fontsize=11)
    meta = {"CreationDate": None, "ModDate": None} if output.suffix == ".pdf" else {}
    fig.savefig(output, dpi=220, metadata=meta)
    plt.close(fig)


def witness_image(root, agent, sensor, first=False):
    from smart_swarm.episode import Replay

    replay = Replay(root, agent)
    frames = [(packet, camera) for packet in replay.packets() for camera in packet["cameras"]
              if camera["sensor_id"] == sensor]
    if not frames:
        raise ValueError(f"No retained pre-query image for {sensor}")
    packet, camera = frames[0 if first else -1]
    exposure = int(packet["source_time"]["monotonic_ns"]) / 1e9
    return replay.image(camera["frame_id"]), exposure


def primary_witnesses(path, output):
    from smart_swarm.episode import load, sha

    report = load(path)
    if report["scope"] != "independent primary test" or report["checkpoint_kind"] != "events_only":
        raise ValueError("The figure requires the sealed event-only primary score")
    group, = (row for row in report["outcomes"] if row["team_size"] == 3 and row["group"] == "primary-test-f1-00")
    score, = group["outcomes"]
    root = Path(score["source"])
    manifest_path = root / "manifest.json"
    manifest = load(manifest_path)
    if score["source_manifest_sha256"] != sha(manifest_path) or manifest["run_mode"] != "mujoco":
        raise ValueError("The selected example must retain its original simulation manifest")
    decision = score["learned_selective_joint_memory"]["decision"]
    decision_time = int(manifest["decision_boundary"]["monotonic_ns"]) / 1e9
    panels = (("SO101 observed placement", "so101-1", "so101-1/top", False),
              ("Tank before relocation", "tank-1", "tank-1/front", True),
              ("Tank after relocation", "tank-1", "tank-1/front", False),
              ("ROSMASTER at query", "rosmaster-1", "rosmaster-1/front", False))
    fig, axes = plt.subplots(2, 2, figsize=(7.1, 5.2), layout="constrained")
    for axis, (label, agent, sensor, first) in zip(axes.flat, panels, strict=True):
        rgb, time_s = witness_image(root, agent, sensor, first)
        axis.imshow(rgb)
        axis.set_title(f"{label} · {time_s:.2f} s", fontsize=9)
        axis.set_axis_off()
    fig.suptitle(f"Held-out F1 simulation\nROSMASTER chose {decision} at {decision_time:.2f} s", fontsize=11)
    meta = {"CreationDate": None, "ModDate": None} if output.suffix == ".pdf" else {}
    fig.savefig(output, dpi=220, metadata=meta)
    plt.close(fig)


def mobile_inspection_views(path, output):
    from mobile_inspection_case import verify_case
    from smart_swarm.episode import load

    cases = (("active-inspection-f6-00", "left", "Cube in left container"),
             ("active-inspection-f6-01", "surface", "Cube on surface"))
    branches = (("scan", "Moving camera"), ("stationary", "Stationary camera"))
    fig, axes = plt.subplots(2, 4, figsize=(9.0, 4.65), layout="constrained")
    for band, (group, expected, label) in enumerate(cases):
        root = path / group
        receipt = load(root / "receipt.json")
        prediction = verify_case(root, receipt)
        if receipt["evaluation"]["location"] != expected or any(
                row["decision"] != "request_inspection" for row in prediction["learned"].values()):
            raise ValueError(f"The declared illustrative F6 receipt did not match {group}")
        policy = load(root / "policy.json")
        for side, (arm, title) in enumerate(branches):
            record = policy[arm]
            for index, sample in enumerate(record["samples"]):
                axis = axes[band, 2 * side + index]
                axis.imshow(plt.imread(root / arm / sample["frame"]))
                center = sample["readout"]["rgb_marker_centers_px"]["cube"]
                if center is not None:
                    axis.scatter(*center, s=90, facecolors="none", edgecolors="#fff2a3", linewidths=1.1)
                elapsed = (sample["at_ns"] - record["start_ns"]) / 1e9
                axis.set_title(f"{title} · {elapsed:.2f} s", fontsize=8)
                axis.set_xticks([])
                axis.set_yticks([])
            answer = record["post_inspection_decision"]["answer"]
            axes[band, 2 * side].set_xlabel(f"Two-view result: {answer}", fontsize=8)
        axes[band, 0].set_ylabel(label, fontsize=9)
    fig.suptitle("Matched post-query ROSMASTER wrist views in two simulated F6 scenes", fontsize=11)
    meta = {"CreationDate": None, "ModDate": None} if output.suffix == ".pdf" else {}
    fig.savefig(output, dpi=220, metadata=meta)
    plt.close(fig)


def witness_mount_views(path, output):
    from smart_swarm.episode import Replay, load

    report = load(path)
    group = report["groups"][0]
    if report["status"] != "verified" or group["verified"] is not True:
        raise ValueError("The image panel requires a verified physical simulation pair")
    fig, axes = plt.subplots(2, 4, figsize=(8.8, 3.6), layout="constrained")
    phases = ("before_relocation", "after_relocation")
    for band, case in enumerate(group["cases"]):
        budget = case["capture"]["observation_budget"]
        for column, agent in enumerate(("tank-1", "tank-2")):
            playback = Replay(Path(case["path"]), agent)
            packets = {int(row["source_time"]["monotonic_ns"]): row for row in playback.packets()}
            for offset, phase in enumerate(phases):
                times = [row["source_time_ns"] for row in budget[phase] if row["agent_id"] == agent]
                packet = packets[times[0 if offset == 0 else -1]]
                camera, = packet["cameras"]
                axis = axes[band, 2 * column + offset]
                axis.imshow(playback.image(camera["frame_id"]))
                if band == 0:
                    axis.set_title(f"{agent.title()} · {'before' if offset == 0 else 'after'}", fontsize=9)
                axis.set_xticks([])
                axis.set_yticks([])
        axes[band, 0].set_ylabel(f"{case['owner'].title()} sees\nthe station", fontsize=9)
    fig.suptitle("The same simulated relocation with alternate informative camera mounts", fontsize=11)
    meta = {"CreationDate": None, "ModDate": None} if output.suffix == ".pdf" else {}
    fig.savefig(output, dpi=220, metadata=meta)
    plt.close(fig)


def smolvla_cpu_stages(path, output):
    report = json.loads(path.read_text())
    if report["status"] != "profiled" or report["verified_training_rows"] != 2:
        raise ValueError("The timing figure requires both original matched CPU requests")
    rows = report["requests"]
    colors = ("#27667b", "#6596a5", "#b76348", "#b9c3c6")
    labels = ("Vision encoder", "Other image and prefix work", "Action denoising", "Other worker time")
    fig, axis = plt.subplots(figsize=(7.1, 2.85), layout="constrained")
    for index, row in enumerate(rows):
        vision = sum(row["vision_encoding_each_image_ns"])
        prefix = row["prefix_before_first_denoising_step_ns"]
        denoise = row["exclusive_stage_ns"]["expert_denoising_total"]
        total = row["complete_worker_ns"]
        segments = (vision, prefix - vision, denoise, total - prefix - denoise)
        edge = 0.
        for duration, label, tint in zip(segments, labels, colors, strict=True):
            axis.barh(index, duration / 1e9, left=edge, color=tint, height=.5, label=label if index == 0 else None)
            edge += duration / 1e9
        axis.text(edge + .08, index, f"{edge:.2f} s", va="center", fontsize=9)
    window = report["chunk_coverage_ns_at_30hz"] / 1e9
    axis.axvline(window, color="#303030", linewidth=1.1, linestyle="--", label="50-action horizon at 30 Hz")
    maximum = max(row["complete_worker_ns"] / 1e9 for row in rows)
    axis.set(xlim=(0, maximum + 1.1), ylim=(len(rows) - .55, -.55), xlabel="Time per chunk (seconds)",
             yticks=list(range(len(rows))), yticklabels=[f"Archived issue {row['issue']}" for row in rows])
    axis.spines[["top", "right", "left"]].set_visible(False)
    axis.set_title("SmolVLA 000085 CPU inference")
    fig.legend(loc="outside lower center", ncol=3, frameon=False, fontsize=8)
    meta = {"CreationDate": None, "ModDate": None} if output.suffix == ".pdf" else {}
    fig.savefig(output, dpi=220, metadata=meta)
    plt.close(fig)


def primary_delivery_delay(path, output):
    report = json.loads(path.read_text())
    labels = ("0 ms", "20 ms", "20 ms\n+ 1 ns", "50 ms", "50 ms\n+ 1 ns")
    keys = ("0", "20000000", "20000001", "50000000", "50000001")
    counts = {}
    for head, teams in report["metrics_by_head_and_team"].items():
        counts[head] = {size: [(table[key]["container_correct"],
                               table[key]["resolvable_correct"] - table[key]["container_correct"])
                              for key in keys] for size, table in teams.items()}
    reference = counts["events_only"]["8"]
    if report["independent_source_groups"] != 100 or any(count != reference for teams in counts.values() for count in teams.values()):
        raise ValueError("The plot requires equal resolving counts across the declared heads and team sizes")
    fig, axes = plt.subplots(1, 2, sharey=True, figsize=(7.4, 3.25), layout="constrained")
    for column, (axis, title, tint) in enumerate(zip(axes, ("Correct container", "Correct surface"), ("#25667a", "#a35037"), strict=True)):
        values = [row[column] for row in reference]
        positions = range(len(keys))
        axis.step(positions, values, where="mid", color=tint, linewidth=1.6)
        axis.scatter(positions, values, color=tint, s=22)
        axis.set(xticks=list(positions), xticklabels=labels, ylim=(-2, 37), title=title)
        axis.grid(axis="y", alpha=.20)
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Correct first answers out of 34")
    fig.suptitle("Frozen selective memory under added reply delay\nBoth evidence heads at N = 3, 4 and 8", fontsize=11)
    fig.supxlabel("Tested modeled delay conditions", fontsize=10)
    meta = {"CreationDate": None, "ModDate": None} if output.suffix == ".pdf" else {}
    fig.savefig(output, dpi=220, metadata=meta)
    plt.close(fig)


def smolvla_initial_approach(path, output):
    report = json.loads(path.read_text())
    if report["status"] != "measured" or {row["marker"] for row in report["cases"]} != {"cyan", "violet"}:
        raise ValueError("Both original validation starts must be verified")
    series = (("teacher", "Teacher", "#236b55", "o"),
              ("initial", "Initialization", "#2a6284", "s"),
              ("fitted_after_85_updates", "After 85 updates", "#ab5039", "^"))
    fig, axes = plt.subplots(1, 2, sharey=True, figsize=(7.1, 3.25), layout="constrained")
    for column, (axis, case) in enumerate(zip(axes, report["cases"], strict=True)):
        rows = case["paired_native_samples"]
        issues = [row["issue"] for row in rows]
        for key, label, tint, marker in series:
            axis.plot(issues, [row[key][0] for row in rows], color=tint, marker=marker, markersize=4.5,
                      linewidth=1.5, label=label if column == 0 else None)
        axis.set(xticks=issues, xlabel="Issue index in first chunk", title=f"{case['marker'].title()} target")
        axis.axhline(0, color="#555555", linestyle="--", linewidth=1)
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Requested shoulder pan (degrees)")
    fig.suptitle("SmolVLA approach on two matched validation starts", fontsize=11)
    fig.legend(loc="outside lower center", ncol=3, frameon=False, fontsize=8)
    meta = {"CreationDate": None, "ModDate": None} if output.suffix == ".pdf" else {}
    fig.savefig(output, dpi=220, metadata=meta)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    figures = {"evidence": evidence_figure, "motor": learning_curve, "queries": query_tradeoff,
               "evidence-fit": evidence_fit, "scaling-windows": scaling_windows, "matched-budget": matched_budget,
               "primary-cost": primary_cost, "primary-inspection": primary_inspection,
               "primary-witnesses": primary_witnesses, "smolvla-teacher-response": smolvla_teacher_response,
               "mobile-inspection-views": mobile_inspection_views, "witness-mount-views": witness_mount_views,
               "smolvla-cpu-stages": smolvla_cpu_stages, "primary-delivery-delay": primary_delivery_delay,
               "smolvla-initial-approach": smolvla_initial_approach}
    parser.add_argument("kind", choices=figures)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    figures[args.kind](args.input, args.output)


if __name__ == "__main__":
    main()
