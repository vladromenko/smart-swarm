"""Measure whether ACT stalls at a stationary, time-dependent teacher hold."""

import argparse
from pathlib import Path

import numpy as np

from smart_swarm.episode import atomic, load
from smart_swarm.motor_diagnostics import first_failed_stage


def same_visual_hold(path):
    with np.load(path / "motor.npz", allow_pickle=False) as archive:
        early, late = 41, 58
        state = np.abs(archive["state"][early] - archive["state"][late])
        actions = archive["requested"].astype(np.float64)
        short_window = actions[early:early + 6], actions[late:late + 6]
        images = {}
        for name in ("wrist", "top"):
            camera = archive[name]
            difference = np.abs(camera[early].astype(float) - camera[late].astype(float))
            images[name] = {"max_pixel_difference": float(difference.max()),
                            "mean_pixel_difference": float(difference.mean()), "pixel_bytes_equal": bool(not difference.any())}
        contrast = np.abs(short_window[0] - short_window[1])
        return {"observation_indices_zero_based": [early, late], "state_absolute_delta": state.tolist(),
                "rgb_difference": images, "maximum_requested_delta_in_first_six_actions": contrast.max(axis=0).tolist()}


def closed_loop(case):
    observations = case["feedback"]
    first_descend = next(row for row in observations if row["tick"] >= 75)
    settled = next(row for row in observations if row["tick"] >= 90)
    gripper = np.array([row["requested"] for row in observations])[:, -1]
    clips = np.asarray([np.abs(np.subtract(row["requested"], row["decoded_and_clipped"])) for row in observations])
    return {"seed": case["seed"], "first_failed_stage": first_failed_stage(observations),
            "minimum_tcp_cube_distance_m": case["minimum_tcp_cube_distance_m"], "stable_containment": case["stable_containment"],
            "learned_requested_during_descent": first_descend["requested"],
            "teacher_requested_during_descent": first_descend["teacher_requested_at_index"],
            "learned_requested_at_tick_90": settled["requested"], "teacher_requested_at_tick_90": settled["teacher_requested_at_index"],
            "applied_at_tick_90": settled["previous_applied_end_of_interval"],
            "max_clipping_adjustment": clips.max(axis=0).tolist(), "gripper_range_percent": [float(gripper.min()), float(gripper.max())]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fit_report", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    fit = load(args.fit_report)
    ancestry = load(Path(fit["dataset"]) / "train/smart-swarm-lineage.json")
    commands = [same_visual_hold(Path(row["raw_path"])) for row in ancestry["episodes"]]
    report = {"recorded_example_comparisons": commands, "closed_loop": [closed_loop(case) for case in fit["fitting_cases"]],
              "accepted_fresh_eval_worlds_consumed": 0,
              "interpretation": "ACT has no elapsed-phase input. Near-stationary inputs persist during the teacher's timed hold. "
              "A later near-identical input can be followed by different setpoints inside the first six executed actions. "
              "The closed-loop traces show where the fitted policy remains near the hold. This supports a state/phase ambiguity diagnosis."}
    atomic(args.output, report)


if __name__ == "__main__":
    main()
