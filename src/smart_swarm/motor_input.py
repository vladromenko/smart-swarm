"""Versioned camera conditioning and issue-time contract for the motor pilot."""

import numpy as np

PROFILE = "aligned-rgb-target-v3"
CLOCKED_PROFILE = "aligned-rgb-target-clock-v4"
FLOAT32_COMMAND_PROFILE = "float32-native-before-execution-v1"
EXACT_REPLAY_ISSUES = (0, 41, 58, 150, 240, 330, 450, 509)
TARGET_RGB = {"cyan": (0, 245, 245), "violet": (194, 8, 250)}
CAMERAS = ("so101-1/wrist", "so101-1/top")
ISSUE_CLOCK_BITS = 10


def conditioned_top(rgb, target, *, issue_index=None):
    """Add the requested marker and optional local clock to the overhead image."""
    if target not in TARGET_RGB:
        raise ValueError(f"Unknown requested bin marker {target!r}")
    output = np.asarray(rgb, dtype=np.uint8).copy()
    output[2:14, 2:14] = (20, 20, 20)
    output[4:12, 4:12] = TARGET_RGB[target]
    if issue_index is not None:
        mark_issue_clock(output, issue_index)
    return output


def mark_issue_clock(image, issue_index):
    """Present elapsed control issues as visible bits and a coarse ramp."""
    if type(issue_index) is not int or not 0 <= issue_index < 2**ISSUE_CLOCK_BITS:
        raise ValueError("Issue index must be an unsigned integer in the clock range")
    if image.shape[0] < 14 or image.shape[1] < 60:
        raise ValueError("Overhead image does not have room for the clock cue")
    image[2:14, 16:60] = 20
    cells = image[4:9, 18:58].reshape(5, ISSUE_CLOCK_BITS, 4, 3)
    bits = (issue_index >> np.arange(ISSUE_CLOCK_BITS - 1, -1, -1)) & 1
    cells[:] = (30 + 210 * bits)[None, :, None, None]
    width = issue_index * 40 // (2**ISSUE_CLOCK_BITS - 1)
    image[10:12, 18:18 + width] = 240


def instruction(base, target):
    if target not in TARGET_RGB:
        raise ValueError(f"Unknown requested bin marker {target!r}")
    return f"{base} The selected bin is marked {target}."


def verify_issue_time(times, starts):
    ticks = np.arange(1, len(times) + 1, dtype=np.int64)
    expected_times = 100_000_000 + ticks * 2000 // 30 * 500_000
    expected_starts = np.r_[100_000_000, expected_times[:-1]]
    if not np.array_equal(times, expected_times) or not np.array_equal(starts, expected_starts):
        raise ValueError("Measured observation and command issue times disagree with aligned-sample-v2")
