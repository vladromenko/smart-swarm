import numpy as np

from smart_swarm.identity_support import marker_support


def test_joint_support_requires_both_markers_in_one_available_frame():
    cyan = np.zeros((12, 12, 3), dtype=np.uint8)
    violet = cyan.copy()
    cyan[:4, :4] = (2, 230, 230)
    violet[6:10, 6:10] = (200, 3, 245)
    separate = marker_support([cyan, violet])
    assert separate["marker_frames"] == {"cyan": 1, "violet": 1}
    assert not separate["supported"]
    joint = marker_support([np.maximum(cyan, violet)])
    assert joint["both_marker_frames"] == 1
    assert joint["supported"]
