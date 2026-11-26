import pytest

from smart_swarm.view_calibration import FORWARD, REVERSE, apply, calibrated_replies, fit


def test_consistently_inverted_mount_can_learn_a_negative_scale():
    reference = {FORWARD: .5, REVERSE: .5}
    wrong_forward = {FORWARD: .02, REVERSE: .98}
    wrong_reverse = {FORWARD: .95, REVERSE: .05}
    groups = {f"{label}-{index}": (label, values)
              for label, values in ((FORWARD, wrong_forward), (REVERSE, wrong_reverse)) for index in range(3)}
    estimated = fit(groups, reference)
    assert estimated["slope"] < 0
    assert apply(wrong_forward, reference, estimated["slope"])[FORWARD] > .5
    assert apply(wrong_reverse, reference, estimated["slope"])[REVERSE] > .5
    assert apply(wrong_reverse, {FORWARD: .6, REVERSE: .4}, 0.) == pytest.approx({FORWARD: .6, REVERSE: .4})


def test_unsupported_mount_has_no_likelihood_in_calibrated_replay():
    reference = {FORWARD: .5, REVERSE: .5}
    groups = {"one": (FORWARD, {FORWARD: .01, REVERSE: .99})}
    not_fitted = fit(groups, reference)
    replies = [{"sender": name, "factor": factor, "probabilities": reference}
               for name, factor in (("so101-1", "containment"), ("tank-1", "association"),
                                    ("tank-2", "association"))]
    emitted = calibrated_replies(replies, {"tank-2": not_fitted}, reference, "tank-1")
    assert not_fitted["slope"] is None
    assert emitted == replies[:2]
