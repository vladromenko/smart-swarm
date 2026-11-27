from base64 import b64encode
from hashlib import sha256

import pytest

from smart_swarm.act_trace import ACTInputTrace
from smart_swarm.episode import load


def test_act_trace_binds_exact_input_and_refuses_a_queue_gap(tmp_path):
    trace = ACTInputTrace(tmp_path / "act-input-trace")
    wrist = b"\0" * (160 * 120 * 3)
    top = b"\xff" * len(wrist)
    request = {"type": "action", "trace_act": True, "state": [0.] * 6, "task": "fixture",
               "images": [b64encode(frame).decode() for frame in (wrist, top)]}
    action = [1.] * 6
    response = {"action": action, "act_queue": {"before": 0, "after": 1}}
    exchange = {"request": request, "response": response}
    control = {"requested": action}
    trace.save(0, 100, 133, exchange, control)
    with pytest.raises(ValueError, match="queue counts"):
        trace.save(1, 133, 166, exchange, control)
    assert trace.report()["saved_issues"] == 1
    response["act_queue"] = {"before": 1, "after": 0}
    trace.save(1, 133, 166, exchange, control)
    row = load(trace.root / "issue-0001.json")
    assert row["chunk_origin_issue"] == 0
    assert row["worker_request"] == request
    assert row["ordered_rgb_sha256"] == [sha256(frame).hexdigest() for frame in (wrist, top)]
    assert trace.report()["refreshes"] == 1
