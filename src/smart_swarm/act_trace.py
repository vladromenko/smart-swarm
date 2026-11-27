"""Exact ACT input and execution receipts for bounded simulator evaluation."""

from base64 import b64decode
from hashlib import sha256

from .episode import atomic
from .motor import IMAGE_SIZE, SENSORS

TRACE_ISSUES = 204


class ACTInputTrace:
    def __init__(self, root):
        root.mkdir(exist_ok=False, parents=True)
        self.root = root
        self.next_issue = 0
        self.origin = None
        self.refreshes = 0
        self.next_observed_ns = None
        self.queued_after = 0

    def save(self, issue, observed_ns, next_observed_ns, exchange, control):
        request, response = exchange["request"], exchange["response"]
        counts = response["act_queue"]
        before, after = counts["before"], counts["after"]
        if any(type(value) is not int or value < 0 for value in (before, after)):
            raise ValueError("ACT trace requires valid native queue counts")
        if (issue != self.next_issue or issue >= TRACE_ISSUES or before != self.queued_after
                or not observed_ns < next_observed_ns
                or (self.next_observed_ns is not None and observed_ns != self.next_observed_ns)
                or response["action"] != control["requested"] or (before and after != before - 1)):
            raise ValueError("ACT trace has inconsistent issue times, queue counts or requested action")
        origin = issue if before == 0 else self.origin
        if origin is None:
            raise ValueError("ACT trace cannot bind the issuing chunk")
        frames = [b64decode(frame, validate=True) for frame in request["images"]]
        if len(frames) != len(SENSORS) or any(len(frame) != IMAGE_SIZE[0] * IMAGE_SIZE[1] * 3 for frame in frames):
            raise ValueError("ACT trace requires the two exact native RGB input arrays")
        atomic(self.root / f"issue-{issue:04}.json",
               {"issue": issue, "observed_at_ns": observed_ns, "next_observed_at_ns": next_observed_ns,
                "chunk_origin_issue": origin, "camera_ids": SENSORS, "image_size_wh": IMAGE_SIZE,
                "ordered_rgb_sha256": [sha256(frame).hexdigest() for frame in frames],
                "worker_request": request, "worker_response": response, "executor": control})
        self.next_issue += 1
        self.origin = origin
        self.refreshes += int(origin == issue)
        self.next_observed_ns = next_observed_ns
        self.queued_after = after

    def report(self):
        return {"path": self.root.name, "saved_issues": self.next_issue, "refreshes": self.refreshes,
                "issue_cap": TRACE_ISSUES}
