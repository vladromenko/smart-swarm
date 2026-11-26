import numpy as np
import pytest

from smart_swarm.contract import time_at
from smart_swarm.episode import DELIVERIES, TRUTH, Recorder, rows
from smart_swarm.physical_slot_reference import REVERSED, SAME, SCOPE_FLAG, UNKNOWN, association_image, verify_decision_reference
from smart_swarm.physical_story_replay import AGENTS, write_decision


def frame(marks, order=None):
    image = np.zeros((60, 120, 3), dtype=np.uint8)
    colors = {"cyan": (0, 220, 220), "violet": (180, 0, 240), "cube": (235, 80, 10),
              "green": (0, 230, 0), "blue": (0, 0, 230)}
    if order is not None:
        first, second = ("green", "blue") if order else ("blue", "green")
        marks = [*marks, (first, 5, 5), (second, 101, 5)]
    for label, x, y in marks:
        image[y:y + 7, x:x + 9] = colors[label]
    return image


def recording(tmp_path, tank_orders, receiver_order):
    clock = {"host_id": "homebrain", "boot_id": "fixture", "kind": "monotonic"}
    cameras = {AGENTS[0]: [AGENTS[0] + "/top"], **{agent: [agent + "/front"] for agent in AGENTS[1:]}}
    recorder = Recorder(tmp_path / "source", "scope-test", "group", cameras, {"run_mode": "hardware", SCOPE_FLAG: True}, clock=clock)
    picture = frame([("cyan", 20, 20), ("cube", 32, 20)])
    arm_packet = recorder.capture(AGENTS[0], {cameras[AGENTS[0]][0]: picture}, 10, [], [])
    arm = [(arm_packet, {cameras[AGENTS[0]][0]: picture})]
    tank = []
    for sample, order in enumerate(tank_orders):
        picture = frame([("cyan", 14 if sample == 0 else 41, 28), ("violet", 78, 28)], order)
        tank.append((recorder.capture(AGENTS[1], {cameras[AGENTS[1]][0]: picture}, 20 + sample, [], []), picture))
    recorder.capture(AGENTS[2], {cameras[AGENTS[2]][0]: frame([], receiver_order)}, 40, [], [])
    return recorder, arm, tank


@pytest.mark.parametrize(
    "tank_orders, receiver_order, mapping, decision",
    [((True, True, True), True, SAME, "left"), ((False, False, False), True, REVERSED, "right"),
     ((False, True, False), True, UNKNOWN, "request_inspection"), ((False, False, False), False, UNKNOWN, "request_inspection"),
     ((None, False, False), True, UNKNOWN, "request_inspection")],
)
def test_native_rule_maps_sides_from_causal_landmarks(tmp_path, tank_orders, receiver_order, mapping, decision):
    recorder, arm, tank = recording(tmp_path, tank_orders, receiver_order)
    outcome = write_decision(recorder, arm, tank, 50, 60, {"status": "unavailable", "reason": "fixture"})
    assert outcome["slot_reference"]["mapping"] == mapping
    assert outcome["query_all"]["decision"] == decision
    reports = rows(recorder.root / DELIVERIES)
    assert len(reports) == (1 if mapping == UNKNOWN else 2)
    recorder.append(TRUTH, "injected", {"terminal": "right", "private_slot_hint": "forbidden"})
    manifest = {**recorder.meta, "query_boundary": time_at(50, recorder.clock)}
    assert verify_decision_reference(recorder.root, manifest, outcome) == outcome["slot_reference"]
    wrong = {**outcome, "slot_reference": outcome["slot_reference"] | {"mapping": "fabricated"}}
    with pytest.raises(ValueError, match="differs from its original scoped RGB"):
        verify_decision_reference(recorder.root, manifest, wrong)


def test_reference_colors_are_hidden_from_the_association_encoder():
    marks = [("cyan", 20, 20), ("violet", 78, 20)]
    first, second = frame(marks, True), frame(marks, False)
    assert not np.array_equal(first, second)
    assert np.array_equal(association_image(first), association_image(second))
    assert np.array_equal(association_image(first), frame(marks))
