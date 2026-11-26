"""Six witnessed histories on one MuJoCo world with physical event checks."""

from collections import defaultdict
import time

import mujoco
import numpy as np

from .contract import interval, time_at
from .episode import LABELS, TRUTH
from .physics import STEP, TABLE, World
from .story import AGENTS, Story
from .teacher import move_box, run_insertion, truth
from .vision import insertion, relocation


class EvidenceStory(Story):
    """Sparse visual capture with an unchanged dense physical controller."""

    def __init__(self, path, family, world, seed, miss):
        super().__init__(path, family, world, seed, miss)
        self.stage_data = defaultdict(list)

    def arm_sample(self, phase, index, start, requested, applied):
        if index % 20 == 19:
            outcome = truth(self.world)
            self.stage_data[phase].append(outcome)
            self.recorder.append(TRUTH, "physics", {"at": time_at(self.world.ns),
                                                     "teacher_phase": phase, **outcome})
        if self.family != "F6" and index == 29:
            self.capture(AGENTS[0])

    def motion_sample(self):
        self.relocation_ticks += 1
        if self.relocation_ticks % 9 == 0:
            self.capture(AGENTS[1])

    def scene(self, pickup):
        self.approach()
        self.capture(AGENTS[0])
        run_insertion(self.teacher, self.arm_sample, pickup=pickup)
        self.world.step(200)
        after_insert = truth(self.world)
        if self.family == "F3":
            repack(self, after_insert)
        after_contents = settled_truth(self.world)
        if self.family != "F6":
            packet, pictures = self.capture(AGENTS[0])
            self.add_evidence(insertion(packet, pictures["so101-1/top"]))
        self.relocate()
        self.capture(AGENTS[2])
        self.recorder.append(TRUTH, "physics", {"at": time_at(self.world.ns), "teacher_phase": "post_relocation",
                                                **settled_truth(self.world)})
        self.recorder.meta["physical_story_gate"] = physical_checks(self, after_insert, after_contents)

    def relocate(self):
        start, first = self.capture(AGENTS[1])
        move_box(self.world, self.motion_sample)
        self.world.step(200)
        end, last = self.capture(AGENTS[1])
        self.add_evidence(relocation([start, end], first["tank-1/front"], last["tank-1/front"]))


def repack(story, inserted):
    if not inserted["inside_carrier"]:
        raise ValueError("F3 first insertion must succeed before removing the cube")
    x, y, z = inserted["cube_xyz_m"]
    pick = (x, y, z - TABLE + .003)
    phases = [
        ("remove_approach", (x, y, .115), .9, 40, 2),
        ("remove_descend", pick, .9, 40, 2),
        ("remove_grasp", pick, -.12, 40, 2),
        ("remove_lift", (x, y, .160), -.12, 40, 2),
        ("remove_carry", (.24, -.10, .145), -.12, 20, 3),
        ("remove_release", (.24, -.10, .026), .9, 40, 2),
        ("remove_withdraw", (.24, -.10, .166), .9, 40, 2),
    ]
    for name, point, grip, speed, seconds in phases:
        story.teacher.advance(name, point, grip, speed, seconds, story.arm_sample)
    story.world.step(200)


def physical_checks(story, first, before):
    final = settled_truth(story.world)
    family = story.family
    insertion_pass = first["inside_carrier"] and first["released"] and first["stationary"]
    carry = story.stage_data["transfer"] + story.stage_data["lower"]
    supported = sum(len(point["gripper_contact_bodies"]) >= 2 for point in carry)
    dx = final["carrier_xyz_m"][0] - first["carrier_xyz_m"][0]
    target_missed = (story.miss or family == "F2") and family != "F3"
    expected_relation = "surface" if family in ("F3", "F2") or target_missed else "carrier"
    if dx < .12 or final["cube_relation"] != expected_relation or not final["stationary"]:
        raise ValueError(f"{family} failed final physical test: displacement={dx}, relation={final['cube_relation']}")
    if not target_missed and (not insertion_pass or supported < 5):
        raise ValueError(f"{family} lacks verified initial insertion, carry or release")
    if target_missed and (first["inside_carrier"] or supported):
        raise ValueError(f"{family} attempted failure physically succeeded on the orange target")
    nested = verifies_nesting(story.world) if family == "F5" else False
    if family == "F5" and not nested:
        raise ValueError("F5 bin is outside the tray footprint or moving at the decision boundary")
    removed = len([p for p in story.stage_data["remove_lift"] if p["gripper_contact_bodies"]]) >= 2
    if family == "F3" and (before["cube_relation"] != "surface" or not removed):
        raise ValueError("F3 lacked contact-verified removal")
    return {"first_insertion": insertion_pass, "carry_contact_samples": supported,
            "contents_after_witness": before["cube_relation"], "final_relation": final["cube_relation"],
            "displacement_m": dx, "nested": nested, "removed": removed}


def verifies_nesting(world):
    carrier = world.data.body("carrier")
    delta = carrier.xpos - world.data.body("nest_tray").xpos
    footprint = np.abs(carrier.xmat.reshape(3, 3)) @ [.061, .052, .024]
    start = int(world.model.joint("box_pose").dofadr[0])
    still = np.linalg.norm(world.data.qvel[start:start + 3]) < .02
    return bool(still and (np.abs(delta[:2]) + footprint[:2] < [.102, .065]).all()
                and -.001 <= delta[2] <= .009)


def settled_truth(world):
    report = truth(world)
    x, y, z = report["cube_xyz_m"]
    # The contact buffer may be empty on a resting solver step at the table boundary.
    on_plane = .00 < x < .73 and -.33 < y < .35 and abs(z - TABLE - .016) < .001
    if report["cube_relation"] == "outside" and on_plane and report["stationary"] and report["released"]:
        report["cube_relation"] = "surface"
    return report


def reveal(story, decision):
    before = settled_truth(story.world)
    story.world.step(200)
    deadline = story.world.ns
    after = settled_truth(story.world)
    box = story.world.data.body("carrier").xpos
    decoy = story.world.data.body("decoy").xpos
    slot = "left" if box[0] < decoy[0] else "right"
    answer = slot if before["inside_carrier"] else before["cube_relation"]
    camera = story.recorder.frame(TRUTH, "evaluator/overhead", 1, story.world.image("reveal"), deadline)
    output = {"camera": camera, "ground_truth": answer, "relation": after["cube_relation"],
              "target_state_time_or_interval": interval(decision, deadline),
              "relations_preserved_until_reveal": before["cube_relation"] == after["cube_relation"],
              "observed_at": time_at(deadline)}
    story.recorder.append(TRUTH, "reveal", output)
    return deadline, output, slot


def prepare_world(assets, seed, family, *, variant, seconds=100, team_size=3):
    hidden = ("surface_cube" if variant == "hidden-surface" else "cube") if family == "F6" else None
    screened = family == "F4" and variant == "screened-association-v2"
    layout = "F4-screened" if screened else family
    renderer = (160, 120)
    reverse = bool(seed // 4 % 2)
    world = World(assets, seed=seed, miss=family == "F2", f6_target=hidden,
                  swap_markers=bool(seed % 2), evidence_family=layout, frame_size=renderer,
                  evidence_reverse_slots=reverse, timeout_s=seconds, team_size=team_size)
    rng = world.rng
    offset = rng.uniform(-.002, .002, size=2)
    raw = int(world.model.joint("cube_pose").qposadr[0])
    world.data.qpos[raw:raw + 2] += offset
    slip = float(rng.uniform(.96, 1.04))
    world.model.geom_friction[world.model.geom("cube_geom").id, 0] = slip
    fovy_offset = float(rng.uniform(-.25, .25))
    world.model.cam_fovy += fovy_offset
    mujoco.mj_forward(world.model, world.data)
    return world, {"cube_xy_offset_m": offset.tolist(), "sliding_friction": slip,
                   "camera_fovy_offset_deg": fovy_offset, "alternate_decoy_position": reverse and (family != "F4" or screened),
                   **({"association_layout_version": variant} if screened else {})}, (
                       .24 + offset[0], -.06 + offset[1])
