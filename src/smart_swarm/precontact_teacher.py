"""Opt-in simulated teaching geometry for a contact-free wrist observation."""

from .teacher import Teacher

STOCK = (0., .019)
PRECONTACT = (-.002, .021)


class ContactFreeApproach:
    """Keep the original episode timing and change only the descent and grasp pose."""

    def __init__(self, world, settings=PRECONTACT):
        self.teacher = Teacher(world)
        self.x_offset, self.height = settings

    def advance(self, phase, point, grip, joint_speed, seconds, on_sample):
        x, y, z = point
        goal = (x + self.x_offset, y, self.height) if phase in ("descend", "grasp") else (x, y, z)
        self.teacher.advance(phase, goal, grip, joint_speed, seconds, on_sample)
