"""One physical workspace with a separated set of local task cameras."""

import hashlib
import json
from math import atan2, cos, sin
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco

TABLE = .22
STEP = .0005
CAMERAS = {
    "so101-1": {"so101-1/wrist": "so/wrist_cam", "so101-1/top": "station"},
    "tank-1": {"tank-1/front": "tank/story_camera"},
    "rosmaster-1": {"rosmaster-1/front": "ros/story_camera"},
}
EVALUATOR = "evaluator/overhead"
SOLVER = {"timestep": str(STEP), "integrator": "implicitfast", "iterations": "100",
          "ls_iterations": "20", "cone": "elliptic", "impratio": "10"}
MARKERS = ("cyan", "violet")
MARKER_RGBA = {"cyan": ".01 .95 .95 1", "violet": ".76 .03 .98 1"}
PARKED_OBSERVERS = (
    ("tank-2", "tank2", "tank", (.94, .80, 0.)),
    ("tank-3", "tank3", "tank", (.39, 1.24, 0.)),
    ("tank-4", "tank4", "tank", (1.43, .95, 0.)),
    ("rosmaster-2", "ros2", "rosmaster", (.03, .86, 0.)),
    ("rosmaster-3", "ros3", "rosmaster", (.89, 1.36, 0.)),
)


def build(asset_root: Path, swap_markers: bool = False, *, f6_target=None, legacy_reveal=False, evidence_family=None,
          evidence_reverse_slots=False, team_size=3):
    if team_size not in (3, 4, 8):
        raise ValueError("The study supports simulated teams of 3, 4 or 8 agents")
    root, world = workspace(swap_markers, legacy_reveal=legacy_reveal)
    if evidence_family:
        extend_evidence_scene(world, evidence_family, evidence_reverse_slots)
    if f6_target is not None:
        add_matched_target(world, f6_target)
    spec = mujoco.MjSpec.from_string(ET.tostring(root, encoding="unicode"))
    positions = {"so": (0, 0, TABLE), "ros": (.44, -.62, 0), "tank": (.43, .73, 0)}
    assets = {}
    for namespace, robot in (("so", "so101"), ("ros", "rosmaster"), ("tank", "tank")):
        xml, provenance = load_component(asset_root, robot)
        configure_component(xml, robot)
        child = mujoco.MjSpec.from_string(ET.tostring(xml, encoding="unicode"))
        spec.attach(child, prefix=f"{namespace}/",
                    frame=spec.worldbody.add_frame(pos=positions[namespace]))
        assets[robot] = provenance
    extras = attach_parked_observers(spec, asset_root, team_size)
    model = spec.compile()
    return model, {"assets": assets, "solver": SOLVER, "mujoco": mujoco.__version__,
                   "model_sha256": hashlib.sha256(spec.to_xml().encode()).hexdigest(),
                   "contact": {"solref": [.001, 1], "solimp": [.95, .99, .001, .5, 2],
                               "object_and_box_friction": [1, .03, .003]},
                   "camera_calibration": "nominal synthetic, not calibrated to hardware",
                   "tank_motion": "force-driven planar surrogate",
                   "rosmaster_motion": "stationary base, actuated arm held for viewing",
                   **({"parked_observers": extras} if extras else {})}


def attach_parked_observers(spec, asset_root, team_size):
    roster = {}
    for agent, namespace, family, position in PARKED_OBSERVERS[:team_size - 3]:
        xml, _ = load_component(asset_root, family)
        configure_component(xml, family)
        child = mujoco.MjSpec.from_string(ET.tostring(xml, encoding="unicode"))
        dx, dy = .42 - position[0], .10 - position[1]
        yaw = atan2(dx, -dy) if family == "tank" else atan2(-dx, dy)
        frame = spec.worldbody.add_frame(pos=position, quat=(cos(yaw / 2), 0., 0., sin(yaw / 2)))
        spec.attach(child, prefix=f"{namespace}/", frame=frame)
        roster[agent] = {"namespace": namespace, "family": family, "parked_position_m": position,
                         "heading_rad": yaw, "camera": f"{namespace}/story_camera"}
    return roster


def load_component(asset_root, robot):
    path = asset_root / f"{robot}.xml"
    source = path.read_bytes()
    provenance = json.loads((asset_root / f"{robot}.json").read_text())
    if hashlib.sha256(source).hexdigest() != provenance["local_xml_sha256"]:
        raise ValueError(f"{robot} model checksum changed")
    root = ET.fromstring(source)
    for mesh in root.findall("asset/mesh[@file]"):
        local = asset_root / mesh.attrib["file"]
        if hashlib.sha256(local.read_bytes()).hexdigest() != local.stem:
            raise ValueError(f"Geometry checksum changed: {local.name}")
        mesh.set("file", str(local.resolve()))
    return root, provenance


def configure_component(root, robot):
    options = root.find("option")
    options.attrib.clear()
    options.attrib.update(SOLVER)
    for part in root.findall("keyframe") + root.findall("visual") + root.findall("statistic"):
        root.remove(part)
    visual = ET.SubElement(root, "visual")
    ET.SubElement(visual, "global", offwidth="640", offheight="480")
    asset = root.find("asset")
    for item in asset.findall("texture[@type='skybox']"):
        asset.remove(item)
    world = root.find("worldbody")
    names = {"floor", "target_box", "task_cube", "target", "dock_zone", "overview"}
    for item in list(world):
        if item.tag == "light" or item.get("name") in names:
            world.remove(item)
    if robot == "tank":
        mount = world.find("body[@name='base']")
        camera(mount, "story_camera", ".02 -.19 .248", "-1 0 0 0 0 1", "51")
    if robot == "rosmaster":
        camera(world, "story_camera", "0 .19 .245", "1 0 0 0 0 1", "45")


def workspace(swap_markers, *, legacy_reveal=False):
    root = ET.Element("mujoco", model="Smart Swarm contact story")
    ET.SubElement(root, "option", **SOLVER)
    ET.SubElement(root, "statistic", extent="2.0", center=".32 0 .22")
    visual = ET.SubElement(root, "visual")
    ET.SubElement(visual, "global", offwidth="640", offheight="480")
    ET.SubElement(visual, "headlight", ambient=".43 .43 .43", diffuse=".46 .46 .46")
    defaults = ET.SubElement(root, "default")
    ET.SubElement(defaults, "geom", friction="1 .03 .003", solref=".001 1",
                  solimp=".95 .99 .001 .5 2")
    world = ET.SubElement(root, "worldbody")
    ET.SubElement(world, "light", pos=".4 -.4 1.5", dir="0 0 -1", castshadow="false")
    ET.SubElement(world, "geom", type="plane", size="2 2 .03", rgba=".21 .25 .28 1")
    geom(world, "station_surface", ".365 .01 .2", ".385 .36 .02", ".29 .33 .37 1")
    camera(world, "station", ".29 .02 .70", "1 0 0 0 1 0", "57")
    reveal = (".39 .02 .78", "60") if legacy_reveal else (".35 -.01 .80", "64")
    camera(world, "reveal", reveal[0], "1 0 0 0 1 0", reveal[1])
    cube = ET.SubElement(world, "body", name="cube", pos=f".24 -.06 {TABLE + .018}")
    ET.SubElement(cube, "freejoint", name="cube_pose")
    geom(cube, "cube_geom", "0 0 0", ".016 .016 .016", "1 .14 .015 1", mass=".035")
    tracks = MARKERS[::-1] if swap_markers else MARKERS
    add_box(world, "carrier", .24, tracks[0], free=True)
    add_box(world, "decoy", .62, tracks[1], free=False)
    add_pusher(root, world)
    return root, world


def camera(parent, name, pos, xyaxes, fovy):
    ET.SubElement(parent, "camera", name=name, pos=pos, xyaxes=xyaxes, fovy=fovy)


def geom(parent, name, pos, size, rgba, **attributes):
    ET.SubElement(parent, "geom", name=name, type="box", pos=pos, size=size,
                  rgba=rgba, **attributes)


def add_box(world, name, x, marker, *, free):
    body = ET.SubElement(world, "body", name=name, pos=f"{x} .10 {TABLE + .0005}")
    if free:
        ET.SubElement(body, "freejoint", name="box_pose")
    neutral = ".43 .47 .51 1"
    geom(body, name + "_base", "0 0 .004", ".06 .05 .004", neutral, mass=".15")
    for side in (-1, 1):
        geom(body, f"{name}_y{side}", f"0 {side*.045} .024", ".06 .005 .024",
             neutral, mass=".03")
        geom(body, f"{name}_x{side}", f"{side*.055} 0 .024", ".005 .04 .024",
             neutral, mass=".03")
    geom(body, name + "_tag", "0 .0506 .030", ".035 .001 .012", MARKER_RGBA[marker],
         contype="0", conaffinity="0", mass="0")
    geom(body, name + "_rim_tag", "0 .044 .0483", ".035 .004 .0006", MARKER_RGBA[marker],
         contype="0", conaffinity="0", mass="0")


def add_matched_target(world, target):
    orange, blue = "1 .14 .015 1", ".045 .18 .85 1"
    if target == "surface_cube":
        world.find("body[@name='cube']/geom").set("rgba", blue)
    prop = ET.SubElement(world, "body", name="surface_cube", pos=f".044 -.258 {TABLE + .018}")
    ET.SubElement(prop, "freejoint", name="surface_cube_pose")
    geom(prop, "surface_cube_geom", "0 0 0", ".016 .016 .016",
         orange if target == "surface_cube" else blue, mass=".035")
    gray = ".31 .35 .39 1"
    for side in (-1, 1):
        geom(world, f"occluder_y{side}", f".044 {-.258 + side*.05} {TABLE+.044}",
             ".064 .004 .044", gray)
    geom(world, "occluder_east", f".104 -.258 {TABLE+.044}", ".004 .05 .044", gray)


def add_pusher(root, world):
    body = ET.SubElement(world, "body", name="external_slide", pos=f".13 .10 {TABLE + .019}")
    ET.SubElement(body, "joint", name="pusher_axis", type="slide", axis="1 0 0",
                  range="-.14 .26", damping="12")
    geom(body, "pusher_face", "0 0 0", ".011 .036 .015", ".40 .43 .47 1", mass=".4")
    actuator = ET.SubElement(root, "actuator")
    ET.SubElement(actuator, "position", name="external_drive", joint="pusher_axis",
                  kp="170", kv="27", forcerange="-14 14", ctrlrange="-.13 .245")


def extend_evidence_scene(world, family, reverse_slots):
    decoy = world.find("body[@name='decoy']")
    if reverse_slots and family not in ("F4", "F4-screened"):
        decoy.set("pos", f".36 .235 {TABLE + .0005}")
    if family in ("F4", "F4-screened"):
        place_unmarked_decoy(world, decoy, family == "F4-screened", reverse_slots)
        for body in ("carrier", "decoy"):
            for name in (body + "_tag", body + "_rim_tag"):
                target = world.find(f"body[@name='{body}']/geom[@name='{name}']")
                target.set("rgba", ".43 .47 .51 1")
    if family == "F5":
        outer = ET.SubElement(world, "body", name="nest_tray", pos=f".427 .1 {TABLE}")
        geom(outer, "tray_floor", "0 0 -.00025", ".102 .067 .00025",
             ".27 .30 .36 1", mass=".10")
        for side in (-1, 1):
            geom(outer, f"tray_side_{side}", f"0 {side*.065} .012", ".102 .002 .012",
                 ".27 .30 .36 1", mass=".025")
        geom(outer, "tray_end", ".100 0 .012", ".002 .067 .012", ".27 .30 .36 1", mass=".025")


def place_unmarked_decoy(world, decoy, screened, reverse_slots):
    x = .62 if screened and reverse_slots else .39
    decoy.set("pos", f"{x} .220 {TABLE + .0005}")
    if screened:
        geom(world, "association_screen", ".447 .445 .265", ".016 .012 .080", ".26 .31 .38 1", group="2")
