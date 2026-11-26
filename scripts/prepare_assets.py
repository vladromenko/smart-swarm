"""One-time, local-only extraction using Explorer's pinned, offline model builders."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import xml.etree.ElementTree as ET

REVISION = "89bba26780619ddec7355da60c7b57a006711daf"


def prepare(args):
    import subprocess

    explorer = args.explorer.resolve()
    commit = subprocess.check_output(["git", "-C", str(explorer), "rev-parse", "HEAD"], text=True).strip()
    if commit != REVISION:
        raise ValueError(f"Expected Explorer {REVISION}, found {commit}")
    tracked = ["explorer4/sim", "explorer4/paths.py", "configs/tank-hardware-evidence.json"]
    subprocess.run(["git", "-C", str(explorer), "diff", "--quiet", "HEAD", "--", *tracked], check=True)
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    cache = args.source_cache.resolve()
    tank = args.output / "tank-cache"
    shutil.copytree(cache / "mentorpi-t1-cad", tank, dirs_exist_ok=True)
    os.environ.update(
        EXPLORER4_DATA=str(args.output / "empty-explorer-data"),
        EXPLORER4_SIM_ASSETS=str(cache / "mujoco_menagerie/robotstudio_so101"),
        EXPLORER4_M3PRO_ASSETS=str(cache / "yahboom-downloads/m3pro-orin/m3pro_ws/src/M3Pro"),
        EXPLORER4_TANK_ASSETS=str(tank),
    )
    sys.path.insert(0, str(explorer))
    export(args.output)


def export(output):
    from explorer4.sim.mobile import model_xml
    from explorer4.sim.models import rosmaster_arm_model, so101_model

    tank, mass, notes = model_xml("tank")
    notes["planar_mass_kg"] = mass
    entries = {"so101": so101_model(), "rosmaster": rosmaster_arm_model(), "tank": (tank, notes)}
    for name, (xml, provenance) in entries.items():
        path = output / f"{name}.xml"
        localize_meshes(xml, output, path)
        provenance.update(explorer_commit=REVISION, local_xml_sha256=digest(path))
        (output / f"{name}.json").write_text(json.dumps(provenance, indent=2) + "\n")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def localize_meshes(xml, output, destination):
    tree = ET.fromstring(xml)
    compiler = tree.find("compiler")
    meshdir = Path(compiler.attrib.pop("meshdir", "."))
    meshes = output / "meshes"
    meshes.mkdir(exist_ok=True)
    for mesh in tree.findall("asset/mesh[@file]"):
        source = meshdir / mesh.attrib["file"]
        mesh.attrib.setdefault("name", source.stem)
        target = meshes / (digest(source) + source.suffix.lower())
        if not target.is_file():
            shutil.copyfile(source, target)
        mesh.set("file", str(target.relative_to(output)))
    destination.write_text(ET.tostring(tree, encoding="unicode") + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--explorer", type=Path, required=True)
    parser.add_argument("--source-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    prepare(parser.parse_args())
