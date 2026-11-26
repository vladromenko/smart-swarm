"""Show preselected native frames for an optional attended outcome review."""

from tempfile import TemporaryDirectory
from pathlib import Path
import webbrowser

from PIL import Image

from .episode import sha


def reviewed_images(root, frames):
    root = root.resolve()
    for frame in frames:
        path = (root / frame["frame_ref"]).resolve()
        if not path.is_relative_to(root) or sha(path) != frame["sha256"]:
            raise ValueError("A saved review image changed or is outside its original story")
        with Image.open(path) as image:
            yield image.convert("RGB")


def open_review(root, frames, destination):
    images = list(reviewed_images(root, frames))
    sheet = Image.new("RGB", (sum(image.width for image in images), max(image.height for image in images)))
    x = 0
    for image in images:
        sheet.paste(image, (x, 0))
        x += image.width
    sheet.save(destination)
    if not webbrowser.open(destination.as_uri(), new=2):
        print(f"Open the preselected images together at {destination}")


def confirm(event, item):
    choices = {str(index): value for index, value in enumerate(item["allowed_labels"], 1)}
    print(f"\n{event.replace('_', ' ').capitalize()}")
    for key, value in choices.items():
        print(f"  {key}  {value.replace('_', ' ')}")
    selected = input("Enter the visible outcome number, or leave it unknown with Enter: ").strip()
    if not selected:
        return None
    if selected not in choices:
        raise ValueError("Choose a listed outcome number or leave the event unknown")
    assignment = event == "relocation_assignment"
    if assignment and input("Could you verify which station is left from ROSMASTER's view? [y/N] ").lower().strip() != "y":
        return None
    basis = input("Briefly describe the visible evidence, or press Enter to leave it unknown: ").strip()
    if not basis:
        return None
    record = {"value": choices[selected], "basis": basis, "directly_observed": True,
              "frame_ids": [frame["frame_id"] for frame in item["review_images"]]}
    if assignment:
        record["rosmaster_left_reference_verified"] = True
    return record


def review(root, available, stored):
    fields = {}
    pending = {name: item for name, item in available.items() if stored[name]["status"] == "needs_confirmation"
               and item["review_images"]}
    if not pending:
        print("No unresolved outcomes have scoped review images.")
    with TemporaryDirectory(prefix="smart-swarm-outcome-review-") as temp:
        for event, item in pending.items():
            open_review(root, item["review_images"], Path(temp) / f"{event}.png")
            observation = confirm(event, item)
            if observation is not None:
                fields[event] = observation
    return fields
