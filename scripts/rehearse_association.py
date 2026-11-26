"""Verify distinct terminal slots in the screened F4 development geometry."""

import argparse
from contextlib import closing
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from smart_swarm.cli import settings
from smart_swarm.episode import atomic
from smart_swarm.rendering import configure_renderer


def screen_pixels(world):
    import mujoco
    import numpy as np

    renderer = world.renderer
    geom_ids = world.model.geom_bodyid
    bins = {world.model.body(name).id for name in ("carrier", "decoy")}
    masks = np.isin(geom_ids, list(bins)).nonzero()[0]
    screen = world.model.geom("association_screen")
    camera = world.cameras["tank-1"]["tank-1/front"]
    hidden = mujoco.MjvOption()
    hidden.geomgroup[screen.group[0]] = False
    renderer.enable_segmentation_rendering()
    try:
        renderer.update_scene(world.data, camera=camera)
        visible = renderer.render()[..., 0].copy()
        renderer.update_scene(world.data, camera=camera, scene_option=hidden)
        unobstructed = renderer.render()[..., 0]
        masked = (visible == screen.id) & np.isin(unobstructed, masks)
        return {"at_ns": world.ns, "screened_bin_pixels": int(masked.sum()),
                "still_visible_bin_pixels": int(np.isin(visible, masks).sum())}
    finally:
        renderer.disable_segmentation_rendering()


def record(assets, root, seed):
    from smart_swarm.episode import LABELS, Replay
    from smart_swarm.evidence_portfolio import labels_for
    from smart_swarm.evidence_rollouts import EvidenceStory, prepare_world, reveal

    class ScreenedStory(EvidenceStory):
        def motion_sample(self):
            super().motion_sample()
            if self.relocation_ticks in (6, 12, 24, 36, 48, 60):
                coverage.append(screen_pixels(self.world))

    path = root / f"f4-{seed}"
    variant = "screened-association-v2"
    world, variation, pickup = prepare_world(assets, seed, "F4", variant=variant)
    coverage = []
    with closing(world):
        story = ScreenedStory(path, "F4", world, seed, False)
        try:
            story.recorder.meta.update(evidence_variant=variant, physical_variation=variation)
            story.scene(pickup)
            query, decision, _ = story.ask()
            finish, terminal, slot = reveal(story, decision)
            if not any(row["screened_bin_pixels"] > 0 and row["still_visible_bin_pixels"] > 0 for row in coverage):
                raise ValueError("The physical screen did not measurably obscure a bin in a Tank frame")
            labels = labels_for(story, terminal, slot)
            story.recorder.append(LABELS, "settled-event", labels)
            story.recorder.meta.update(status="valid", engineering_screen_coverage=coverage)
            story.recorder.close(query, decision, finish)
        except (Exception, KeyboardInterrupt) as error:
            story.recorder.quarantine(str(error) or type(error).__name__)
            raise
    return {"path": str(path), "seed": seed, "terminal": terminal["ground_truth"], "labelled_association": labels["relocation_assignment"],
            "policy_prediction": Replay(path, "rosmaster-1").predict()["decision"], "screen_coverage": coverage}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = settings()
    configure_renderer(config)
    args.output.mkdir(parents=True, exist_ok=False)
    shots = []
    summary = {"scope": "Independent screened association development probe", "status": "running", "scenes": shots}
    path = args.output / "screened-association.json"
    try:
        for seed in (49_008, 49_012):
            shots.append(record(config["asset_root"], args.output, seed))
            atomic(path, summary)
        if {row["terminal"] for row in shots} != {"left", "right"}:
            raise ValueError("The opt-in F4 layouts do not yet cover both real terminal placements")
        summary.update(status="verified", independent_source_groups=len(shots))
    except (Exception, KeyboardInterrupt) as error:
        summary.update(status="incomplete", reason=str(error) or type(error).__name__)
        raise
    finally:
        atomic(path, summary)


if __name__ == "__main__":
    main()
