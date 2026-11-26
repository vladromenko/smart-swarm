# Source and reuse map

Read-only review on Homebrain on 25 November 2026. All three remote main revisions matched the ZIP. Explorer had unrelated dirty README, paper and run files. The simulator modules, simulator model notes, dataset module and training module were checked against HEAD. The SO101 operating checkout was clean at the authorized revision. The offline model preparation uses read-only upstream inputs and a staged copy of Tank CAD for its STL conversion.

| Approved source | Revision | Inspected and reused |
| --- | --- | --- |
| [Explorer 4.0](https://github.com/vladromenko/explorer-4.0/tree/89bba26780619ddec7355da60c7b57a006711daf) | `89bba26780619ddec7355da60c7b57a006711daf` | `models.py`, `manufacturer.py`, `tank_model.py` and `mobile.py` prepare the offline assets once. `fleet.py` provides the MjSpec namespace pattern. `environment.py` and `expert.py` provide the contact teacher, lag and state conventions adapted into the experiment. `factory.py`, `capture.py`, `paired_capture.py`, `datasets.py` and `training.py` inform the narrower recording, grouping and official trainer path. `world_memory.py` is a latest-sighting comparator candidate. |
| [SO101 Studio](https://github.com/vladromenko/so101-lerobot-imitation-learning/tree/5c5c8043668db08601b68f1c0fa11a0c77e3dbc2) | `5c5c8043668db08601b68f1c0fa11a0c77e3dbc2` | `recording.py` validates aligned command/state and each camera's read interval. `training.py` preserves action chunks and complete processors. The platform owns reuse of the native services. |
| [ROSMASTER CubeBot](https://github.com/vladromenko/rosmaster-cubebot/tree/56e02dbab48bb4847ec75707b3a59fc1318ebe82) | `56e02dbab48bb4847ec75707b3a59fc1318ebe82` | The Cube experiment logger writes annotated MJPEG AVI plus trial rows. Treat these as historical diagnostics. |

## Physical dataset candidate

SO101 Studio's `<local Studio dataset ID>` has 50 complete episode records in 10 declared sessions, hardware source and a draft manifest. The documented instruction is to place a blue cube in a white box and withdraw. The camera schema is wrist only. The ordered state/action names are shoulder pan, shoulder lift, elbow flex, wrist flex, wrist roll and gripper. The recorded units are degrees and calibrated gripper percentage. A dataset SHA and calibration records are present. Existing job and export metadata reference the candidate, so it is limited to already-exposed development data until a per-episode outcome and training-overlap audit. Metadata completion is a recording status. No episode was admitted into Smart Swarm training.

## Asset provenance and approximations

The episode manifests record source model, XML and mesh hashes, MuJoCo version, simulated units and contact parameters. SO101 uses Menagerie `370c43965323c0944b433b202aac5058b5f907a9` under Apache-2.0. ROSMASTER uses the Yahboom original URDF meshes and manufacturer SRDF joint/collision mapping. Upstream metadata says Apache-2.0 with inherited attribution. Tank uses the MentorPi T1 CAD mirror and verified input file checksums. The Tank CAD redistribution license was unverified, so the file pack stays local. Both mobile camera mounts used by B1 are task-specific simulation views. ROSMASTER's arm is held in an actuated position, and Tank uses an assumed planar force model.
