# Smart Swarm

Smart Swarm studies whether robots can find a hidden object by combining separate visual histories. In the shared MuJoCo scenario, SO101 observes a placement attempt, Tank observes container relocation, and ROSMASTER chooses a search location or requests inspection. Failed placement and incomplete evidence are part of the task.

## Start with the software and contract checks

Use Python 3.11 or later. From a virtual environment in this checkout, install the package and its test dependency.

```bash
python -m pip install -e '.[test]'
./smart-swarm --help
python -m pytest -q tests/test_contract.py tests/test_memory.py tests/test_evidence_budget.py tests/test_identity_support.py
```

Set `SMART_SWARM_PYTHON` if the launcher should use a different interpreter. The checked-in `contracts/fixture` contains a small synthetic scoped recording for replay and contract tests.

## Configure a simulation or attended recording

Copy `configs/homebrain.example.toml` to `configs/homebrain.toml` and set your own absolute asset, data, and existing service paths. `SMART_SWARM_CONFIG` can point to an alternative configuration file. For training, copy `configs/training.example.json` to `configs/training.json` and configure the local LeRobot runtime and model cache. Full scenario replay uses the prepared SO101, Tank, and ROSMASTER MuJoCo assets. Their preparation and source relationships are documented in the reuse map. Native arm control, cameras, recording, and STOP use the existing native services.

```bash
./smart-swarm doctor
./smart-swarm demo --mode mujoco --family F1
./smart-swarm status
./Smart_Swarm.command
```

The operator guide covers the four-attempt SO101 component pilot and the distinct 24-story three-agent collection. The experiment descriptor template can be filled in with local paths for a Robot Lab menu.

## Research and results

The independent simulation uses 100 source worlds and reports seven methods at three team sizes. The paper describes causal input isolation, the event-only and event-plus-reveal models, and the status of physical verification. The publication includes source code, tests, configuration templates, the synthetic contract fixture, and simulation result tables and figures. The research host retains the complete evaluation bundle, raw recordings, model weights, and local service configuration.

- [Paper draft PDF](docs/PAPER_DRAFT.pdf) · [Markdown](docs/PAPER_DRAFT.md)
- [Independent simulation results](docs/PRIMARY_RESULTS_INDEPENDENT.md)
- [Machine-readable simulation table](evidence/learning-pilot-v3/primary-method-table-test.json)
- [Matched communication controls](docs/PRIMARY_MATCHED_COMPACT_RESULTS.md)
- [Research protocol](docs/RESEARCH_PROTOCOL.md)
- [Claims and limitations](docs/CLAIMS_AND_LIMITATIONS.md)
- [Reuse and model assets](docs/REUSE_MAP.md)
- [Operator guide](docs/OPERATOR_GUIDE.md)
- [Experiment descriptor template](experiment.example.json)
