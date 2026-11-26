# Smart Swarm implementation pass

## Software integration and data validity

Smart Swarm owns integration and research. The standalone `Smart_Swarm.command` launches the pilot on Homebrain. Robot Lab recognized the descriptor when pointed at the absolute Smart Swarm handover directory. The menu needs the entrypoint and working directory.

- A's original fixture service recorded three attempts. The real A exporter produced two complete fixture exports. Smart Swarm independently reopened both with the official LeRobot loader. The interrupted attempt retained its native quarantine reason.
- The deployed SO101 Studio export adapter re-exported two existing technical recordings with six frames. Smart Swarm directly verified the wrist/top RGB, action and state vectors, episode boundaries, task metadata and paired source provenance. The native source manifest hash matched before and after the operation.
- Studio dataset `<local Studio dataset ID>` is prepared for four attempts. It currently contains zero recordings. The [operator guide](OPERATOR_GUIDE.md) covers the native preflight, placement, reset, terminal label and import. The consumer keeps completed failures as candidate story data and interrupted episodes as incomplete.
- Explorer's native archive adapter opened 26 historical Tank and 26 historical ROSMASTER RGB entries through isolated role scopes. The cross-role frame request was denied. The archive has sensor compatibility evidence. Live Tank and ROSMASTER clocks are stale, and the live ROSMASTER observation endpoint returned HTTP 503. Fresh streams and attended preflight are the remaining physical gates.

Native export rehearsal · A fixture rehearsal · Fleet adapter check

## Motor diagnostics

The five sources were the three original training trajectories and two original validation trajectories. The exact recorded simulator configuration and initial snapshots were reconstructed. Both historical and aligned executor profiles achieved bilateral contact, lift, carry, release and stable containment on all five. The historical evaluator sampled the policy input a control interval before command issue. The aligned executor uses the teacher's issue boundary and 30 Hz floor-quantized timing. Its recorded and replayed observation timestamps matched exactly. The precontact state residual was below 0.000007 in the ordered native units. Both first-frame camera byte arrays matched on every reset.

The ACT trace measured zero difference between raw live input and the original LeRobot sample after camera and proprioception preprocessing. The action normalization round-trip error was below 0.000004, chunk order and reset matched exactly, and the decoded command passed through the same simulator conversion and clipping. The very first decoded ACT elbow request was 32.51° versus the teacher's 16.16°. Its gripper request was 37.29% versus 53.76%. Offline native-unit action errors are tabulated separately by joint and stage.

ACT and SmolVLA each failed to reach the approach gate on all five matched, aligned development runs. Closest sampled gripper-to-cube distances were 12.8–13.2 cm for ACT and 6.4–7.7 cm for SmolVLA, with a 5 cm approach gate. Neither policy established a grasp. The ACT historical-timing diagnostic on the first reset also failed at approach. This identifies the earliest observed failure and verifies the common execution path. It leaves the training-side cause unresolved. The original two 0/10 evaluations stay archived as consumed development evidence. Final evaluation requires fresh starts.

Teacher replays · ACT trace · SmolVLA comparison · Execution handoff

## Training jobs

| Pass | Purpose and model | Data | Completed updates | Checkpoint outcome |
| --- | --- | --- | --- | --- |
| Historical | ACT motor pipeline | Three train and two validation MuJoCo trajectories | 8 | Preserved with 0/10 original trials |
| Historical | SmolVLA first CPU attempt | Same motor split | 0 | Interrupted during BF16 fallback |
| Historical | SmolVLA CPU FP32 motor pipeline | Same motor split | 2 | Preserved with 0/10 original trials |
| Current | Temporal evidence with event supervision | Three independent train groups, one calibration group, one held-out pilot group | 120 | Saved and evaluated |
| Current | Temporal evidence with event and reveal supervision | Same grouped pilot split | 120 | Saved and evaluated |

The current evidence job ran both training profiles sequentially under Studio's existing CPU and recording leases. It used the existing local job runner with a 30-minute cap and allocated zero CUDA devices. Current motor optimizer updates totalled zero. The pass manifest includes exact source paths, model configs, processor and checkpoint hashes, and the full evaluations.

## Learned evidence versus the B1 rule reference

B1 placement evidence came from local image rules. Tank matched exterior cues across its local views. ROSMASTER combined the timed factors in a finite joint belief. Scripts controlled the motions, capture windows and communication schedule. Simulator contacts, object identity, and terminal observations stayed in the training and evaluator scopes.

The learned pilot comprised five independently seeded scene groups and 25 matched variants. Fifteen variants have a settled placement label. All 25 have relocation and reveal labels. Intermediate events remained unknown. Both learned variants requested inspection in all five held-out variants. They were invariant on the two F6 alternatives and scored **0/3 on resolvable decisions**. The rule reference scored **3/3** on those controls. Terminal NLL was 1.307 for events alone and 1.238 with reveal supervision on this single held-out scene group. The result leaves the learned competence and calibration gates open.

Across all five F6 pairs, Smart Swarm verified complete admitted history and received messages, including proprioception, timing, query text, camera scope, cold cache and normalized opaque IDs. Changing restricted evaluator truth and training-sidecar contents left the B1 prediction unchanged. Both trained heads give identical probabilities to the held-out F6 pair under the same observations. F3–F5, broader balanced layouts, a fresh final split and physical three-agent results remain study work.

Full-input audit · Counterfactual controls · Evidence training record

## Bounded learning follow-up

The [new learning-pilot report](LEARNING_PILOT_2026-11-26.md) records the versioned issue-time dataset, pre-inspection evidence diagnosis, checked three-agent simulation and current reservation blocker. The [operator guide](OPERATOR_GUIDE.md) contains the live collection entry point and the four attended cards.
