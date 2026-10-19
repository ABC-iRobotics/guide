# S2b: native motion with OSCBF behind ROS2Robot (design spec)

- **Status:** written 2026-10-19 from the approved design sections, for your review.
- **Roadmap step:** S2b in `docs/superpowers/plans/2026-10-19-libero-isaac-roadmap.md`. Branch `feat/libero-native-motion`.
- **Live tracker:** https://claude.ai/artifact/HC3YKe5CcfYdCf4wbTLVzE (S2b page: the architecture drawing, the comparison, the fact sheet, the measurement report).

## Rulings (user, 2026-10-19 16:03 — these override any earlier text in this document)

1. **Infeasible QP / non-finite output:** apply joint damping **plus gravity compensation** (torque mode), never damping alone.
2. **Reset under `controller: oscbf`:** the host serves GUIDE's Reset by sending a joint-mode target to the robot's home configuration (drive `targetPosition`s are ignored by an effort- or velocity-driven arm).
3. **Speed and velocity profile:** GUIDE-EX's `speed` maps to the planner's cruise speed `v_max = speed × 0.5 m/s`. The streamed targets follow an exponential speed profile:
   - **Approach:** `v(d) = v_max · (1 − e^(−d/λ))` with `d` the remaining distance and `λ = v_max / ω`, where ω = √kp_task (12.2 rad/s for kp 150). The commanded deceleration then matches the controller's critically damped decay.
   - **Start:** a mirrored exponential ramp-up from rest, capped by the robot's acceleration limit.
   - **Arrival:** a 5 mm/s floor, and the exact goal is sent once `d` is below one 50 ms step.
   - **Timeout:** `2 s + 2 × planned duration`.
4. **Coordinates:** every interface a solver, policy or dataset sees is in the **robot's base frame** (`robots.<name>.base_name`, e.g. `fr3_link0`): GUIDE-EX poses, `MotionTarget`, `ROS2Robot`, the PoseRequest results and the recorded states and actions. GUIDE alone converts between that frame and the actual scene/world coordinates, per scene and per robot, so several scenes work at once and datasets are alike whichever scene recorded them. No `/tf` lookup in the solver.
5. **Recording the controller inputs:** a record field in the task's `init.yaml` `dataset` section selects it (e.g. `action: controller_input`). The recorder then stores the `MotionTarget` handed to the controller at 20 Hz (delta action, gripper width, absolute target) instead of the joint command.
6. **Strict version compatibility:** `uv pip check` must report **zero** lines; no `--override`. If lerobot's caps conflict with Isaac's exact pins, find a lerobot release that resolves cleanly with Isaac 6.0.1's pins (or report BLOCKED with the exact conflict). Never weaken a pin.
7. **OSCBF is a git submodule, not vendored** (user, 2026-10-19): `guide_motion/external/oscbf` at 6082329 (github.com/StanfordASL/oscbf, MIT), reached through the symlink `guide_motion/oscbf` -> `external/oscbf/oscbf`. Its code never enters the guide repo's history (the unpushed branch was rewritten). Not a pip dependency: OSCBF's own metadata pins `cbfpy==0.0.1` (numpy<2), pybullet and matplotlib.

## 1. Goal

Replace MoveIt as GUIDE's motion layer with:
- a planner (Pink IK first);
- OSCBF: operational space control with control-barrier-function safety, solved as one small QP per step.

The result must serve two purposes:
1. drive scripted GUIDE-EX demonstrations;
2. execute LIBERO-style 20 Hz end-effector actions.

It must work identically in Isaac Sim and, later, on the real FR3. Torque control is used wherever the robot and the simulation allow it.

**In scope:**
- the `guide_motion` package;
- the robot description pipeline;
- the Isaac host adapter;
- `MotionTarget.msg`;
- the `ROS2Robot` OSCBF mode;
- the GUIDE-EX seam cleanup and sim-time waits;
- the Panda and FR3 in simulation.

**Out of scope here:**
- the real FR3 host (component 7, its own spec once the sim path works);
- UR5e;
- cuRobo;
- JAX on GPU;
- removing MoveIt. It stays as `controller: moveit` until the OSCBF mode is proven.

## 2. Decisions this spec rests on (user, 2026-10-19)

| Decision | Source |
|---|---|
| ROS 2 stays the transport between processes; the solver is Isaac-free | tracker answer |
| The system goes behind `ROS2Robot.send_action` | chat |
| Torque wherever possible; torque-capable robots first | tracker answer |
| USD is the single robot description (visual, collision, planning); joint-limit narrowing vs franka_description is not critical | tracker answers |
| The architecture: torque inner loop co-located with the physics | tracker answer ("Architecture is fine.") |
| `guide_motion` lives in the guide repo | tracker answer |
| Round 1 on both Panda and FR3, FR3 preferred | tracker answer |
| Save as much RTF as possible | chat |
| OSCBF adopted as the control layer | chat |

## 3. Architecture (approved)

There are two processes, and ROS 2 joins them.

- **Solver process (Isaac-free).** GUIDE-EX or a LIBERO policy calls `ROS2Robot.send_action`. That uses the `guide_motion` planner and streams `MotionTarget`s at 20 Hz on sim time.
- **Isaac process.** Each 60 Hz rendered frame runs two 120 Hz physics substeps. Before each substep, a `PHYSICS_PRE_STEP` callback (the host adapter in `guide_core`) reads the latest target and calls the `guide_motion` controller. It then writes joint efforts (torque mode) or velocity targets (velocity mode) to the articulation.
- **States** return over the existing per-frame `/joint_states`.
- **No ROS message crosses per substep,** and nothing is rendered for control.
- **On the real FR3,** the same `guide_motion` controller runs inside franky/libfranka's callback (later spec).

The tracker's S2b page has the drawing.

## 4. Components

### 4.1 `guide_motion` (new top-level pure-Python package in the guide repo)

No Isaac imports, no ROS imports. It is a plain pyproject package installed into `.venv`, and later vla-inference-ui's env. Modules:
- **`model`:** loads URDF + XRDF and exposes kinematics and dynamics (via OSCBF's `Manipulator.from_urdf`) plus the collision spheres. `set_mass_properties(masses, coms, inertias)` overrides the URDF inertials with the simulator's values (see 4.2).
- **`control`:** `Controller(model, mode="torque" | "velocity", gains, barriers)`.
  - `warm_up()`: compiles the JAX functions.
  - `step(target, q, qd) -> command`: returns torques (torque mode) or joint velocities (velocity mode).
  - Barriers: joint limits, self-collision from the XRDF spheres, singularity, and an end-effector workspace box. Obstacles are added as spheres or half-spaces when a task needs them.
- **`targets`:** the `Target` dataclass (mode, pose, twist, joints, gripper width).
  - **Actions are metric everywhere in GUIDE** (user decision 2026-10-19): Δp in metres, Δω as a rotation vector in radians, gripper width in metres. LeRobot normalizes the recorded data for training, so no [−1, 1] interval is used in recording, the controller or the target topic.
  - `delta_action_to_target(action, measured_pose)` reproduces robosuite `OSC_POSE`'s goal semantics in metric units:
    1. clamp Δp to ±0.05 m and Δω to ±0.5 rad per axis (LIBERO's per-step bound, kept as a safety limit);
    2. apply it as a delta from the **measured** end-effector pose at the policy step (`goal_pos = p + Δp`, `goal_ori = R(Δω)·R`), not from the previous target;
    3. hold the result for one 50 ms step.
  - `from_libero_normalized(action)` / `to_libero_normalized(action)` convert to and from LIBERO's normalized convention (×0.05 m, ×0.5 rad, gripper ±1 ↔ width). They are used **only** at the 2×2 evaluation boundary, chosen by a per-checkpoint flag: when a LIBERO-trained policy runs in GUIDE, or a GUIDE-trained policy runs in MuJoCo LIBERO.
  - **Recorded action = the controller's input exactly as given** (user decision 2026-10-19). At each 20 Hz step the recorder stores what `MotionTarget` handed the controller:
    - the delta action in metres and radians;
    - the gripper width;
    - the absolute target pose alongside.

    It never stores the pose change measured between frames, which trails the command by the tracking lag (≈ 15 mm at 10 cm/s).
  - **Recording rate:** LIBERO tasks record frames and actions at the **same 20 Hz** as the action rate (`dataset.fps: 20`). GUIDE's tasks record at 10 Hz today.
  - The recorder change belongs to S6/S7, which own the dataset schema; this spec fixes what they must record.
- **`planner`:** `plan(goal, q) -> list[Target]` with Pink IK, used for moves larger than one target step. It runs only on new goals.
- **`description`:** the pipeline in 4.2, run as a CLI.

**Dependencies (exact):** `jax==0.4.30`, `jaxlib==0.4.30`, `cbfpy==0.1.0`, `qpax==0.1.4`, `ml-dtypes`, `opt-einsum`, `pin` / `pin-pink` (already in `.venv`). OSCBF itself is vendored as a pinned dependency at its MIT-licensed upstream commit.

- Installed with the repo's pins file (`-c $PINS`).
- **Prerequisite:** the `.venv` must first be brought back onto Isaac's pins. Today it has numpy 2.2.6 and torch 2.10.0 against 2.3.1 / 2.11.0, with 9 `uv pip check` incompatibilities.

### 4.2 Robot description pipeline (`guide_motion.description`, run once per robot)

1. **USD → URDF** with Isaac's `isaacsim.asset.exporter.urdf`, timeline stopped. Then post-process:
   - mimic multiplier → +1;
   - prismatic finger velocity → the USD's value without the degree conversion;
   - add `effort` limits from the USD drives;
   - keep the USD's joint limits.
2. **XRDF** from `isaacsim.robot_setup.xrdf_editor`'s headless classes. Then:
   - add `tool_frames: [<robot>_hand_tcp]`;
   - add the ignore line `<robot>_link7: [<robot>_hand, <robot>_leftfinger, <robot>_rightfinger]`;
   - set acceleration and jerk limits:
     - FR3: 10 rad/s² and 5000 rad/s³ on every joint (franka_description, Panda vs FR3 rundown);
     - Panda: [15, 7.5, 10, 12.5, 15, 20, 20] and [7500, 3750, 5000, 6250, 7500, 10000, 10000] (Isaac's shipped Panda XRDF);
   - keep link0 self-collision-only;
   - re-express every sphere centre in the exported URDF's link frames (fixed per-link transform at q = 0).
3. **Mass properties must match the simulation.** Torque control was unstable with the Panda asset as shipped: its authored inertias are about 1e-6 to 1e-8 kg·m², so the wrist chattered at ±12 Nm.
   - **Panda:** the pipeline writes franka_description's link inertials and centres of mass into a USD override layer that the scene references. The sim becomes physically plausible and matches the controller.
   - **FR3:** the asset authors masses and centres of mass but no inertia; PhysX derives inertia from the colliders.
   - **Both robots:** at registration, the host reads the articulation's actual masses, centres of mass and inertias from PhysX and passes them to `model.set_mass_properties()`. The controller's M and g are then the simulator's.
4. **Outputs** (URDF, XRDF, the override layer and a `description.json` with the source asset path and pipeline version) are committed under `guide_motion/robots/<robot>/`.
5. **Each run validates itself and fails loudly if a check fails:**
   - cuMotion's loader accepts the pair;
   - the home pose is free of self-collision;
   - random-configuration self-collision ≤ 8% (Panda reference 5.3%; FR3 with fixes 4.5–6.6%);
   - M and g from `guide_motion` agree with PhysX within 1e-3 relative.

### 4.3 Sim host adapter (`guide_core/guide_core/control/oscbf_host.py`, about 100 lines, Isaac-bound)

- **Configuration:** enabled per robot in the task's `init.yaml` (`robots.<name>.controller: oscbf`, with `mode`, `gains` and `rate`).
- **At registration:**
  - load the robot's description;
  - read the mass properties (4.2 step 3);
  - switch the 7 arm DOFs' control mode: effort (stiffness and damping 0) for torque, velocity for velocity mode; the gripper stays position-driven;
  - set up the JAX persistent compilation cache (`~/.cache/guide_motion/jax`);
  - call `Controller.warm_up()`. Without a warm-up, the first physics step stalls 4.2 s (torque) or 1.6 s (velocity).
- **Each physics substep** (`SimulationManager.register_callback(..., PHYSICS_PRE_STEP)`):
  - read q and qd;
  - read the latest target from a single-slot holder;
  - call `step`;
  - write efforts or velocity targets.
- **Rates:**
  - torque mode runs every substep (120 Hz);
  - velocity mode runs every second substep (once per rendered frame), measured to track just as well (0.7 mm path deviation) at half the cost.
- **I/O goes through the articulation's tensor view.** It saves 0.2 ms per call against the experimental API's getters. If the private attribute is missing in a future Isaac, it falls back to the API automatically.
- **Targets arrive on GUIDE's existing ROS thread:**
  - subscriber depth 1, `KEEP_LAST`;
  - the slot is written with one lock-protected assignment.

  This avoids the OmniGraph subscriber's oldest-of-10 queue, measured at 110 ms of staleness for a 100 Hz publisher.

### 4.4 `MotionTarget.msg` (`guide_msgs`)

```
builtin_interfaces/Time stamp          # sim time
uint8 MODE_POSE=0
uint8 MODE_TWIST=1
uint8 MODE_JOINTS=2
uint8 mode
string frame_id                        # the robot base link
geometry_msgs/Pose pose
geometry_msgs/Twist twist
float64[] joints
float64 gripper_width                  # metres, absolute
```

The topic is `/Sim_0/Scene_<i>/<robot>/motion_target`, published at 20 Hz.

### 4.5 `ROS2Robot` OSCBF mode (`irob_lerobot_ros`)

- **Configuration:** `controller: oscbf | moveit` (default `moveit` until S2b is proven).
- **`send_action(Pose | Twist | joints, wait_for_execution)`:**
  - a pose more than one step away is planned with `guide_motion.planner`, and the resulting targets are streamed at 20 Hz on sim time;
  - completion: end-effector position error < 5 mm, orientation error < 2°, and speed < 1 cm/s, held for 0.1 s of sim time;
  - otherwise it times out after `2 s + 2 × path length / 0.1 m/s` of sim time;
  - returns True or False.
- **Gripper:** set through `gripper_width` on the same topic.
- **Observation features:** fix the unreachable gripper features (`ros2robot.py:239-240`).

### 4.6 GUIDE-EX seam (`guide_ex`)

- `MoveToCartesianPose` and `MoveToJointConfiguration` stop reaching into `robot._moveit2` (`cartesian_move.py:55,62-68`, `joint_move.py:46`). They use `send_action` results only.
- `WaitForSeconds`, `SetGripperState`'s settle and `solve_task`'s post-Randomize pause wait on **sim time** (the `/clock`-based sleep that block_bin_eval's `sleep_sim` already uses).

## 5. Control modes and gains (measured)

| Mode | Where used | Rate | Gains | Measured, model matched (Panda, light scene) |
|---|---|---|---|---|
| Torque (default where the model matches) | FR3, and the Panda with the inertial override | every substep, 120 Hz | LIBERO-like (the law of robosuite `OSC_POSE` as LIBERO configures it): task kp 150 and kd 2√150 ≈ 24.5 on all six axes (ζ = 1); nullspace kp 10, kd 2√10 ≈ 6.32 towards the joint configuration at the episode start. Known differences: 120 Hz vs 500 Hz recomputation, coupled vs uncoupled Λ, PhysX vs MuJoCo. Details: tracker report "What LIBERO-like gains means". | hold 0.03 mm; 20 Hz square path deviation 1.7 mm mean, 4.6 mm peak; lag 15 mm behind the held target |
| Velocity (fallback) | a robot whose mass properties are not trusted; UR5e later | once per frame, 60 Hz | OSCBF velocity gains | hold 0.09 mm; path deviation 0.7 mm mean, 2.0 mm peak |

240 Hz physics was **not** needed for stability once the model matched (0.07 → 0.01 mm hold), and it costs ≈ 2× the callback time plus 1 ms of physics per frame. Physics stays at 120 Hz.

## 6. RTF budget (your constraint)

Measured added time per rendered frame, against an awake-arm baseline:

| Config | Added ms per frame |
|---|---|
| Torque 120 Hz | +2.10 (+1.67 with the tensor view) |
| Velocity once per frame | +0.96 |

On GUIDE's ~31.5 ms frame this is an estimated ≈ −5% RTF for torque and ≈ −3% for velocity. Rules:
- no per-substep ROS traffic;
- the planner runs only on new goals;
- JIT warm-up and persistent cache, so there are no stalls mid-episode;
- physics stays at 120 Hz;
- an RTF regression check is part of the tests (7).

The OSCBF call is 2.6× slower inside Isaac than standalone (thread contention). This is recorded, not chased.

## 7. Testing

**Unit tests (`guide_motion/test`, no Isaac):**
- kinematics and dynamics against Pinocchio on the same URDF;
- the controller step against recorded inputs (deterministic outputs);
- `delta_action_to_target` against actions and end-effector poses from `~/Downloads/libero_spatial` HDF5 demos (via `from_libero_normalized`);
- the description post-processor against the exporter defects found by the XRDF test;
- `delta_action_to_target` and the LIBERO-normalized converters round-trip on real LIBERO HDF5 actions, and the clamp holds at ±0.05 m / ±0.5 rad;
- the target holder (latest value wins, stale detection).

**Headless Isaac integration (`guide_core/test`, run manually):**
- Panda and FR3:
  - hold the pose for 5 s (≤ 0.1 mm);
  - the 20 Hz square (path deviation ≤ 3 mm torque, ≤ 2 mm velocity);
  - no NaN, torque ≤ limits.
- RTF: the frame time at 120 Hz with the controller ≤ awake baseline + 2.5 ms (torque) and + 1.2 ms (velocity).
- Same seed twice gives the same joint trajectory, max difference ≤ 1e-6 rad.

**`ROS2Robot` OSCBF mode with GUIDE:**
- block_bin with `controller: oscbf` generates 5 episodes, all successful;
- recorded frames show no wall-clock waits.

## 8. Error handling

| Situation | Behaviour |
|---|---|
| No target for 0.25 s of sim time | hold the current pose (torque: gravity compensation + hold; velocity: zero velocity) and set a stale flag in the host's diagnostics |
| QP infeasible or non-finite output | apply joint damping only, count the event, keep running |
| Command outside limits | clamp to the URDF effort and velocity limits |
| Description validation fails | refuse to register the robot, with the failing check named |
| `send_action` timeout or planner failure | returns False; GUIDE-EX retries or falls back as today |

## 9. Risks and open items

- **Unexplained ~73 ms frame spikes** occur with or without the controller (pre-existing; measured spread ±3% RTF).
- **FR3 mass properties are derived by PhysX from the colliders.** Check them against the datasheet masses once, during the pipeline's validation.
- **Isaac APIs:**
  - the tensor-view attribute is private (hence the fallback);
  - GUIDE still uses `isaacsim.core.api`, deprecated since 6.0.0. The host adapter uses the non-deprecated `SimulationManager` and experimental `Articulation`.
- **The `.venv` drift must be fixed first** (4.1 prerequisite).
