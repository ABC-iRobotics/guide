# S2b Plan C — `ROS2Robot` `controller: oscbf`, the GUIDE-EX seam, sim-time waits, block_bin on OSCBF

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** GUIDE-EX's steps drive a robot only through `ROS2Robot.send_action`. With `controller: oscbf`, ROS2Robot plans with `guide_motion`, streams `MotionTarget`s at 20 Hz of sim time to Plan B's host, and reports completion or a sim-time timeout. All GUIDE-EX waits run on sim time. block_bin generates 5 successful episodes this way.

**Architecture:** `irob_lerobot_ros` (its own git repo, `~/ros2_ws/src/guide/modules/irob_lerobot_ros`) gains two `ROS2RobotConfig` fields (`controller`, `motion_robot`), a sim-clock `sleep`, the fixed observation features, and an oscbf mode: `init_oscbf` (model, publisher), one streaming thread, `_send_oscbf` (pose in the base frame / twist / joints / gripper → `guide_motion.planner.plan(..., speed)` → queue), `_wait_reached` (the spec's completion test on `/joint_states` and FK), and `hold_off` / `reset_scene` for GUIDE's Reset. MoveIt's speed and straight-line check move from the GUIDE-EX steps into `send_action` keyword arguments, so `guide_ex` never touches `robot._moveit2`. GUIDE answers PoseRequests in the requesting robot's base frame (Task 3b), so no solver looks up `/tf`. block_bin's launch reads the same `robots.franka.controller` switch GUIDE reads (Plan B), so a single edit in `init.yaml` switches both processes.

**Tech Stack:** ROS 2 Jazzy rclpy, lerobot 0.6.1's `Robot`, `guide_motion` (Plan A), `guide_msgs/MotionTarget` and the OSCBF host (Plan B), numpy/scipy, pytest 9 with fake nodes and clocks.

**Spec:** `docs/superpowers/specs/2026-10-19-s2b-native-motion-design.md`, sections 4.5, 4.6, 7 ("`ROS2Robot` OSCBF mode with GUIDE") and 8 (row "`send_action` timeout or planner failure").

**Depends on:** Plan A (`guide_motion` installed; `robots/fr3` committed) and Plan B (`guide_msgs/MotionTarget` built; the host wired to `init.yaml`) complete.

**Sources:** `ROS2Robot` and its config (`modules/irob_lerobot_ros/irob_lerobot_ros/{ros2robot.py,config.py}`); the steps (`guide_ex/guide_ex/steps/manipulation/{cartesian_move.py,joint_move.py}`, `steps/end_effector/gripper_control.py`, `utility/wait.py`); block_bin (`guide_tasks/block_bin/block_bin/solve_task.py`, `launch/bringup.launch.py`); block_bin_eval's `sleep_sim` (`~/ros2_ws/src/block_bin_eval/block_bin_eval/eval_policy.py:622-642`), the `/clock` sleep spec 4.6 names; the fact sheet `/home/user/libero_assets/s2b_inputs/s2b-facts.md` §2 and §6.

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
8. **lerobot 0.6.1** (user, 2026-10-19): `.venv` upgraded in place from 0.6.0 (only draccus 0.10.0 -> 0.11.6 moved; guide_core + guide_ex 161 tests and block_bin_eval 351 tests pass). 0.6.1 carries the same `numpy<2.3` / `packaging<26` caps as 0.6.0, so Task 1's strict re-pin still waits on the single-env question.
9. **Single strict env with a rebuilt lerobot** (user, 2026-10-19): lerobot is a git submodule, `modules/lerobot` (already declared in `.gitmodules`), pinned at v0.6.1 (7e241bd). `.venv` installs a wheel built from a copy of it with exactly two `pyproject.toml` lines changed (`numpy>=2.0.0,<2.4.0`, `packaging>=24.2,<27.0`) and the version `0.6.1+guide.1`. `uv pip install --no-sources` keeps lerobot's own cu128 torch index out. The submodule is never edited. INSTALLATION.md documents the method step by step, with a check that fails if the upstream lines differ. Pins: `huggingface-hub==1.16.1` (1.24 needs click>=8.4.2; Isaac pins click==8.1.7) and `fsspec==2026.2.0` join `$PINS`. Gate: `uv pip check` reports zero lines.
10. **Correct the FR3's mass properties** (user, 2026-10-19): the description pipeline's mass override sets `fr3_link8` and `fr3_hand_tcp` (frames on the real robot, 1.0 kg bodies in Isaac 6.0's asset) to 1 g with a negligible inertia. `fr3_hand` and both fingers take franka_description's mass, COM and inertia. link0 is left as is (fixed base). The override layer is committed as `robots/fr3/fr3.usda`, the self-checks re-run against the corrected PhysX model, and GUIDE's scenes must load that layer (Plan B).

**Controller rulings (2026-10-20, from Plan A's final review; they override the text below, and the tasks they touch are rewritten before Plan C is dispatched):**

- **Speed (Ruling 3; supersedes spec question 2 for oscbf):** `send_action(..., speed=)` passes `speed` to `plan(model, goal, q, speed=...)`, so `v_max = speed × 0.5 m/s`. `speed=None` means 1.0, as for MoveIt (no scaling) and GUIDE-EX's step default. The timeout stays 2 s + 2 × planned duration (spec question 1).
- **Frames (Ruling 4; supersedes spec question 3):** no `/tf` lookup in the solver. Poses reaching ROS2Robot are in the robot's base frame; a pose in another frame is refused (False plus an error). GUIDE converts: GUIDE-EX configs use the base link as `frame_id`, and GUIDE's pose requests return poses in the requesting robot's base frame, per scene. Plan C gains the task for that GUIDE-side conversion.
- **Reset (Ruling 2, from Plan B's final review I2):** the host's `home()` holds until the next fresh target, and `_stream` re-sends `_last` every 50 ms, so a streaming solver would undo a Reset within 50 ms. ROS2Robot gains `hold_off()`, which clears the queue and `_last` and stops streaming. Every Reset path (GUIDE-EX's reset step and any Reset ROS2Robot triggers) calls it first, then waits on sim time, with a timeout, until the arm is within the reach tolerance of the robot's `default_joint_states` (the pose the host's Reset homes to; it equals `Model.home` for block_bin only). Streaming resumes with the next `send_action`. Add one test each for the order and the wait.
- **Logs (Plan B final review I3):** the host logs QP infeasibility as warnings (at counts 1, 2, 4, …) and every refusal line contains "refused", so Task 5's `grep -c "infeasible\|refused"` counts real events.
- **gripper_width (Plan B FIX B1):** `MotionTarget.gripper_width` defaults to 0.0, which closes the hand (rosidl rejects a NaN default). Every MotionTarget the solver publishes sets it explicitly: NaN to leave the gripper, the width otherwise.
- **lerobot:** irob_lerobot_ros's `feat/oscbf-mode` branches from `chore/lerobot-0.6.1` (its lerobot submodule at v0.6.1, matching GUIDE's `modules/lerobot`).
- **Deferred from Plan A:** `Model.ee_speed` is exercised by `_wait_reached`'s tests; if `cartesian=True` must stay on the line (`min_fraction`), check the planned targets' line deviation.

**Pre-flight against Plans A and B as built (2026-10-20; ledger `.superpowers/sdd/2026-10-19-s2b-plan-c-rosrobot-guide-ex/progress.md`):**

- **Speed.** Measured on Plan A's planner from the FR3 home pose: a 0.2 m move plans 19 targets at speed 1.0 (first step 6.83 mm) and 48 at speed 0.2; a 0.1 m move plans 15 (first step 4.60 mm). Task 2's tests use these numbers.
- **Frames.** block_bin's `Scene_0` is not `fr3_link0`. `assets/block_bin.usd` places `/Scene/fr3` at (−0.3, 0, 0), and MoveIt mounts the arm there too (`launch/bringup.launch.py`: `connected_to: Scene_<i>`, `xyz: -0.3 0 0`). Physically, `/Scene/fr3/root_joint` fixes the robot to `/Scene/simple_room/table_low_327` (localPos0 (0, −0.2, 0.6692), +90° about z, in a room turned −90° about z: no net rotation); an older measurement put the base at (−0.30669, −0.00004, 0.01307) in the scene (block_bin_eval `eval_policy_pink.py:188`; that measurement's FK of home, (0.30689, 0, 0.48688), equals Plan A's). block_bin's datasets of 2026-10-10 (base-frame recorder, MoveIt) end at (0.300, 0.000, 0.500) and start at (0.300, 0.000, 0.400) in `fr3_link0`, equal to Plan A's FK of their recorded joints. Task 3b therefore puts the rest pose at (0.3, 0, 0.5) in `fr3_link0`, unchanged for the arm. Cube and bin poses come from GUIDE's live base pose; Task 5 Step 2 records that offset.
- **Reset.** GUIDE-EX has no reset step, and nothing calls `/Sim_0/Reset` today (block_bin creates a `robot.reset` client and never calls it). `ROS2Robot.reset_scene` (Task 2) is the one Reset path: `hold_off()`, the service, then `_wait_reached` on the JOINTS goal `home` (the reach tolerance) within 2 s + 2 × the host's homing plan.
- **Line deviation (deferred item).** Not added as a runtime check: every POSE plan is the straight line or a `PlanningError`. Measured deviation from the segment: at most 0.003 mm over 40 block_bin-like approaches and the test moves.
- **Ruling 12.** Plan C's waits and timeouts are on sim time, and `/joint_states` still arrive once per rendered frame, so the 60 Hz torque controller changes no rate in Plan C. It changes only the expected `attached:` log line (`controller every 2 substep(s)`) and the frame time (+2.47 ms, Plan B's committed measurement).

## Spec questions (recorded; the plan follows the reading shown)

1. **Timeout `2 s + 2 × path length / 0.1 m/s`.** Ruling 3 restates it as 2 s + 2 × planned duration, the planner now cruising at speed × 0.5 m/s. Reading: timeout = 2 s + 2 × (number of planned targets × 50 ms). This gives a rotation-only move (path length 0) time to finish too.
2. **GUIDE-EX's `speed` and `min_fraction`.** *Superseded for oscbf by the controller ruling "Speed":* both become `send_action(..., speed=, min_fraction=)` keyword arguments; the moveit branch applies them exactly as the steps did before, and `controller: oscbf` passes `speed` to `plan(..., speed=)` (`None` = 1.0) and needs no `min_fraction` (its POSE plans are straight lines or a `PlanningError`).
3. **Frames.** *Superseded by Ruling 4 and the controller ruling "Frames":* ROS2Robot never looks up `/tf`; it refuses a pose whose `config.frame_id` is not the base link, and GUIDE answers PoseRequests in the requesting robot's base frame (Task 3b).
4. **Twist in oscbf mode** (`CARTESIAN_VELOCITY`, which GUIDE-EX's `MoveWithCartesianVelocity` sends once, untimed). Reading: a one-shot `MODE_TWIST` target. The host follows it for 0.25 s, then holds the pose (spec 8), the closest match to MoveIt Servo's command timeout.
5. **"Recorded frames show no wall-clock waits."** Reading: checked on the dataset. Every block_bin episode ends with Return's `WaitForSeconds(5.0)`. At the dataset's 10 Hz that is at least 48 still frames when the wait runs on sim time; the datasets recorded so far, with the wall-clock wait at RTF ≈ 0.15–0.2, end 6–11 still frames after the last move.
6. **The dataset's `action` columns** (`joint_command.*`, the articulation's applied position targets) mean nothing for an effort- or velocity-driven arm. The recorded action is S6/S7's (spec 4.1) and is not changed here; the run judges success and waits only.
7. **Gripper.** GUIDE-EX's `gripper_goal_pos` is one finger's travel (`0.04` open, `0.01` squeezing a cube). Reading: `gripper_width` = 2 × that value on the target topic. Plan B's host halves it again for the finger joint.

## Global Constraints

Copied from the spec:
- "Configuration: `controller: oscbf | moveit` (default `moveit` until S2b is proven)."
- "`send_action(Pose | Twist | joints, wait_for_execution)`: a pose more than one step away is planned with `guide_motion.planner`, and the resulting targets are streamed at 20 Hz on sim time; completion: end-effector position error < 5 mm, orientation error < 2°, and speed < 1 cm/s, held for 0.1 s of sim time; otherwise it times out after `2 s + 2 × path length / 0.1 m/s` of sim time; returns True or False."
- "Gripper: set through `gripper_width` on the same topic." "Observation features: fix the unreachable gripper features (`ros2robot.py:239-240`)."
- "`MoveToCartesianPose` and `MoveToJointConfiguration` stop reaching into `robot._moveit2` (`cartesian_move.py:55,62-68`, `joint_move.py:46`). They use `send_action` results only."
- "`WaitForSeconds`, `SetGripperState`'s settle and `solve_task`'s post-Randomize pause wait on **sim time** (the `/clock`-based sleep that block_bin_eval's `sleep_sim` already uses)."
- "`send_action` timeout or planner failure: returns False; GUIDE-EX retries or falls back as today."
- Section 7: "block_bin with `controller: oscbf` generates 5 episodes, all successful; recorded frames show no wall-clock waits."
- "The topic is `/Sim_0/Scene_<i>/<robot>/motion_target`, published at 20 Hz." Actions metric everywhere.

Process (this workspace):
- Two repositories. `irob_lerobot_ros` (`~/ros2_ws/src/guide/modules/irob_lerobot_ros`, checked out on `main`) gets the branch `feat/oscbf-mode`, created from `chore/lerobot-0.6.1` (6ce7461, its lerobot submodule at v0.6.1) in Task 1, Step 0; its commits go there. Everything else is committed in `~/ros2_ws/src/guide` on `feat/libero-native-motion`. Commits are authored by each repo's configured user only: **no `Co-Authored-By`, no Claude/Generated-with lines.** Never push, never open PRs.
- Never stage `guide_core/config/init.yaml`; never run `git checkout --`, `git restore`, `git reset --hard`, `git stash` or `git clean`. Stage files by name.
- `irob_lerobot_ros`, `guide_msgs`, `guide_core`, `guide_ex` and `block_bin` are colcon copy installs: after editing, `cd ~/ros2_ws && source /opt/ros/jazzy/setup.zsh && colcon build --packages-select <pkg> && source install/setup.zsh` (zsh). A `.srv` change also needs a GUIDE restart (the type hash changes).
- Tests from a sourced shell with `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 ~/ros2_ws/.venv/bin/python -m pytest -q …`, ignoring `*test_flake8.py`, `*test_pep257.py`, `*test_copyright.py` (flake8 is not in the venv). `guide_ex`'s tests import the **installed** `guide_ex` (no conftest puts the source first): build it before a GREEN run. `guide_core`'s and `irob_lerobot_ros`'s tests import their source.
- Isaac/GUIDE: one simulator per machine; `pgrep -af "kit/kit|guide_core|solve_task"` must print nothing before a launch; `export CYCLONEDDS_URI=file://$HOME/ros2_ws/install/guide_core/share/guide_core/config/cyclonedds_localhost.xml` in every shell that talks to GUIDE. Datasets go to `~/dataset/<task>/` (here `~/dataset/block_bin`). Never upload to Hugging Face.

## Review Focus

1. **Isaac's clock restarts during a wait** (a `RegisterScene` stops and plays the timeline, and the sim time starts again from 0). Expected: the wait resyncs and still times out after `timeout` s of sim time instead of waiting forever. Pinned in Task 2 (`test_a_clock_restart_during_a_wait_still_times_out`).
2. **A new goal while the previous plan still streams** (`wait_for_execution=False`, then another `send_action`). Expected: the queue is replaced, and the new plan starts from the measured joint state, not from the old plan's end. Pinned in Task 2 (`test_a_new_goal_replaces_the_queue_and_starts_from_the_measured_state`).
3. **A target pose outside the base frame** (Ruling 4): the solver must not transform it, and GUIDE must. Expected: ROS2Robot refuses a pose whose `config.frame_id` is not the base link (False, an error, nothing queued); GUIDE answers a PoseRequest by `T_base⁻¹ · T_prim` with the base found in the requested prim's own scene, so a sign or order error would mirror a turned base's answer. Pinned in Task 2 (`test_a_pose_outside_the_base_frame_is_refused`) and Task 3b (`test_a_pose_request_answers_in_the_requesting_robots_base_frame`: a base on Scene_1 turned 90°); Task 5 Step 2 checks block_bin's offset live.
4. **A joint action naming only some arm joints.** Expected: refused with a warning, nothing queued, `False`. Pinned in Task 2 (`test_a_partial_joint_goal_is_refused`).
5. **The moveit path after the seam moved:** a straight line MoveIt can finish only partly must still be refused before the arm moves. Expected: `send_action(cartesian=True, min_fraction=0.95)` returns False without `move_to_pose`. Pinned in Task 1 (`test_moveit_refuses_a_cartesian_path_it_cannot_finish`); Task 5 also reruns block_bin on MoveIt.
6. **Reset while a goal streams** (Plan B final review I2/N1): the 20 Hz repeat of the last goal is a fresh target, which replaces the host's homing within 50 ms. Expected: `reset_scene` empties the queue and drops the last target before it calls the service, streams nothing afterwards, and waits on sim time for `home` with a timeout. Pinned in Task 2 (`test_reset_holds_off_the_stream_before_it_calls_the_service`, `test_reset_waits_on_sim_time_for_the_home_configuration`).

---

## File Structure

```
modules/irob_lerobot_ros/ (own repo)
  irob_lerobot_ros/config.py        controller, motion_robot (FR3: "fr3")
  irob_lerobot_ros/ros2robot.py     sim-clock sleep; observation fix; send_action(speed, min_fraction);
                                    oscbf mode: init_oscbf, _stream, _motion_target_msg, _send_oscbf,
                                    hold_off, reset_scene, _arm_state, _wait_reached
  package.xml                       exec_depend guide_msgs
  test/test_oscbf_mode.py           new
guide_ex/guide_ex/steps/manipulation/cartesian_move.py   speed/min_fraction through send_action
guide_ex/guide_ex/steps/manipulation/joint_move.py       speed through send_action
guide_ex/guide_ex/steps/end_effector/gripper_control.py  settle on robot.sleep (sim time)
guide_ex/guide_ex/steps/simulation/isaac/prim.py         GetPrimPose asks in robot.config.frame_id
guide_ex/test/test_cartesian_move.py, test_node_fixes.py
guide_msgs/srv/Pose.srv                           request field frame_id (empty: world, as before)
guide_core/guide_core/ros/guide_ros.py            PoseRequest answers in frame_id (pose_frame)
guide_core/test/test_command_fixes.py             one test
guide_tasks/block_bin/launch/bringup.launch.py   reads robots.franka.controller; oscbf: no move_group
guide_tasks/block_bin/block_bin/solve_task.py     base frame (fr3_link0); --controller; sim-time waits
INSTALLATION.md                                   one bullet on the switch
docs/design/s2b_block_bin_oscbf_run.json          new: the run's results
```

---

### Task 1: `ROS2Robot`: config fields, sim-clock sleep, observation fix, MoveIt's speed and straight-line check

**Files:**
- Modify: `modules/irob_lerobot_ros/irob_lerobot_ros/config.py`
- Modify: `modules/irob_lerobot_ros/irob_lerobot_ros/ros2robot.py` (imports; `_motors_obs_ft`, lines 239–240; `send_action`, lines 379–381 and 433–434; `sleep`, lines 533–536)
- Test: `modules/irob_lerobot_ros/test/test_oscbf_mode.py`

**Interfaces:**
- Consumes: `guide_motion.model.Model.load("fr3")` (test helper only).
- Produces: `ROS2RobotConfig.controller: str = "moveit"`, `ROS2RobotConfig.motion_robot: str = ""`, `FR3RobotConfig.motion_robot = "fr3"`; `ROS2Robot.sleep(seconds: float, stall_timeout: float = 30.0) -> None` (node clock = sim time; raises `TimeoutError` when the clock stops); `ROS2Robot.send_action(action, cartesian=False, wait_for_execution=False, speed: float | None = None, min_fraction: float = 0.0) -> bool` (`speed`: MoveIt's scaling, `None` keeps the current one; Task 2 gives it its oscbf meaning); `observation_features` include the gripper joints.

- [ ] **Step 0: Branch the module repo**

```bash
cd ~/ros2_ws/src/guide/modules/irob_lerobot_ros && git status --porcelain && git switch -c feat/oscbf-mode chore/lerobot-0.6.1 && git log --oneline -1
```
Expected: no local changes; `Switched to a new branch 'feat/oscbf-mode'`; `6ce7461 chore: pin the lerobot submodule at v0.6.1`. (The module's own `modules/lerobot` submodule is not initialised; nothing imports it, `.venv` carries lerobot 0.6.1+guide.1.)

- [ ] **Step 1: Write the failing tests**

`modules/irob_lerobot_ros/test/test_oscbf_mode.py`:
```python
"""ROS2Robot's controller: oscbf mode and the sim-time sleep, against a fake node and clock."""

import sys
import threading
import time
from collections import deque
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from geometry_msgs.msg import Pose, Twist
from rclpy.time import Time
from scipy.spatial.transform import Rotation
from sensor_msgs.msg import JointState

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from irob_lerobot_ros.config import ActionType, ObservationType  # noqa: E402
from irob_lerobot_ros.ros2robot import ROS2Robot  # noqa: E402

ARM = [f"fr3_joint{i}" for i in range(1, 8)]
HOME = [0.0, -0.785398, 0.0, -2.35619, 0.0, 1.5708, 0.785398]


class FakeClock:
    """Sim time that advances `step` s on every read, so the polling loops make progress."""

    def __init__(self, step=0.001):
        self.t, self.step = 0, int(step * 1e9)

    def now(self):
        self.t += self.step
        return Time(nanoseconds=self.t)


def robot(q=HOME, clock=None, controller="oscbf"):
    from guide_motion.model import Model

    r = ROS2Robot.__new__(ROS2Robot)
    r.logs = []
    log = r.logs.append
    clock = clock or FakeClock()
    r.node = SimpleNamespace(get_clock=lambda: clock, get_logger=lambda: SimpleNamespace(error=log, warn=log, warning=log, info=log))
    r.cameras = {}
    r.config = SimpleNamespace(controller=controller, arm_joint_names=ARM, gripper_joint_names=["fr3_finger_joint1"],
                               frame_id="fr3_link0", arm_action_type=ActionType.CARTESIAN_POSE)
    r.joint_state = JointState(name=ARM, position=list(q), velocity=[0.0] * 7)
    r._motion = Model.load("fr3")
    r._queue, r._last, r._width, r._queue_lock = deque(), None, float("nan"), threading.Lock()
    return r


@pytest.fixture
def make():
    """robot(...) factory; drops the fake node afterwards, or __del__ would 'disconnect' it."""
    made = []
    yield lambda **kwargs: made.append(robot(**kwargs)) or made[-1]
    for r in made:
        r.node = None


def pose(position, rotation):
    x, y, z, w = Rotation.from_matrix(rotation).as_quat()
    p = Pose()
    p.position.x, p.position.y, p.position.z = (float(v) for v in position)
    p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w = float(x), float(y), float(z), float(w)
    return p


def test_sleep_counts_sim_time_not_wall_time(make):
    r = make(clock=FakeClock(step=0.01))
    start = time.perf_counter()
    r.sleep(1.0)
    assert r.node.get_clock().t >= 1e9 and time.perf_counter() - start < 1.0


def test_sleep_raises_when_the_clock_stops(make):
    r = make(clock=FakeClock(step=0.0))
    with pytest.raises(TimeoutError):
        r.sleep(1.0, stall_timeout=0.05)


def test_moveit_refuses_a_cartesian_path_it_cannot_finish(make):
    r = make(controller="moveit")
    moved = []
    r._moveit2 = SimpleNamespace(max_velocity=1.0, plan=lambda **kwargs: None, move_to_pose=lambda **kwargs: moved.append(kwargs) or True)
    assert not r.send_action(Pose(), cartesian=True, wait_for_execution=True, speed=0.2, min_fraction=0.95)
    assert moved == [] and r._moveit2.max_velocity == 0.2


def test_observation_features_include_the_gripper(make):
    r = make()
    r.config.arm_observation_type = ObservationType.JOINT_POSITION
    r.config.gripper_observation_type = ObservationType.JOINT_POSITION
    assert list(r._motors_obs_ft) == [f"{j}.pos" for j in ARM] + ["fr3_finger_joint1.pos"]
```

- [ ] **Step 2: Run them to see them fail**

Run (sourced shell): `cd ~/ros2_ws/src/guide/modules/irob_lerobot_ros && PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 ~/ros2_ws/.venv/bin/python -m pytest -q test/test_oscbf_mode.py`
Expected: 4 failures (checked on a scratch copy of 6ce7461): the old `sleep` (`AttributeError: 'types.SimpleNamespace' object has no attribute 'create_timer'`, and `TypeError: ROS2Robot.sleep() got an unexpected keyword argument 'stall_timeout'`), `send_action() got an unexpected keyword argument 'speed'`, and the feature list without `fr3_finger_joint1.pos`.

- [ ] **Step 3: Config fields**

In `irob_lerobot_ros/config.py`, replace `    frame_id: str = "base_link"` (in `ROS2RobotConfig`) with:
```python
    frame_id: str = "base_link"

    # Motion layer: "moveit" (pymoveit2 + move_group) or "oscbf" (S2b: guide_motion's planner,
    # MotionTargets at 20 Hz to the OSCBF controller in the simulator's or robot's control loop).
    controller: str = "moveit"
    motion_robot: str = ""  # guide_motion robot description for controller: oscbf ("fr3", "panda")
```
and in `FR3RobotConfig` replace `    directly_publish: bool = False` with:
```python
    directly_publish: bool = False
    motion_robot: str = "fr3"
```

- [ ] **Step 4: `ros2robot.py` edits**

The file has trailing whitespace on many lines: take each `old` text from the file itself when editing.

(a) After `from rclpy.task import Future` add `from rclpy.duration import Duration`.

(b) In `_motors_obs_ft`, replace the two lines
```python
        return {**arm_ft}
        return {**arm_ft, **gripper_ft}
```
with `        return {**arm_ft, **gripper_ft}`.

(c) Replace the head of `send_action`
```python
    def send_action(self, action: Pose | Twist | dict[str, float], cartesian=False, wait_for_execution=False) -> bool:
        if not self.is_connected:
            raise DeviceNotConnectedError("Robot is not connected.")
```
with
```python
    def send_action(self, action: Pose | Twist | dict[str, float], cartesian=False, wait_for_execution=False,
                    speed: float | None = None, min_fraction: float = 0.0) -> bool:
        """speed: GUIDE-EX's speed, 0 to 1: MoveIt's velocity scaling (None keeps the current one), or
        controller: oscbf's cruise speed, speed x 0.5 m/s (None = 1.0). min_fraction: MoveIt's; a
        Cartesian path MoveIt can compute for less than this share of the way is refused before the
        arm moves (controller: oscbf plans straight lines or refuses)."""
        if not self.is_connected:
            raise DeviceNotConnectedError("Robot is not connected.")
        if speed is not None and self._moveit2 is not None:
            self._moveit2.max_velocity = speed
```

(d) Replace
```python
        elif self.config.arm_action_type == ActionType.CARTESIAN_POSE:
            if isinstance(action, Pose):
```
with (the check `cartesian_move.py:57-79` did before, moved here unchanged in effect)
```python
        elif self.config.arm_action_type == ActionType.CARTESIAN_POSE:
            # MoveIt executes whatever part of a Cartesian path it could compute and reports success
            # (pymoveit2's threshold defaults to 0): stopping short of a cube, or at the edge of a
            # self-collision no later plan can start from. Ask for the path first.
            if isinstance(action, Pose) and cartesian and min_fraction > 0 and self._moveit2.plan(
                pose=action,
                frame_id=self.config.frame_id,
                cartesian=True,
                cartesian_fraction_threshold=min_fraction,
                start_joint_state=self.joint_state,
            ) is None:
                self.node.get_logger().warning(f"No straight path (< {min_fraction:.0%}) to the pose; not executed.")
                return False
            if isinstance(action, Pose):
```

(e) Replace the broken `sleep` (it called `create_timer` without a callback)
```python
    def sleep(self, seconds: float):
        rate = self.node.create_timer(seconds)
        rate.sleep()
        rate.destroy()
```
with block_bin_eval's `sleep_sim`, whose stall guard watches the clock stopping instead of the total wall time (a slow simulator is not a dead one):
```python
    def sleep(self, seconds: float, stall_timeout: float = 30.0) -> None:
        """Block for `seconds` of the node's clock: SIM time under use_sim_time.

        GUIDE-EX's waits use it, so a wait lasts the same simulated time at any RTF. Raises
        when the clock stops advancing for `stall_timeout` s of wall time (a dead simulator).
        """
        clock = self.node.get_clock()
        last = clock.now()
        end = last + Duration(seconds=seconds)
        wall = time.perf_counter()
        while (now := clock.now()) < end:
            if now != last:
                last, wall = now, time.perf_counter()
            elif time.perf_counter() - wall > stall_timeout:
                raise TimeoutError(f"The clock did not advance for {stall_timeout} s of wall time; is the simulation running?")
            time.sleep(0.002)
```

- [ ] **Step 5: Run them to see them pass**

Run: `cd ~/ros2_ws/src/guide/modules/irob_lerobot_ros && PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 ~/ros2_ws/.venv/bin/python -m pytest -q test/test_oscbf_mode.py test/test_depth_stream.py`
Expected: `8 passed` (these 4 and the 4 depth-stream tests).

- [ ] **Step 6: Commit (module repo)**

```bash
cd ~/ros2_ws/src/guide/modules/irob_lerobot_ros
git add irob_lerobot_ros/config.py irob_lerobot_ros/ros2robot.py test/test_oscbf_mode.py
git commit -m "feat(ros2robot): sim-clock sleep, gripper observation features, MoveIt speed and straight-line check in send_action"
```

---

### Task 2: `ROS2Robot` `controller: oscbf`

**Files:**
- Modify: `modules/irob_lerobot_ros/irob_lerobot_ros/ros2robot.py` (imports; `connect`, lines 144–154; `send_action` head; `disconnect`; new methods at the end of the class)
- Modify: `modules/irob_lerobot_ros/package.xml`
- Test: `modules/irob_lerobot_ros/test/test_oscbf_mode.py` (append)

**Interfaces:**
- Consumes: Plan A's `Model.load(config.motion_robot)` (`arm_joints`, `base_link` = `fr3_link0`, `ee_pose`, `ee_speed`), `guide_motion.planner.plan(model, goal, q, speed) -> list[Target]` (cruise `speed × 0.5 m/s`, targets 50 ms apart, the last one the goal), `PlanningError`, `guide_motion.targets.Target/TWIST/JOINTS` (`gripper_width` NaN = leave the gripper); Plan B's `guide_msgs.msg.MotionTarget` and its Reset (`OscbfHost.home()`: Plan A's JOINTS plan at speed 1.0 to the scene's `default_joint_states[:7]`, held until the next fresh target); Task 1's config fields and `send_action` signature.
- Produces: in oscbf mode `connect()` starts no MoveIt, gripper or servo interfaces and calls `init_oscbf()` (publisher `motion_target`, depth 1, in the robot's namespace, i.e. `/Sim_0/Scene_0/franka/motion_target`). `send_action(Pose, speed=)` takes the pose in the base frame (a `config.frame_id` other than `Model.base_link` is refused: False plus an error), plans from the measured joints at cruise `speed × 0.5 m/s` (`None` = 1.0) and queues the targets; with `wait_for_execution=True` it returns `_wait_reached(goal, 2 s + 2 × planned duration)`. `send_action(Twist)` queues a one-shot `MODE_TWIST`. `send_action({"<joint>.pos": …})` sets the gripper width (2 × the finger value) and/or plans a full arm joint goal at the same speed. The streaming thread publishes the next target every 50 ms of sim time and repeats the last pose or joint target; every message sets `gripper_width` (NaN until a gripper command). `disconnect()` stops it. `hold_off()` empties the queue and drops the last target. `reset_scene(service, request, home) -> bool` is the one Reset path: `hold_off()`, the service, then `_wait_reached` on the JOINTS goal `home` within 2 s + 2 × the host's homing plan.

- [ ] **Step 1: Write the failing tests**

Append to `modules/irob_lerobot_ros/test/test_oscbf_mode.py`:
```python
def test_a_reached_goal_must_be_held_for_a_tenth_of_a_second(make):
    from guide_motion.targets import Target

    r = make()
    p, R = r._motion.ee_pose(HOME)
    assert r._wait_reached(Target(position=p, rotation=R), timeout=1.0)
    assert r.node.get_clock().t >= 0.1e9


def test_an_arm_elsewhere_times_out_on_sim_time(make):
    from guide_motion.targets import Target

    r = make()
    p, R = r._motion.ee_pose(HOME)
    assert not r._wait_reached(Target(position=p + [0.1, 0, 0], rotation=R), timeout=0.3)
    assert any("timed out" in str(line) for line in r.logs)


def test_an_arm_at_the_goal_still_moving_has_not_arrived(make):
    from guide_motion.targets import Target

    r = make()
    r.joint_state.velocity = [0.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0]  # joint 2 at 0.1 rad/s: the tool moves ~5 cm/s
    p, R = r._motion.ee_pose(HOME)
    assert not r._wait_reached(Target(position=p, rotation=R), timeout=0.3)


def test_speed_sets_the_cruise_speed_and_none_means_full_speed(make):
    r = make()
    p, R = r._motion.ee_pose(HOME)
    assert r.send_action(pose(p + [0.0, 0.2, 0.0], R), wait_for_execution=False)
    assert len(r._queue) == 19  # speed None = 1.0: cruise 0.5 m/s
    np.testing.assert_allclose(r._queue[-1].position, p + [0.0, 0.2, 0.0], atol=1e-9)
    assert r.send_action(pose(p + [0.0, 0.2, 0.0], R), speed=0.2)
    assert len(r._queue) == 48  # cruise 0.1 m/s


def test_an_unreachable_pose_returns_false(make):
    r = make()
    p, R = r._motion.ee_pose(HOME)
    assert not r.send_action(pose(p + [1.5, 0.0, 0.0], R))
    assert any("planner" in str(line) for line in r.logs)


def test_a_finger_position_becomes_the_width_and_starts_a_hold(make):
    r = make()
    assert r.send_action({"fr3_finger_joint1.pos": 0.04})
    assert r._width == pytest.approx(0.08) and r._last is not None
    np.testing.assert_allclose(r._last.position, r._motion.ee_pose(HOME)[0])


def test_the_stream_sends_the_plan_then_repeats_the_last_target(make):
    from guide_motion.targets import TWIST, Target

    r = make()
    sent = []
    r._motion_target_msg = lambda target, width: (target, width)
    r._motion_pub = SimpleNamespace(publish=sent.append)
    r._stop_stream = threading.Event()
    targets = [Target(position=np.full(3, i)) for i in range(3)]
    r._queue = deque(targets)
    thread = threading.Thread(target=r._stream, daemon=True)
    thread.start()
    time.sleep(0.8)
    assert [t for t, _ in sent[:3]] == targets and all(t is targets[-1] for t, _ in sent[3:]) and len(sent) > 4
    assert all(np.isnan(width) for _, width in sent)  # no gripper command yet: every target leaves the hand
    with r._queue_lock:
        r._queue = deque([Target(mode=TWIST, twist=np.ones(6))])
    time.sleep(0.5)
    r._stop_stream.set()
    thread.join()
    twists = [t for t, _ in sent if t.mode == TWIST]
    assert len(twists) == 1 and sent[-1][0].mode == TWIST  # one-shot, nothing after it


def test_twist_is_queued_without_waiting(make):
    r = make()
    twist = Twist()
    twist.linear.x = 0.05
    assert r.send_action(twist, wait_for_execution=True)
    assert r._queue[0].twist[0] == pytest.approx(0.05)


class RestartingClock(FakeClock):
    """Restarts at 0 once, after 0.05 s, as Isaac's clock does on stop/play."""

    def now(self):
        if self.t > 0.05e9 and not getattr(self, "restarted", False):
            self.t, self.restarted = 0, True
        return super().now()


def test_a_clock_restart_during_a_wait_still_times_out(make):
    from guide_motion.targets import Target

    r = make(clock=RestartingClock())
    p, R = r._motion.ee_pose(HOME)
    started = time.perf_counter()
    assert not r._wait_reached(Target(position=p + [0.1, 0, 0], rotation=R), timeout=0.2)
    assert time.perf_counter() - started < 5.0


def test_a_new_goal_replaces_the_queue_and_starts_from_the_measured_state(make):
    r = make()
    p, R = r._motion.ee_pose(HOME)
    r.send_action(pose(p + [0.0, 0.2, 0.0], R))
    r.send_action(pose(p + [0.0, 0.0, -0.1], R))
    assert np.allclose(r._queue[-1].position, p + [0.0, 0.0, -0.1])
    assert np.linalg.norm(r._queue[0].position - p) < 0.01  # 4.6 mm from where the arm is; the old plan ended 0.2 m away


def test_a_pose_outside_the_base_frame_is_refused(make):
    r = make()
    r.config.frame_id = "Scene_0"
    p, R = r._motion.ee_pose(HOME)
    assert not r.send_action(pose(p, R))
    assert not r._queue and any("refused" in str(line) for line in r.logs)


def test_a_partial_joint_goal_is_refused(make):
    r = make()
    assert not r.send_action({"fr3_joint1.pos": 0.0})
    assert not r._queue


def test_a_target_becomes_a_motion_target_message(make):
    from guide_msgs.msg import MotionTarget

    from guide_motion.targets import JOINTS, Target

    r = make()
    r._motion_msg = MotionTarget
    msg = r._motion_target_msg(Target(mode=JOINTS, joints=np.array(HOME)), 0.08)
    assert msg.mode == MotionTarget.MODE_JOINTS and msg.frame_id == "fr3_link0"
    assert list(msg.joints) == pytest.approx(HOME) and msg.gripper_width == 0.08
    assert msg.pose.orientation.w == pytest.approx(1.0)  # identity rotation for a joint target
    assert np.isnan(r._motion_target_msg(Target(), float("nan")).gripper_width)  # NaN goes out as NaN, not 0.0


def test_reset_holds_off_the_stream_before_it_calls_the_service(make):
    r = make()
    p, R = r._motion.ee_pose(HOME)
    r.send_action(pose(p + [0.0, 0.2, 0.0], R))
    r._last = r._queue.popleft()  # mid-plan: one target sent, the rest queued
    seen = []
    r.callService = lambda service, request, message=None: seen.append((len(r._queue), r._last)) or SimpleNamespace(success=True)
    assert r.reset_scene("/Sim_0/Reset", "request", HOME)
    assert seen == [(0, None)]  # when the host starts homing, nothing is queued or repeated
    assert not r._queue and r._last is None  # and nothing streams until the next send_action


def test_reset_waits_on_sim_time_for_the_home_configuration(make):
    away = list(HOME)
    away[0] = 0.5
    r = make(q=away, clock=FakeClock(step=0.01))
    r.callService = lambda service, request, message=None: SimpleNamespace(success=True)
    assert not r.reset_scene("/Sim_0/Reset", "request", HOME)  # the arm never arrives
    assert 3.4e9 <= r.node.get_clock().t <= 3.45e9  # timeout 2 s + 2 x 14 homing targets x 50 ms, sim time
    r.joint_state.position = list(HOME)  # the host has homed it
    assert r.reset_scene("/Sim_0/Reset", "request", HOME)
```

- [ ] **Step 2: Run them to see them fail**

Run: `cd ~/ros2_ws/src/guide/modules/irob_lerobot_ros && PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 ~/ros2_ws/.venv/bin/python -m pytest -q test/test_oscbf_mode.py`
Expected (checked on a scratch copy): `15 failed, 4 passed`: `AttributeError: 'ROS2Robot' object has no attribute '_wait_reached'` (4), `'_stream'`, `'_motion_target_msg'`, `'reset_scene'`; `send_action` in oscbf mode still on the MoveIt branch (`'ROS2Robot' object has no attribute '_moveit2'`, 5); and the gripper branch reading the fake config's missing `gripper_action_type` (3).

- [ ] **Step 3: Imports and the period**

Replace the first five lines of `ros2robot.py`
```python
import time
from functools import cached_property
from functools import wraps
from threading import Event, Thread
from typing import Optional
```
with
```python
import time
from collections import deque
from functools import cached_property
from functools import wraps
from threading import Event, Lock, Thread
from typing import Optional

import numpy as np
from scipy.spatial.transform import Rotation
```
and replace `from geometry_msgs.msg import Twist` with
```python
from geometry_msgs.msg import Twist

MOTION_DT = 0.05  # MotionTarget period, s of sim time (20 Hz, S2b; guide_motion.planner.DT)
```

- [ ] **Step 4: `connect`, `send_action`, `disconnect`**

In `connect`, replace the block from `        # Connect to MoveIt2 and Gripper interfaces based on the specified action types` through `            self.init_servo_interface()` (lines 144–154 at 6ce7461, 145–155 after Task 1) with:
```python
        if self.config.controller == "oscbf":
            self.init_oscbf()  # guide_motion plans and streams; nothing of MoveIt is started
        else:
            # Connect to MoveIt2 and Gripper interfaces based on the specified action types
            if self.config.arm_action_type in (ActionType.JOINT_POSITION, ActionType.JOINT_TRAJECTORY, ActionType.CARTESIAN_POSE):
                self.init_moveit2()
            if self.config.gripper_action_type in (ActionType.JOINT_POSITION, ActionType.JOINT_TRAJECTORY):
                self.init_gripper_interface()
            if self.config.arm_action_type == ActionType.CARTESIAN_VELOCITY:
                self.init_servo_interface()
```
In `send_action`, replace (Task 1's line)
```python
        if speed is not None and self._moveit2 is not None:
```
with
```python
        if self.config.controller == "oscbf":
            return self._send_oscbf(action, wait_for_execution, speed)
        if speed is not None and self._moveit2 is not None:
```
In `disconnect`, replace `        print('Shutting down ROS node...')` with
```python
        if getattr(self, "_stream_thread", None) is not None:
            self._stop_stream.set()
            self._stream_thread.join()
            self._stream_thread = None

        print('Shutting down ROS node...')
```

- [ ] **Step 5: The oscbf methods**

Append to the end of `ros2robot.py` (inside `class ROS2Robot`, after `callService`):
```python

    # ------------------------------------------------------------------
    # controller: oscbf (S2b): guide_motion plans, MotionTargets stream at 20 Hz of sim time
    # to the OSCBF controller that runs in the simulator's (or the robot's) control loop.
    # ------------------------------------------------------------------
    def init_oscbf(self):
        from guide_motion.model import Model
        from guide_msgs.msg import MotionTarget

        self._motion = Model.load(self.config.motion_robot)
        assert self._motion.arm_joints == list(self.config.arm_joint_names), self._motion.arm_joints
        self._motion_msg = MotionTarget
        self._motion_pub = self.node.create_publisher(MotionTarget, "motion_target", 1)
        self._queue, self._last, self._width = deque(), None, float("nan")
        self._queue_lock, self._stop_stream = Lock(), Event()
        self._stream_thread = Thread(target=self._stream, daemon=True)
        self._stream_thread.start()

    def _stream(self):
        """Every 50 ms of sim time: the next planned target, else the last pose/joint target
        again, so the controller never goes stale while it holds a goal."""
        from guide_motion.targets import TWIST

        clock = self.node.get_clock()
        due = clock.now()
        while not self._stop_stream.wait(0.002):  # poll the clock every 2 ms of wall time
            now = clock.now()
            if now < due and (due - now).nanoseconds < 1e9:  # a clock > 1 s behind restarted: send now
                continue
            due = now + Duration(seconds=MOTION_DT)
            with self._queue_lock:
                if self._queue:
                    self._last = self._queue.popleft()
                target, width = self._last, self._width
                if target is not None and target.mode == TWIST:
                    self._last = None  # one-shot, like a servo command: the host holds the pose after 0.25 s
            if target is not None:
                self._motion_pub.publish(self._motion_target_msg(target, width))

    def _motion_target_msg(self, target, width):
        msg = self._motion_msg()
        msg.stamp = self.node.get_clock().now().to_msg()
        msg.mode = int(target.mode)
        msg.frame_id = self._motion.base_link
        x, y, z, w = Rotation.from_matrix(target.rotation).as_quat()
        msg.pose.position.x, msg.pose.position.y, msg.pose.position.z = (float(v) for v in target.position)
        msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z, msg.pose.orientation.w = float(x), float(y), float(z), float(w)
        msg.twist.linear.x, msg.twist.linear.y, msg.twist.linear.z = (float(v) for v in target.twist[:3])
        msg.twist.angular.x, msg.twist.angular.y, msg.twist.angular.z = (float(v) for v in target.twist[3:])
        msg.joints = [float(v) for v in target.joints]
        msg.gripper_width = float(width)  # always set: the field's default 0.0 closes the hand; NaN leaves it
        return msg

    def _send_oscbf(self, action, wait_for_execution: bool, speed: float | None) -> bool:
        from guide_motion.planner import PlanningError, plan
        from guide_motion.targets import JOINTS, TWIST, Target

        names = list(self.config.arm_joint_names)
        if isinstance(action, Pose):
            # Ruling 4: poses arrive in the robot's base frame; GUIDE converts, the solver looks nothing up
            if self.config.frame_id != self._motion.base_link:
                self.node.get_logger().error(
                    f"Pose in '{self.config.frame_id}', not the base frame '{self._motion.base_link}': refused.")
                return False
            p, o = action.position, action.orientation
            goal = Target(position=np.array([p.x, p.y, p.z], float), rotation=Rotation.from_quat([o.x, o.y, o.z, o.w]).as_matrix())
        elif isinstance(action, Twist):
            v, w = action.linear, action.angular
            goal = Target(mode=TWIST, twist=np.array([v.x, v.y, v.z, w.x, w.y, w.z], float))
        elif isinstance(action, dict):
            fingers = [v for k, v in action.items() if k.removesuffix(".pos") in self.config.gripper_joint_names]
            arm = {k.removesuffix(".pos"): v for k, v in action.items() if k.removesuffix(".pos") in names}
            if fingers:
                with self._queue_lock:
                    self._width = 2.0 * float(fingers[0])  # one finger's travel -> opening width
                    if self._last is None and not self._queue:  # nothing streams yet: hold here so the width goes out
                        position, rotation = self._motion.ee_pose(self._arm_state()[0])
                        self._last = Target(position=position, rotation=rotation)
            if arm and len(arm) != len(names):
                self.node.get_logger().warn("Incomplete arm joint configuration provided in action.")
                return False
            if not arm:
                return bool(fingers)
            goal = Target(mode=JOINTS, joints=np.array([arm[j] for j in names], float))
        else:
            return False
        try:
            # Ruling 3: GUIDE-EX's speed sets the cruise speed, speed x 0.5 m/s; None = 1.0
            targets = plan(self._motion, goal, self._arm_state()[0], speed=1.0 if speed is None else speed)
        except PlanningError as error:
            self.node.get_logger().error(f"guide_motion planner: {error}")
            return False
        with self._queue_lock:
            self._queue = deque(targets)
        if not wait_for_execution or goal.mode == TWIST:
            return True
        return self._wait_reached(goal, timeout=2.0 + 2.0 * len(targets) * MOTION_DT)

    def hold_off(self) -> None:
        """Stop streaming: nothing queued, nothing repeated (Plan B final review I2). GUIDE's Reset
        homes the arm through the host until the next fresh target, and the 20 Hz repeat of the last
        goal would be one within 50 ms. The next send_action resumes the stream."""
        with self._queue_lock:
            self._queue, self._last = deque(), None

    def reset_scene(self, service, request, home) -> bool:
        """GUIDE's Reset under controller: oscbf, the one Reset path: hold off first, call `service`
        (GUIDE's Reset), then wait on sim time until the tool is within the reach tolerance of
        `home` (the scene's default_joint_states[:7], where the host homes the arm), for at most
        2 s + 2 x the homing plan's duration (the host plans it at speed 1.0)."""
        from guide_motion.planner import plan
        from guide_motion.targets import JOINTS, Target

        self.hold_off()
        goal = Target(mode=JOINTS, joints=np.asarray(home, float))
        homing = plan(self._motion, goal, self._arm_state()[0], speed=1.0)
        response = self.callService(service, request, "Resetting the scene")
        if not response.success:
            self.node.get_logger().error(f"Reset failed: {response.message}")
            return False
        return self._wait_reached(goal, timeout=2.0 + 2.0 * len(homing) * MOTION_DT)

    def _arm_state(self):
        js = self.joint_state
        index = [js.name.index(j) for j in self.config.arm_joint_names]
        q = np.array([js.position[i] for i in index], float)
        qd = np.array([js.velocity[i] for i in index], float) if len(js.velocity) == len(js.name) else np.zeros(len(index))
        return q, qd

    def _wait_reached(self, goal, timeout: float, stall_timeout: float = 30.0) -> bool:
        """Tool within 5 mm and 2 deg of the goal, slower than 1 cm/s, held 0.1 s of sim time (S2b 4.5)."""
        from guide_motion.targets import JOINTS

        position, rotation = self._motion.ee_pose(goal.joints) if goal.mode == JOINTS else (goal.position, goal.rotation)
        clock = self.node.get_clock()
        start = last = clock.now()
        wall, held = time.perf_counter(), None
        while (now := clock.now()) - start < Duration(seconds=timeout):
            if now < start:  # the simulator restarted, and its clock with it
                start, held = now, None
            if now != last:
                last, wall = now, time.perf_counter()
            elif time.perf_counter() - wall > stall_timeout:
                break
            q, qd = self._arm_state()
            p, R = self._motion.ee_pose(q)
            reached = (np.linalg.norm(p - position) < 0.005
                       and Rotation.from_matrix(rotation.T @ R).magnitude() < np.radians(2.0)
                       and self._motion.ee_speed(q, qd) < 0.01)
            held = (held or now) if reached else None
            if held is not None and now - held >= Duration(seconds=0.1):
                return True
            time.sleep(0.002)
        self.node.get_logger().error(f"OSCBF move timed out after {timeout:.1f} s of sim time.")
        return False
```

In `package.xml`, after `  <exec_depend>pymoveit2</exec_depend>` add:
```xml
  <!-- controller: oscbf (S2b) publishes guide_msgs/MotionTarget -->
  <exec_depend>guide_msgs</exec_depend>
```

- [ ] **Step 6: Run the tests and build**

```bash
cd ~/ros2_ws/src/guide/modules/irob_lerobot_ros && PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 ~/ros2_ws/.venv/bin/python -m pytest -q test/test_oscbf_mode.py test/test_depth_stream.py
cd ~/ros2_ws && colcon build --packages-select irob_lerobot_ros && source install/setup.zsh
```
Expected: `23 passed` (19 in test_oscbf_mode, about 7 s, and the 4 depth tests); `Finished <<< irob_lerobot_ros`.

- [ ] **Step 7: Commit (module repo)**

```bash
cd ~/ros2_ws/src/guide/modules/irob_lerobot_ros
git add irob_lerobot_ros/ros2robot.py package.xml test/test_oscbf_mode.py
git commit -m "feat(ros2robot): controller: oscbf: guide_motion planning at GUIDE-EX's speed, 20 Hz MotionTarget stream on sim time, Reset holds off"
```

---

### Task 3: The GUIDE-EX seam

**Files:**
- Modify: `guide_ex/guide_ex/steps/manipulation/cartesian_move.py` (lines 32–86), `guide_ex/guide_ex/steps/manipulation/joint_move.py` (lines 46–50), `guide_ex/guide_ex/steps/end_effector/gripper_control.py` (lines 1, 57)
- Test: `guide_ex/test/test_cartesian_move.py` (rewrite), `guide_ex/test/test_node_fixes.py` (one test)

**Interfaces:**
- Consumes: Task 1's `send_action(..., speed=, min_fraction=)` and `ROS2Robot.sleep`.
- Produces: `MoveToCartesianPose.run(robot, target_pose, speed=1.0, cartesian=False, min_fraction=0.95)` and `MoveToJointConfiguration.run(robot, target_configuration, speed=1.0)` calling only `robot.send_action(...)` (3 tries, as before); `SetGripperState` settling with `robot.sleep(SETTLE_SECONDS)`.

- [ ] **Step 1: Write the failing tests**

Replace `guide_ex/test/test_cartesian_move.py` with:
```python
"""MoveToCartesianPose / MoveToJointConfiguration only talk to the robot through send_action."""

import logging
from types import SimpleNamespace

from irob_lerobot_ros.config import ActionType
from irob_lerobot_ros.ros2robot import ROS2Robot

from guide_core.types.geometry import Point, Pose
from guide_ex.core.states import DemoStatus
from guide_ex.steps.manipulation.cartesian_move import MoveToCartesianPose
from guide_ex.steps.manipulation.joint_move import MoveToJointConfiguration

TARGET = Pose(position=Point([0.2, 0.0, 0.3]))


class FakeRobot(ROS2Robot):
    def __del__(self):  # never connected, nothing to disconnect
        pass


def robot_answering(result):
    """A ROS2Robot with no MoveIt at all, whose send_action answers `result` and records its kwargs."""
    robot = FakeRobot.__new__(FakeRobot)
    robot.node = SimpleNamespace(get_logger=lambda: logging.getLogger("test"))
    robot.config = SimpleNamespace(arm_action_type=ActionType.CARTESIAN_POSE, frame_id="fr3_link0")
    robot._moveit2 = None  # the seam: a step that reaches into MoveIt fails here
    robot.sent = []
    robot.send_action = lambda action, **kwargs: robot.sent.append(kwargs) or result
    return robot


def test_speed_and_straightness_go_to_the_robot():
    robot = robot_answering(True)

    result = MoveToCartesianPose().run(robot, TARGET, speed=0.2, cartesian=True, min_fraction=1.0)

    assert result.status == DemoStatus.PERFECT
    assert robot.sent == [{"cartesian": True, "wait_for_execution": True, "speed": 0.2, "min_fraction": 1.0}]


def test_a_refused_move_fails_after_three_tries():
    robot = robot_answering(False)

    assert MoveToCartesianPose().run(robot, TARGET, cartesian=True).status == DemoStatus.FAILURE
    assert len(robot.sent) == 3


def test_a_joint_move_passes_its_speed():
    robot = robot_answering(True)
    robot.config.arm_action_type = ActionType.JOINT_POSITION

    result = MoveToJointConfiguration().run(robot, {"fr3_joint1.pos": 0.0}, speed=0.3)

    assert result.status == DemoStatus.PERFECT
    assert robot.sent == [{"wait_for_execution": True, "speed": 0.3}]
```
In `guide_ex/test/test_node_fixes.py`, replace `test_set_gripper_state_sends_every_joint_it_is_given` (the whole function) with:
```python
def test_set_gripper_state_sends_every_joint_and_settles_on_sim_time():
    sent, slept = [], []
    robot = fake_robot(
        config=SimpleNamespace(gripper_action_type=ActionType.JOINT_POSITION),
        send_action=lambda action, **kwargs: sent.append(action) or True,
        sleep=slept.append,  # ROS2Robot.sleep: the node's (sim) clock
    )

    result = SetGripperState().run(robot, {"fr3_finger_joint1": 0.04, "fr3_finger_joint2": 0.04})

    assert result.status == DemoStatus.PERFECT
    assert sent == [{"fr3_finger_joint1.pos": 0.04, "fr3_finger_joint2.pos": 0.04}]
    assert slept == [SetGripperState.SETTLE_SECONDS]
```

- [ ] **Step 2: Run them to see them fail**

Run (sourced, Task 2 built): `cd ~/ros2_ws/src/guide && PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 ~/ros2_ws/.venv/bin/python -m pytest -q guide_ex/test/test_cartesian_move.py guide_ex/test/test_node_fixes.py`
Expected: `4 failed, 2 passed` (checked on a scratch copy): the three seam tests fail with `AttributeError: 'NoneType' object has no attribute 'max_velocity'` (the installed steps still touch `robot._moveit2`); the gripper test fails with `assert [] == [1.0]` (it still calls `time.sleep`, for 1 s).

- [ ] **Step 3: The steps talk to `send_action` only**

In `guide_ex/guide_ex/steps/manipulation/cartesian_move.py`, `MoveToCartesianPose.run`: replace the block from `            robot._moveit2.max_velocity = speed  # Set the speed for the move` through the end of the retry loop (`                if success:` / `                    break`), i.e. the speed line, the `if cartesian and (robot._moveit2.plan(...) is None): ... return ExecutionResult(FAILURE ...)` check and the loop, with:
```python
            # speed and min_fraction go to the robot: MoveIt's velocity scaling and straight-line
            # check (controller: moveit), or the planner's cruise speed (controller: oscbf).
            for _ in range(3):  # Retry logic for robustness
                success = robot.send_action(
                    action=target_pose.to_ros(),
                    cartesian=cartesian,
                    wait_for_execution=True,
                    speed=speed,
                    min_fraction=min_fraction,
                )
                if success:
                    break
```
and in its docstring replace the `speed` and `min_fraction` lines with:
```python
            speed (float): 0 to 1 (default: 1.0): MoveIt's velocity scaling factor, or
                controller: oscbf's cruise speed, speed x 0.5 m/s.
            cartesian (bool): Whether to execute the move in Cartesian space (default: False).
            min_fraction (float): A straight-line move MoveIt can compute for less than
                this share of the way is refused before the arm moves (controller: moveit;
                oscbf plans straight lines or refuses).
```
In `guide_ex/guide_ex/steps/manipulation/joint_move.py` replace its docstring's `speed` line
```python
            speed (float): MoveIt's velocity scaling factor for the move, 0 to 1 (default: 1.0).
```
with
```python
            speed (float): 0 to 1 (default: 1.0): MoveIt's velocity scaling factor, or
                controller: oscbf's cruise speed, speed x 0.5 m/s.
```
delete the line `            robot._moveit2.max_velocity = speed  # Set the speed for the move` (and the blank line after it) and replace
```python
                success = robot.send_action(action=target_configuration, wait_for_execution=True)
```
with
```python
                success = robot.send_action(
                    action=target_configuration, wait_for_execution=True, speed=speed
                )
```
In `guide_ex/guide_ex/steps/end_effector/gripper_control.py` delete `import time` (line 1) and replace `            time.sleep(self.SETTLE_SECONDS)` with
```python
            robot.sleep(self.SETTLE_SECONDS)  # sim time: the same settle at any RTF
```

- [ ] **Step 4: Build, then run them to see them pass**

The tests import the installed `guide_ex`, so the build comes first.
```bash
cd ~/ros2_ws/src/guide && grep -rn "_moveit2" guide_ex/guide_ex || echo "no MoveIt in guide_ex"
cd ~/ros2_ws && colcon build --packages-select guide_ex && source install/setup.zsh
cd ~/ros2_ws/src/guide && PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 ~/ros2_ws/.venv/bin/python -m pytest -q guide_ex/test \
  --ignore-glob='*test_flake8.py' --ignore-glob='*test_pep257.py' --ignore-glob='*test_copyright.py'
```
Expected: `no MoveIt in guide_ex`; `Finished <<< guide_ex`; `21 passed` (checked on a scratch copy).

- [ ] **Step 5: Commit**

```bash
cd ~/ros2_ws/src/guide
git add guide_ex/guide_ex/steps/manipulation/cartesian_move.py guide_ex/guide_ex/steps/manipulation/joint_move.py \
  guide_ex/guide_ex/steps/end_effector/gripper_control.py guide_ex/test/test_cartesian_move.py guide_ex/test/test_node_fixes.py
git commit -m "refactor(guide_ex): motion steps use send_action only; gripper settle on sim time"
```

---

### Task 3b: GUIDE answers in the robot's base frame (Ruling 4)

**Files:**
- Modify: `guide_msgs/srv/Pose.srv`; `guide_core/guide_core/ros/guide_ros.py` (`_pose_request_callback`; new `pose_frame` before `target_from_msg`); `guide_ex/guide_ex/steps/simulation/isaac/prim.py` (`GetPrimPose.run`); `guide_tasks/block_bin/block_bin/solve_task.py` (an import; `rest_pose`; the GetTargetPose, Pick, Place and PickRecovery maps; `main`'s `FR3RobotConfig`)
- Test: `guide_core/test/test_command_fixes.py` (one test); `guide_ex/test/test_node_fixes.py` (one test, one fixture argument)

**Interfaces:**
- Consumes: each scene's parsed `init.yaml` (`SceneManager._scenes[i]._config["robots"][<name>]`: `path` and `base_name`, block_bin `/fr3` and `fr3_link0`, the base link a direct child of the robot prim, as `block_bin.usd`'s `/Scene/fr3/fr3_link0`); the runtime's `get_world_poses(prim_path)`; `Pose.inv()` and `*` (`guide_core.types.geometry`); `ROS2Robot.config.frame_id` (`FR3RobotConfig`'s default is `fr3_link0`).
- Produces: `Pose.srv` request field `string frame_id` (empty: the world frame, so block_bin_eval's own PoseRequest clients are unchanged); `guide_ros.pose_frame(scenes, path, frame_id) -> str`; `GetPrimPose` asking in `robot.config.frame_id`; block_bin wholly in `fr3_link0` (solver `frame_id`, rest pose, every GetPrimPose answer). cube_stack (frame_id `Scene_<i>`) gets answers in its scene frame, which its own scene-inverse steps then leave as they were.

- [ ] **Step 1: The request names its frame**

Replace `guide_msgs/srv/Pose.srv`'s first line `string path` with:
```
string path
# The frame the pose is answered in (S2b Ruling 4): a robot's base link (robots.<name>.base_name,
# e.g. fr3_link0) or the scene itself (Scene_<i>), looked up in the scene `path` lies in. Empty: world.
string frame_id
```
```bash
cd ~/ros2_ws && colcon build --packages-select guide_msgs && source install/setup.zsh
```
Expected: `Finished <<< guide_msgs`. GUIDE must be restarted to serve the new type (any running instance keeps the old hash; calls then never complete).

- [ ] **Step 2: Write the failing tests**

Append to `guide_core/test/test_command_fixes.py`:
```python
def test_a_pose_request_answers_in_the_requesting_robots_base_frame(isaac_import):
    """Ruling 4: GUIDE converts. A block on Scene_1 (world y + 10, z + 1), its robot's base at
    (-0.3, 10, 1.013) turned 90 deg about z: the answer is the block as that base sees it."""
    import numpy as np
    from guide_msgs.srv import Pose as PoseSrv
    from scipy.spatial.transform import Rotation as R

    from guide_core.types.geometry import Point, Pose, Rotation

    ros = isaac_import("guide_core.ros.guide_ros").GUIDEROS2Interface
    world = {
        "/Scene_1": Pose(Point([0.0, 10.0, 1.0])),
        "/Scene_1/fr3/fr3_link0": Pose(Point([-0.3, 10.0, 1.013]), Rotation(R.from_euler("z", 90, degrees=True))),
        # 0.4 ahead of the base (world +y once it is turned) and 0.4 above it
        "/Scene_1/blocks/red_block": Pose(Point([-0.3, 10.4, 1.413])),
    }
    config = {"robots": {"franka": {"path": "/fr3", "base_name": "fr3_link0"}}}
    backend = SimpleNamespace(
        call=lambda name, prim_path: world[prim_path],
        _scene_manager=SimpleNamespace(_scenes=[None, SimpleNamespace(_config=config)]),
    )
    me = SimpleNamespace(_backend=backend)

    def ask(frame_id):
        request = PoseSrv.Request(path="/Scene_1/blocks/red_block", frame_id=frame_id)
        return ros._pose_request_callback(me, request, None)

    reply = ask("fr3_link0")
    p, q = reply.pose.position, reply.pose.orientation
    assert reply.success and np.allclose([p.x, p.y, p.z], [0.4, 0.0, 0.4])
    assert np.allclose([q.x, q.y, q.z, q.w], [0.0, 0.0, -np.sqrt(0.5), np.sqrt(0.5)])  # the base's -90 deg
    p = ask("Scene_1").pose.position  # the scene's own frame, as a solver with frame_id Scene_<i> moves in
    assert np.allclose([p.x, p.y, p.z], [-0.3, 0.4, 0.413])
    p = ask("").pose.position  # no frame: world, as before
    assert np.allclose([p.x, p.y, p.z], [-0.3, 10.4, 1.413])
    reply = ask("ur_base_link")
    assert not reply.success and "ur_base_link" in reply.message
```
In `guide_ex/test/test_node_fixes.py`, give `test_get_prim_pose_fails_when_the_simulator_has_no_such_prim`'s robot a frame: replace
```python
    robot = fake_robot(pose=object(), callService=lambda client, request, message: no_prim)
```
with
```python
    robot = fake_robot(
        pose=object(),
        config=SimpleNamespace(frame_id="fr3_link0"),
        callService=lambda client, request, message: no_prim,
    )
```
and append:
```python


def test_get_prim_pose_asks_for_the_robots_own_frame():
    asked = []
    robot = fake_robot(
        pose=object(),
        config=SimpleNamespace(frame_id="fr3_link0"),
        callService=lambda client, request, message: asked.append(request) or PoseSrv.Response(success=True),
    )

    result = GetPrimPose().run(robot, "/Sim_0", "/Scene_0", "/blocks/red_block")

    assert result.status == DemoStatus.PERFECT
    assert (asked[0].path, asked[0].frame_id) == ("/Scene_0/blocks/red_block", "fr3_link0")
```

- [ ] **Step 3: Run them to see them fail**

```bash
cd ~/ros2_ws/src/guide && PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 ~/ros2_ws/.venv/bin/python -m pytest -q guide_core/test/test_command_fixes.py -k pose_request
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 ~/ros2_ws/.venv/bin/python -m pytest -q guide_ex/test/test_node_fixes.py
```
Expected (checked on scratch copies): one failure in each. The guide_core test gets the world answer (`np.allclose([-0.3, 10.4, 1.413], [0.4, 0.0, 0.4])` is False); the guide_ex test sees `('/Scene_0/blocks/red_block', '') == (…, 'fr3_link0')` (the installed step leaves `frame_id` empty). The other node tests pass.

- [ ] **Step 4: GUIDE converts, GetPrimPose asks**

In `guide_core/guide_core/ros/guide_ros.py`, `_pose_request_callback`: replace
```python
            pose: Pose = self._backend.call("get_world_poses", prim_path=path)
            if isinstance(pose, bool):
                raise Exception("Getting pose failed!")

            response.pose = pose.to_ros()
```
with
```python
            pose: Pose = self._backend.call("get_world_poses", prim_path=path)
            if isinstance(pose, bool):
                raise Exception("Getting pose failed!")
            if request.frame_id:  # Ruling 4: GUIDE converts into the requesting robot's frame, per scene
                frame = pose_frame(self._backend._scene_manager._scenes, path, request.frame_id)
                pose = self._backend.call("get_world_poses", prim_path=frame).inv() * pose

            response.pose = pose.to_ros()
```
and add before `def target_from_msg(msg: MotionTarget) -> Target:`
```python
def pose_frame(scenes, path: str, frame_id: str) -> str:
    """The prim whose frame a PoseRequest for `path` answers in (S2b Ruling 4), looked up in the
    scene `path` lies in: the scene root for frame_id "Scene_<i>", else the robot of that scene
    whose base_name (init.yaml, directly under the robot's path) is frame_id."""
    scene = path.split("/")[1]  # "/Scene_<i>/..." -> "Scene_<i>"
    if frame_id == scene:
        return f"/{scene}"
    robots = scenes[int(scene.removeprefix("Scene_"))]._config.get("robots", {})
    # ponytail: the first robot with this base_name; two robots of one scene sharing one need the robot's name in the request
    for robot in robots.values():
        if robot.get("base_name") == frame_id:
            return f"/{scene}{robot['path']}/{frame_id}"
    raise ValueError(f"{scene} has no robot with base_name '{frame_id}'")


```
In `guide_ex/guide_ex/steps/simulation/isaac/prim.py`, `GetPrimPose.run`: replace
```python
        request = PoseSrv.Request()
        request.path = scene_namespace + prim_path
```
with
```python
        request = PoseSrv.Request()
        request.path = scene_namespace + prim_path
        # Ruling 4: GUIDE answers in the frame this robot moves in (its base link), within its scene
        request.frame_id = robot.config.frame_id
```
and in its docstring replace `        Retrieves the current world pose of the specified primitive.` with
```python
        Retrieves the current pose of the specified primitive in the robot's frame_id
        (Ruling 4: its base link; GUIDE converts from the scene).
```
and, in its `Returns:` line, `the current world pose of the primitive` with `the primitive's current pose in the robot's frame_id`.

```bash
cd ~/ros2_ws && colcon build --packages-select guide_core guide_ex && source install/setup.zsh
cd ~/ros2_ws/src/guide && for t in guide_core/test guide_ex/test; do PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 ~/ros2_ws/.venv/bin/python -m pytest -q $t \
  --ignore-glob='*test_flake8.py' --ignore-glob='*test_pep257.py' --ignore-glob='*test_copyright.py' | tail -1; done
```
Expected: both `Finished`; `194 passed` (guide_core: 193 + 1), then `22 passed` (guide_ex: 21 + 1); checked on scratch copies.

- [ ] **Step 5: block_bin in the base frame**

In `guide_tasks/block_bin/block_bin/solve_task.py`:
- replace `from guide_ex.utility.pose import InvertPose, TransformPose` with `from guide_ex.utility.pose import TransformPose`;
- replace
```python
    # rest_pose is the only pose fed into the GUIDE-EX graph as context; every
    # other pose (scene/cube/bin/approach/retreat) is computed inside the node
    # tree (GetPrimPose / InvertPose / TransformPose), so no imperative pose
    # math lives here.
    rest_pose = PoseType(
        position=PointType([0.0, 0.0, 0.5]),
```
with
```python
    # rest_pose is the only pose fed into the GUIDE-EX graph as context; every
    # other pose (cube/bin/approach/retreat) is computed inside the node tree
    # (GetPrimPose / TransformPose), so no imperative pose math lives here.
    # All of them are in the robot's base frame, fr3_link0 (S2b Ruling 4), the frame
    # GetPrimPose answers in. The rest pose stays where the arm has always had it:
    # (0, 0, 0.5) of Scene_0 with the base at (-0.3, 0, 0) there, as MoveIt mounts it
    # (launch/bringup.launch.py) and as the base-frame datasets of 2026-10-10 end.
    rest_pose = PoseType(
        position=PointType([0.3, 0.0, 0.5]),
```
- delete the context line `        "scene_path": "",`;
- in `get_target_pose_sequence`, delete the map line `            "scene_path": "scene_path",` and the three children `GetPrimPose(alias="GetScenePose", …)`, `InvertPose(alias="InvertScenePose", …)` and `TransformPose(alias="TransformCubePose", …)` (GetPrimPose now answers in the base frame; the scene's inverse would move the cube back into the scene frame), and in its `output_map` delete `            "scene_pose": "scene_pose",` and `            "scene_pose_inv": "scene_pose_inv",`;
- in `pick_subtask`'s `dynamic_map`, delete `            "scene_path": "scene_path",`;
- in `place_subtask`'s `dynamic_map`, delete `            "scene_pose_inv": "scene_pose_inv",`, and delete its child `TransformPose(alias="TransformBinPose", …)`;
- in the `PickRecovery` node's `dynamic_map`, delete `                    "scene_path": "scene_path",`;
- in `main`, delete the `FR3RobotConfig` argument line `        frame_id=namespace_base.split("/")[-1] if namespace_base else "world",` (the config's default is the base link, `fr3_link0`).

Check it with a fake robot (the script stays in `/tmp`; Task 4 reruns it):
```bash
cat > /tmp/s2b_dry_run.py <<'PY'
"""Dry-run block_bin's tree with a fake robot: what GUIDE is asked, every move, every wait."""
import json
import logging
from types import SimpleNamespace

import block_bin.solve_task as st
from guide_core.types.geometry import Point, Pose
from guide_msgs.srv import Pose as PoseSrv
from irob_lerobot_ros.config import ActionType
from irob_lerobot_ros.ros2robot import ROS2Robot


class FakeRobot(ROS2Robot):
    def __del__(self):  # never connected
        pass


robot = FakeRobot.__new__(FakeRobot)
robot.node = SimpleNamespace(get_logger=lambda: logging.getLogger("dry"))
robot.config = SimpleNamespace(frame_id="fr3_link0", gripper_joint_names=["fr3_finger_joint1"],
                               arm_action_type=ActionType.CARTESIAN_POSE, gripper_action_type=ActionType.JOINT_POSITION)
robot._reentrant_callback_group = None
moves, asked, slept = [], [], []
robot.send_action = lambda action, **kwargs: moves.append(action) or True
robot.sleep = slept.append
for name in ("randomize", "pose", "collision", "is_success", "start_recording", "stop_recording", "set_prompt"):
    setattr(robot, name, SimpleNamespace(srv_name=name))
CUBE, BIN = [0.5, 0.1, 0.012], [0.45, -0.25, -0.013]  # in fr3_link0, where GUIDE now answers


def call(client, request=None, message=None, **kwargs):
    if client.srv_name == "randomize":
        return SimpleNamespace(success=True, message=json.dumps({"target": "/blocks/red_block", "goal": "/bin_0"}))
    if client.srv_name == "pose":
        asked.append((request.path, request.frame_id))
        return PoseSrv.Response(pose=Pose(Point(CUBE if "block" in request.path else BIN)).to_ros(), success=True)
    return SimpleNamespace(success=True, message="", collision=True)


robot.callService = call
st.time.sleep = lambda seconds: None  # a wall-clock wait left in the tree costs nothing here, and shows in `slept` by its absence
st.namespace_base = "/Sim_0/Scene_0"
assert st.solveTask(0, robot)
print("asked", asked)
print("moves", [tuple(float(round(v, 3)) for v in (m.position.x, m.position.y, m.position.z)) for m in moves if hasattr(m, "position")])
print("slept", slept)
PY
cd ~/ros2_ws/src/guide && grep -n "scene_path\|scene_pose\|InvertPose\|frame_id=namespace" guide_tasks/block_bin/block_bin/solve_task.py
cd ~/ros2_ws && colcon build --packages-select block_bin && source install/setup.zsh
~/ros2_ws/.venv/bin/python /tmp/s2b_dry_run.py 2>&1 | grep "^asked\|^moves\|^slept"
```
Expected: no grep match; `Finished <<< block_bin`; then (checked on scratch copies)
```
asked [('/Scene_0/blocks/red_block', 'fr3_link0'), ('/Scene_0/bin_0', 'fr3_link0')]
moves [(0.3, 0.0, 0.5), (0.3, 0.0, 0.4), (0.5, 0.1, 0.112), (0.5, 0.1, 0.022), (0.5, 0.1, 0.292), (0.45, -0.25, 0.137), (0.3, 0.0, 0.5)]
slept [1.0, 1.0, 1.0]
```
(rest, unclutch, cube approach / grasp / retreat, bin approach, rest: the poses the fake GUIDE gave plus the tree's offsets, all in `fr3_link0`; only the gripper settles wait on `robot.sleep` until Task 4).

- [ ] **Step 6: Commit**

```bash
cd ~/ros2_ws/src/guide
git add guide_msgs/srv/Pose.srv guide_core/guide_core/ros/guide_ros.py guide_core/test/test_command_fixes.py \
  guide_ex/guide_ex/steps/simulation/isaac/prim.py guide_ex/test/test_node_fixes.py guide_tasks/block_bin/block_bin/solve_task.py
git commit -m "feat(guide): PoseRequest answers in the requesting robot's base frame; block_bin in fr3_link0 (S2b Ruling 4)"
```

---

### Task 4: block_bin follows the switch; its waits run on sim time

**Files:**
- Modify: `guide_tasks/block_bin/launch/bringup.launch.py` (imports; `generate_nodes`)
- Modify: `guide_tasks/block_bin/block_bin/solve_task.py` (the post-Randomize pause; the five `WaitForSeconds`; `main`'s argument parser and `FR3RobotConfig`)
- Modify: `INSTALLATION.md` (§5, one bullet)

**Interfaces:**
- Consumes: Plan B's `robots.franka.controller` key in `config/init.yaml`; Task 1's `controller` field and `ROS2Robot.sleep`; GUIDE-EX's `WaitForSeconds.run(seconds, timer=time)` (unchanged: `timer` is anything with `.sleep(seconds)`); Task 3b's `/tmp/s2b_dry_run.py`.
- Produces: `solve_task --namespace /Sim_0/Scene_<i> --controller {moveit,oscbf}`; the launch starts move_group only for `controller: moveit`.

- [ ] **Step 1: The launch reads the switch**

In `guide_tasks/block_bin/launch/bringup.launch.py`: replace `import os` with
```python
import os

import yaml
```
replace
```python
    num_env = int(LaunchConfiguration("num_env").perform(context))
```
with
```python
    num_env = int(LaunchConfiguration("num_env").perform(context))

    # The motion layer: robots.franka.controller in this task's config/init.yaml, the switch GUIDE
    # reads when it registers the scene (S2b). oscbf drives the arm from Isaac's physics step, so
    # no move_group or ros2_control is started for it.
    with open(os.path.join(get_package_share_directory("block_bin"), "config", "init.yaml")) as f:
        controller = yaml.safe_load(f)["robots"]["franka"].get("controller", "moveit")
```
replace
```python
    move_groups = []
    for i in range(num_env):
```
with
```python
    move_groups = []
    for i in range(num_env if controller == "moveit" else 0):
```
and replace `                arguments=["--namespace", f"/Sim_0/Scene_{i}"],` with
```python
                arguments=["--namespace", f"/Sim_0/Scene_{i}", "--controller", controller],
```

- [ ] **Step 2: The solver passes it on and waits on sim time**

In `guide_tasks/block_bin/block_bin/solve_task.py`:
- replace
```python
    # Allow the (now scene-level, seeded) randomization to settle before solving.
    time.sleep(5)
```
with
```python
    # Allow the (now scene-level, seeded) randomization to settle before solving (sim time).
    robot.sleep(5)
```
- replace every `static_args={"seconds": ` (the five `WaitForSeconds` nodes, one Edit with replace-all) with `dynamic_map={"timer": "robot"}, static_args={"seconds": ` — `WaitForSeconds` then calls `robot.sleep(seconds)`; every composite holding one maps `robot` into its context already;
- replace `    argparser.add_argument("--namespace", type=str, default="")` with
```python
    argparser.add_argument("--namespace", type=str, default="")
    argparser.add_argument("--controller", choices=["moveit", "oscbf"], default="moveit")
```
- replace `        gripper_action_type=ActionType.JOINT_POSITION,` (inside `FR3RobotConfig(...)` in `main`) with
```python
        gripper_action_type=ActionType.JOINT_POSITION,
        controller=args[0].controller,
```

- [ ] **Step 3: Check and build**

```bash
cd ~/ros2_ws/src/guide && grep -c 'dynamic_map={"timer": "robot"}' guide_tasks/block_bin/block_bin/solve_task.py
grep -n "time.sleep" guide_tasks/block_bin/block_bin/solve_task.py
cd ~/ros2_ws && colcon build --packages-select block_bin && source install/setup.zsh
~/ros2_ws/.venv/bin/python /tmp/s2b_dry_run.py 2>&1 | grep "^slept"
```
Expected: `5`; `time.sleep` remains only at the post-connect pause (`time.sleep(5)  # Wait for connections to establish`) and the idle loop (`time.sleep(1)`), both outside episodes; `Finished <<< block_bin`; `slept [5, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 5.0]` (checked on a scratch copy): the post-Randomize pause, the three gripper settles and the four `WaitForSeconds` of a run without recovery (the fifth sits in `PickRecovery`), every one on `robot.sleep`.

- [ ] **Step 4: Document the switch**

In `INSTALLATION.md` §5 ("Generate a dataset"), add a bullet after the `Output:` bullet:
```markdown
- **Motion layer (S2b):** `robots.franka.controller` in `guide_tasks/block_bin/config/init.yaml` is
  `moveit` (move_group + ros2_control) or `oscbf` (guide_motion's OSCBF controller in Isaac's physics
  step; ROS2Robot streams `guide_msgs/MotionTarget` at 20 Hz). GUIDE and the block_bin bring-up both
  read it: rebuild block_bin after changing it, then register the scene again.
```

- [ ] **Step 5: Commit**

```bash
cd ~/ros2_ws/src/guide
git add guide_tasks/block_bin/launch/bringup.launch.py guide_tasks/block_bin/block_bin/solve_task.py INSTALLATION.md
git commit -m "feat(block_bin): follow robots.franka.controller; GUIDE-EX waits on sim time"
```

---

### Task 5: block_bin on OSCBF — 5 episodes, then MoveIt again

**Files:**
- Modify, temporarily (reverted in Step 7): `guide_tasks/block_bin/config/init.yaml`
- Create (results): `docs/design/s2b_block_bin_oscbf_run.json`

**Interfaces:**
- Consumes: everything above and Plan B.
- Produces: a 5-episode block_bin dataset under `~/dataset/block_bin/` (not committed), the run's numbers, and a MoveIt regression run.

- [ ] **Step 1: Switch block_bin to OSCBF**

Set `controller: oscbf` under `robots: franka:` in `guide_tasks/block_bin/config/init.yaml`, then:
```bash
cd ~/ros2_ws && colcon build --packages-select block_bin && source install/setup.zsh
pgrep -af "kit/kit|guide_core|solve_task"   # nothing
mkdir -p /tmp/s2b_run && export CYCLONEDDS_URI=file://$HOME/ros2_ws/install/guide_core/share/guide_core/config/cyclonedds_localhost.xml
```

- [ ] **Step 2: Start GUIDE and register the scene**

```bash
ros2 launch guide_core bringup.launch.py > /tmp/s2b_run/guide.log 2>&1 &
until grep -q "Running simulation loop" /tmp/s2b_run/guide.log; do sleep 5; done
ros2 service call /Sim_0/Register guide_msgs/srv/RegisterScene "{path: 'block_bin'}"
until grep -q "\[oscbf\] attached" /tmp/s2b_run/guide.log; do sleep 5; done
grep "\[oscbf\]" /tmp/s2b_run/guide.log
ros2 service call /Sim_0/PoseRequest guide_msgs/srv/Pose "{path: '/Scene_0', frame_id: 'fr3_link0'}" | tee /tmp/s2b_run/scene_in_base.txt
```
Expected:
- `success=True`, `message=''` (Plan B names a refused robot there);
- the `[oscbf]` lines: `fr3 torque: OSCBF step compiled in … s`; the wrist camera's bodies joining the articulation; `PhysX mass properties differ from the description (extra links, kg: …): recompiled in … s`; `attached: torque, controller every 2 substep(s), tensor view I/O` (Ruling 12); no `robot not registered`. Cold, RegisterScene took 34–37 s of wall time in Plan B's smoke run, and the attach recompile about 13.5 s more.
- the PoseRequest: the scene origin as the base sees it (Task 3b, live). Orientation within 0.5° of identity (|x|, |y|, |z| < 0.005); position (0.3, 0.0, 0.0) within 15 mm. (0.3, 0, 0) is the authored mount, (0.30669, 0.00004, −0.01307) the older measurement. A turned base, or an offset outside that range: stop and report it, since block_bin's rest pose and its z offsets assume a level base about there.

- [ ] **Step 3: Start the solver**

```bash
ros2 launch block_bin bringup.launch.py > /tmp/s2b_run/solver.log 2>&1 &
until grep -q "Ready to receive demonstration generation requests" /tmp/s2b_run/solver.log; do sleep 5; done
ros2 node list | grep -c move_group
```
Expected: `0` (no move_group under oscbf).

- [ ] **Step 4: Generate 5 episodes**

```bash
ros2 service call /Sim_0/Scene_0/generate_demonstration guide_msgs/srv/Demonstration "{path: '~/dataset/block_bin', zones: [], counts: [5]}"
until grep -q "Recording dataset finalized\|Error during generation" /tmp/s2b_run/solver.log; do sleep 30; done
grep -c "\[SUCCESS\]" /tmp/s2b_run/solver.log; grep -c "\[FAILURE\]" /tmp/s2b_run/solver.log
grep -c "OSCBF move timed out\|guide_motion planner\|refused" /tmp/s2b_run/solver.log
grep -c "QP infeasible\|refused" /tmp/s2b_run/guide.log
grep -c "dropped" /tmp/s2b_run/guide.log
```
Expected: `5` successes and `0` failures ("all successful": no attempt was discarded); the count of solver timeouts, planner errors or refused poses (report it; 0 expected); `0` host lines for QP infeasibility (Plan B warns at counts 1, 2, 4, … since attach: `QP infeasible or non-finite output (n since attach): damping + gravity`) or refusal (`… refused: <reason> …`); `0` dropped MotionTargets (frame, mode, joint count and finite values all accepted). A failure or timeout: read the episode's log lines around it (which step, which pose), stop, and report it with the pose instead of retuning gains here. Report the wall time per episode. Its sim time is at least the 11 s of recorded waits plus the moves at a cruise of speed × 0.5 m/s (0.1–0.5 m/s in block_bin). The torque controller at 60 Hz adds about 2.5 ms to each rendered frame (Ruling 12), and the recorder's three cameras keep RTF low.

- [ ] **Step 5: Check the waits ran on sim time, from the dataset**

```bash
cat > /tmp/s2b_run/still_frames.py <<'PY'
"""Per episode: frames, and the still frames at its end (tool moving < 0.5 mm per 0.1 s frame)."""
import glob, json, sys
import numpy as np, pandas as pd
root = sys.argv[1]
names = json.load(open(f"{root}/meta/info.json"))["features"]["observation.state"]["names"]
xyz = [names.index(k) for k in ("x", "y", "z")]  # the state is [x..wz, joint1..7, gripper], not init.yaml's order
df = pd.concat(pd.read_parquet(f) for f in sorted(glob.glob(f"{root}/data/*/*.parquet")))
result = {"dataset": root, "episodes": {}}
for episode, frames in df.groupby("episode_index"):
    p = np.stack(frames.sort_values("frame_index")["observation.state"].to_numpy())[:, xyz]
    moving = np.linalg.norm(np.diff(p, axis=0), axis=1) > 5e-4
    trailing = len(moving) - (np.flatnonzero(moving).max() + 1 if moving.any() else 0)
    result["episodes"][int(episode)] = {"frames": len(p), "trailing_still_frames": int(trailing)}
print(json.dumps(result, indent=1))
PY
cat > /tmp/s2b_run/ends.py <<'PY'
"""Per episode: the first and last recorded tool positions (state x, y, z: fr3_link0)."""
import glob, json, sys
import numpy as np, pandas as pd
root = sys.argv[1]
names = json.load(open(f"{root}/meta/info.json"))["features"]["observation.state"]["names"]
xyz = [names.index(k) for k in ("x", "y", "z")]
df = pd.concat(pd.read_parquet(f) for f in sorted(glob.glob(f"{root}/data/*/*.parquet")))
for episode, frames in df.groupby("episode_index"):
    p = np.stack(frames.sort_values("frame_index")["observation.state"].to_numpy())[:, xyz]
    print(int(episode), np.round(p[0], 3).tolist(), np.round(p[-1], 3).tolist())
PY
~/ros2_ws/.venv/bin/python /tmp/s2b_run/still_frames.py "$(ls -dt ~/dataset/block_bin/dataset_* | head -1)" | tee /tmp/s2b_run/dataset_check.json
~/ros2_ws/.venv/bin/python /tmp/s2b_run/ends.py "$(ls -dt ~/dataset/block_bin/dataset_* | head -1)"
~/ros2_ws/.venv/bin/python /tmp/s2b_run/still_frames.py ~/dataset/block_bin/dataset_0_0_2026_10_10_15_44_02 | grep trailing
```
Expected: 5 episodes, each with `trailing_still_frames` ≥ 48. Return's `WaitForSeconds(5.0)` ends every episode: 50 frames at 10 Hz when it waits on sim time (spec question 5). The second command reads a dataset recorded before this change, whose wait ran on the wall clock: its episodes end 6–11 still frames after the last move (5 s of wall time at RTF ≈ 0.15–0.2; measured while writing this plan). If a new episode ends with a short tail like that, a wait still runs on wall time: find it, and report it. `ends.py`: every episode starts at the unclutch pose and ends at rest, (0.3, 0.0, 0.4) and (0.3, 0.0, 0.5) within 5 mm, as the 2026-10-10 MoveIt datasets do (measured: (0.300, 0.000, 0.400) and (0.300, 0.000, 0.500)).

- [ ] **Step 6: Stop**

```bash
pkill -INT -f solve_task; pkill -INT -f "ros2 launch block_bin"; sleep 5
pkill -INT -f guide_core; sleep 15; pgrep -af "kit/kit|guide_core|solve_task"   # nothing left
```

- [ ] **Step 7: Back to MoveIt, and a regression run**

Set `controller: moveit` back in `guide_tasks/block_bin/config/init.yaml`, check it matches the last commit, rebuild, and run 2 episodes on MoveIt the same way (Steps 2–4 with `counts: [2]`, GUIDE's log to `/tmp/s2b_run/guide_moveit.log`, the solver's to `/tmp/s2b_run/solver_moveit.log`):
```bash
cd ~/ros2_ws/src/guide && git diff --stat guide_tasks/block_bin/config/init.yaml   # empty
cd ~/ros2_ws && colcon build --packages-select block_bin && source install/setup.zsh
```
Expected: no `[oscbf]` line in `guide_moveit.log`, `move_group` present (`ros2 node list | grep -c move_group` ≥ 1), and `2` `[SUCCESS]` lines in `solver_moveit.log`: the seam moved MoveIt's speed and straight-line check without changing them, and MoveIt plans the base-frame poses (Task 3b) in `fr3_link0`. `still_frames.py` on that run's dataset also shows ≥ 48 still frames per episode (the waits run on sim time under MoveIt too), and `ends.py` the same start and end as Step 5. Then stop everything as in Step 6.

- [ ] **Step 8: Record and commit**

```bash
cd ~/ros2_ws && .venv/bin/python - <<'EOF'
import json
from pathlib import Path
run = json.loads(Path("/tmp/s2b_run/dataset_check.json").read_text())
log = Path("/tmp/s2b_run/solver.log").read_text()
run.update(controller="oscbf", successes=log.count("[SUCCESS]"), failures=log.count("[FAILURE]"),
           timeouts=log.count("OSCBF move timed out"), planner_errors=log.count("guide_motion planner"),
           moveit_regression_successes=Path("/tmp/s2b_run/solver_moveit.log").read_text().count("[SUCCESS]"),
           scene_in_base=Path("/tmp/s2b_run/scene_in_base.txt").read_text().strip())
Path("src/guide/docs/design/s2b_block_bin_oscbf_run.json").write_text(json.dumps(run, indent=1))
print({k: v for k, v in run.items() if k != "episodes"})
EOF
cd ~/ros2_ws/src/guide
git add docs/design/s2b_block_bin_oscbf_run.json
git commit -m "test(block_bin): 5 episodes on controller: oscbf, MoveIt regression run"
git status --porcelain   # only guide_core/config/init.yaml ' M'
```
The committed default stays `controller: moveit`: switching block_bin to OSCBF for good is the user's decision once S2b is accepted.

---

## Self-review (run by the plan's author)

- **Spec coverage.** 4.5 configuration → Task 1 (`controller`, `motion_robot`); `send_action(Pose | Twist | joints, wait_for_execution)`, planning beyond one step at GUIDE-EX's speed (Ruling 3), 20 Hz streaming on sim time, completion test, timeout, True/False → Task 2; gripper via `gripper_width`, set on every message → Task 2 (`_width`, Plan B's host); observation features → Task 1. 4.6 seam → Task 3; sim-time waits (`WaitForSeconds`, `SetGripperState`'s settle, `solve_task`'s post-Randomize pause) → Tasks 1 (`sleep`), 3 and 4. Ruling 4 → Task 2 (refuses a non-base pose) and Task 3b (GUIDE converts; block_bin in `fr3_link0`). Reset (I2/N1) → Task 2 (`hold_off`, `reset_scene`). Section 7 → Task 5 (5 episodes, all successful; frames show sim-time waits). Section 8 "`send_action` timeout or planner failure returns False" → Task 2 (`PlanningError`, `_wait_reached`); "GUIDE-EX retries as today" → unchanged 3-try loops (Task 3).
- **Placeholders.** None: every edit names its old text and gives the new code; every run its command and expected output.
- **Type consistency.** `send_action(action, cartesian, wait_for_execution, speed, min_fraction)` is the same in Tasks 1–3; `plan(model, goal, q, speed)`, `Target`, `PlanningError`, `Model.ee_pose/ee_speed/base_link/arm_joints` match Plan A; `MotionTarget` fields and `MODE_*` match Plan B; the topic `motion_target` in the robot's namespace (`/Sim_0/Scene_0/franka`) matches Plan B's `host.topic`; `Pose.srv`'s `frame_id` is read by `guide_ros` and set by `GetPrimPose` (Task 3b).
- **Review Focus.** Each line has its test in Task 1, 2 or 3b; line 3 is also checked live in Task 5 Step 2, line 5 by Task 5's MoveIt run.
- **Pre-flight (2026-10-20).** Tasks 1, 2, 3 and 3b were run on scratch copies (irob_lerobot_ros at 6ce7461, guide_ex, guide_core and block_bin at this branch, guide_msgs built in a scratch workspace) with the RED and GREEN results quoted in their steps; Task 4's edits and the dry run too.
