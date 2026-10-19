# S2b Plan B — `MotionTarget` and the OSCBF host in Isaac's physics step

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** a robot whose task `init.yaml` says `controller: oscbf` is driven, in torque or velocity mode, by `guide_motion`'s OSCBF controller inside Isaac's `PHYSICS_PRE_STEP` callback, fed with `guide_msgs/MotionTarget`s that GUIDE's ROS node receives at depth 1, and validated headless against the spec's hold, path, determinism and RTF thresholds.

**Architecture:** `guide_msgs/msg/MotionTarget.msg` is the 20 Hz target topic. `guide_core/guide_core/control/oscbf_host.py` (`OscbfHost`, Isaac-bound, ROS-free) owns one robot. At registration it loads the `guide_motion` description and compiles the controller. At the first substep after each play it reads the masses from PhysX, checks M and g, and switches the arm's drives. On every substep it steps the controller through the articulation's tensor view, falling back to the experimental API. `_cmd_register_scene` creates the hosts and keeps the oscbf robots' joint-state graph from subscribing `joint_command`. GUIDE's ROS node subscribes each host's `motion_target` and writes its single-slot holder. A manual headless script under `guide_core/test` measures the result.

**Tech Stack:** Isaac Sim 6.0.1 (`isaacsim.core.simulation_manager.SimulationManager`, `isaacsim.core.experimental.prims.Articulation`, warp), ROS 2 Jazzy rclpy, `guide_motion` (Plan A), numpy/scipy, pytest 9 with Isaac stubbed.

**Spec:** `docs/superpowers/specs/2026-10-19-s2b-native-motion-design.md`, sections 3, 4.3, 4.4, 5, 6, 7 (headless Isaac integration) and 8.

**Depends on:** Plan A (`2026-10-19-s2b-plan-a-guide-motion.md`) complete: `guide_motion` installed in `.venv`; `guide_motion/guide_motion/robots/{fr3,panda}` committed. Plan C depends on this plan.

**Sources:** the measurement that already ran OSCBF in this callback: `/home/user/.claude/jobs/d2413306/tmp/oscbf_isaac/run.py` (callback registration, `switch_dof_control_mode`, `arm._physics_articulation_view` raw I/O with `wp.from_numpy(..., device=...)` and a `wp.int32` index array, mass-property getters, warm-up) and its `report.md`; GUIDE's command/ROS layout in `/home/user/.claude/jobs/d2413306/tmp/s2b-facts.md`.

## Spec questions (recorded; the plan follows the reading shown)

1. **"At registration: … read the mass properties … call `Controller.warm_up()`."** PhysX's mass properties exist only after play, and the compiled step bakes the masses in. Reading: the host compiles at registration with the committed description, whose inertials are PhysX's own for the bare asset (Plan A). At the first substep after play it reads PhysX and recompiles only when the scene differs (for example block_bin's wrist camera). That one-off stall comes right after registration, never mid-episode, and is logged.
2. **"Description validation fails → refuse to register the robot."** Reading: at that first substep the host compares the controller's M and g with PhysX's (1e-3 relative, as in 4.2.5). On failure it never switches the drives (the robot stays position-driven), never commands it, and names the check in `diagnostics["refused"]` and the log. PhysX links the description lacks (a camera mount) are first folded into their parent as fixed links instead of failing.
3. **"No target for 0.25 s of sim time."** Reading: the host measures it on its own physics clock (the sum of substep `dt`s), from the moment a new message reaches the holder. It does not compare the message stamp, which the solver sets from `/clock` (one rendered frame coarse).
4. **`rate`.** Reading: `oscbf.rate` in Hz, and the controller runs every `round(physics_hz / rate)` substeps. Torque defaults to every substep (`rate: 120`), velocity to `rate: 60` (once per rendered frame). Between velocity steps the targets persist; between torque steps the last efforts are written again.
5. **The joint-state graph** keeps publishing `joint_states` (spec 3) but no longer subscribes `joint_command` for an oscbf robot. Otherwise two writers would drive one arm.
6. **Reset homing.** block_bin's `config/reset.yaml` homes the arm through drive `targetPosition`s, which an effort- or velocity-driven arm ignores. This is documented in `init.yaml` and not changed here. block_bin's generation does not call Reset (it moves home with GUIDE-EX), but an evaluation tool that relies on Reset does not get a homed arm under `controller: oscbf`.

## Global Constraints

Copied from the spec:
- "ROS 2 stays the transport between processes; the solver is Isaac-free." "No ROS message crosses per substep, and nothing is rendered for control." "States return over the existing per-frame `/joint_states`."
- "Configuration: enabled per robot in the task's `init.yaml` (`robots.<name>.controller: oscbf`, with `mode`, `gains` and `rate`)."
- "switch the 7 arm DOFs' control mode: effort (stiffness and damping 0) for torque, velocity for velocity mode; the gripper stays position-driven; set up the JAX persistent compilation cache (`~/.cache/guide_motion/jax`); call `Controller.warm_up()`."
- "Each physics substep (`SimulationManager.register_callback(..., PHYSICS_PRE_STEP)`): read q and qd; read the latest target from a single-slot holder; call `step`; write efforts or velocity targets."
- "torque mode runs every substep (120 Hz); velocity mode runs every second substep (once per rendered frame)."
- "I/O goes through the articulation's tensor view. … If the private attribute is missing in a future Isaac, it falls back to the API automatically."
- "Targets arrive on GUIDE's existing ROS thread: subscriber depth 1, `KEEP_LAST`; the slot is written with one lock-protected assignment."
- `MotionTarget.msg` exactly as in spec 4.4; "The topic is `/Sim_0/Scene_<i>/<robot>/motion_target`, published at 20 Hz."
- Section 6 rules: "no per-substep ROS traffic; the planner runs only on new goals; JIT warm-up and persistent cache, so there are no stalls mid-episode; physics stays at 120 Hz; an RTF regression check is part of the tests."
- Section 7: "Panda and FR3: hold the pose for 5 s (≤ 0.1 mm); the 20 Hz square (path deviation ≤ 3 mm torque, ≤ 2 mm velocity); no NaN, torque ≤ limits. RTF: the frame time at 120 Hz with the controller ≤ awake baseline + 2.5 ms (torque) and + 1.2 ms (velocity). Same seed twice gives the same joint trajectory, max difference ≤ 1e-6 rad."
- Section 8: "No target for 0.25 s of sim time: hold the current pose (torque: gravity compensation + hold; velocity: zero velocity) and set a stale flag in the host's diagnostics"; "Description validation fails: refuse to register the robot, with the failing check named."
- "GUIDE still uses `isaacsim.core.api`, deprecated since 6.0.0. The host adapter uses the non-deprecated `SimulationManager` and experimental `Articulation`."

Process (this workspace):
- Work in `~/ros2_ws/src/guide` on branch `feat/libero-native-motion`. Commits authored by the repo's configured user only: **no `Co-Authored-By`, no Claude/Generated-with lines.** Never push, never open PRs.
- Never stage `guide_core/config/init.yaml`; never run `git checkout --`, `git restore`, `git reset --hard`, `git stash` or `git clean`. Stage files by name.
- guide_core, guide_msgs and block_bin are colcon **copy** installs: after editing them, `cd ~/ros2_ws && source /opt/ros/jazzy/setup.zsh && colcon build --packages-select <pkg> && source install/setup.zsh` (zsh: the `.zsh` setup files).
- Tests: from a sourced shell, `cd ~/ros2_ws/src/guide && PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 ~/ros2_ws/.venv/bin/python -m pytest -q guide_core/test --ignore-glob='*test_flake8.py' --ignore-glob='*test_pep257.py' --ignore-glob='*test_copyright.py'`.
- Isaac: headless through `~/ros2_ws/.venv/bin/python`, `OMNI_KIT_ACCEPT_EULA=YES`, `env -u CUDA_DEVICE_ORDER`; `pgrep -af "isaac|kit/kit|guide_core"` must print nothing before a launch. Shell `export CYCLONEDDS_URI=file://$HOME/ros2_ws/install/guide_core/share/guide_core/config/cyclonedds_localhost.xml` for every shell that talks to GUIDE.
- Never upload anything to Hugging Face.

## Review Focus

1. **A non-finite target from a buggy publisher** (NaN in a pose or joint field). Expected: dropped at the ROS boundary with a throttled warning; the holder keeps the last good target. Pinned in Task 3 (`test_motion_targets_are_subscribed_depth_one_and_converted`).
2. **A gripper width outside the hand** (0.2 m, or negative). Expected: the finger target is clamped to the finger joint's PhysX range. Pinned in Task 2 (`test_a_gripper_width_outside_the_hand_is_clamped`).
3. **GPU physics** (`physics_device: cuda:N` in `guide_core/config/init.yaml`): the tensor view's arrays live on the GPU. Expected: the raw writes use the view's device, not `"cpu"`. Pinned in Task 2 (`test_tensor_view_io_uses_the_views_device`).
4. **A second `RegisterScene`** (GUIDE stops and plays the timeline, and PhysX rebuilds the articulation). Expected: every existing host re-attaches and switches its drives again, without recompiling when the masses did not change. Pinned in Task 2 (`test_a_stop_attaches_again_on_the_next_play_without_recompiling`).
5. **A target that arrives between play and the first substep** (the solver starts publishing while the host has not attached yet). Expected: kept in the holder and used at the first substep. Pinned in Task 2 (`test_a_target_sent_before_the_first_substep_is_used`).

---

## File Structure

```
guide_msgs/msg/MotionTarget.msg                 new: the 20 Hz target (spec 4.4)
guide_msgs/CMakeLists.txt, package.xml          generate it; builtin_interfaces dependency
guide_core/guide_core/control/oscbf_host.py     new: OscbfHost (Isaac-bound, ROS-free)
guide_core/guide_core/core/commands/_cmd_robot.py      joint-state graph: optional joint_command subscriber
guide_core/guide_core/core/commands/_cmd_simulator.py  register_scene: an OscbfHost per oscbf robot
guide_core/guide_core/core/guide_simulator.py   motion_hosts(scene_id) for the ROS side
guide_core/guide_core/scene/scene_orchestrator.py      robot dicts carry controller + oscbf settings
guide_core/guide_core/ros/guide_ros.py          depth-1 motion_target subscription, MotionTarget -> Target
guide_tasks/block_bin/config/init.yaml          the documented switch (controller: moveit stays the default)
guide_core/test/test_oscbf_host.py              new: host logic against a fake articulation
guide_core/test/test_command_fixes.py           + wiring tests
guide_core/test/isaac_oscbf_check.py            new: the manual headless checks (not collected by pytest)
docs/design/s2b_oscbf_isaac_checks.json         new: the checks' results
```

`guide_core/guide_core/control/` has no `__init__.py`, like `core/` and `scene/`: guide_core's `setup.py` uses `find_namespace_packages`.

---

### Task 1: `MotionTarget.msg`

**Files:**
- Create: `guide_msgs/msg/MotionTarget.msg`
- Modify: `guide_msgs/CMakeLists.txt`, `guide_msgs/package.xml`

**Interfaces:**
- Consumes: nothing.
- Produces: `guide_msgs.msg.MotionTarget` with `MODE_POSE = 0`, `MODE_TWIST = 1`, `MODE_JOINTS = 2` (equal to `guide_motion.targets.POSE/TWIST/JOINTS`), fields `stamp: builtin_interfaces/Time`, `mode: uint8`, `frame_id: string`, `pose: geometry_msgs/Pose`, `twist: geometry_msgs/Twist`, `joints: float64[]`, `gripper_width: float64`.

- [ ] **Step 1: Write the failing check**

Run (sourced shell):
```bash
~/ros2_ws/.venv/bin/python -c "from guide_msgs.msg import MotionTarget; m = MotionTarget(); assert (m.MODE_POSE, m.MODE_TWIST, m.MODE_JOINTS) == (0, 1, 2); print('ok')"
```
Expected: `ModuleNotFoundError: No module named 'guide_msgs.msg'`.

- [ ] **Step 2: Add the message**

`guide_msgs/msg/MotionTarget.msg`:
```text
# One motion target for a robot's controller (S2b), published at 20 Hz on
# /Sim_<s>/Scene_<i>/<robot>/motion_target. Metric everywhere: metres, radians.
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

In `guide_msgs/CMakeLists.txt` replace
```cmake
find_package(geometry_msgs REQUIRED)
find_package(rosidl_default_generators REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  # Messages

```
with
```cmake
find_package(builtin_interfaces REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(rosidl_default_generators REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  # Messages
  "msg/MotionTarget.msg"

```
and `  DEPENDENCIES geometry_msgs` with `  DEPENDENCIES builtin_interfaces geometry_msgs`.

In `guide_msgs/package.xml`, before `  <depend>geometry_msgs</depend>` add `  <depend>builtin_interfaces</depend>`.

- [ ] **Step 3: Build and check**

```bash
cd ~/ros2_ws && source /opt/ros/jazzy/setup.zsh && colcon build --packages-select guide_msgs && source install/setup.zsh
ros2 interface show guide_msgs/msg/MotionTarget | head -12
~/ros2_ws/.venv/bin/python -c "from guide_msgs.msg import MotionTarget; m = MotionTarget(); assert (m.MODE_POSE, m.MODE_TWIST, m.MODE_JOINTS) == (0, 1, 2); print('ok')"
```
Expected: `Finished <<< guide_msgs`; the interface text above; `ok`.

- [ ] **Step 4: Commit**

```bash
cd ~/ros2_ws/src/guide
git add guide_msgs/msg/MotionTarget.msg guide_msgs/CMakeLists.txt guide_msgs/package.xml
git commit -m "feat(guide_msgs): MotionTarget, the 20 Hz target of the OSCBF controller"
```

---

### Task 2: `OscbfHost`

**Files:**
- Create: `guide_core/guide_core/control/oscbf_host.py`
- Test: `guide_core/test/test_oscbf_host.py`

**Interfaces:**
- Consumes (Plan A): `guide_motion.model.Model.load(robot)` (`arm_joints`, `gripper_joint`, `ee_pose`, `usd_to_urdf`, `manipulator`, `set_mass_properties(masses, coms, inertias, extra_links) -> bool`, `home`, `lower`, `upper`); `guide_motion.control.Controller(model, mode, gains, barriers, q_rest)` with `warm_up() -> float`, `step(target, q, qd) -> ndarray(7)`, `events`; `guide_motion.control.enable_compilation_cache()`; `guide_motion.targets.Target`, `TargetHolder`. Isaac: `Articulation(path)` (`dof_names`, `link_names`, `get_dof_limits()`, `get_link_masses()`, `get_link_coms()`, `get_link_inertias()`, `get_mass_matrices()`, `get_dof_gravity_compensation_forces()`, `get_dof_positions()`, `get_dof_velocities()`, `set_dof_efforts`, `set_dof_velocity_targets`, `set_dof_position_targets`, `switch_dof_control_mode`, private `_physics_articulation_view` with `get_dof_positions/get_dof_velocities/set_dof_actuation_forces/set_dof_velocity_targets/get_link_transforms/shared_metatype`); `SimulationManager.register_callback(fn, SimulationEvent.PHYSICS_PRE_STEP | SIMULATION_STOPPED)`.
- Produces `guide_core.control.oscbf_host.OscbfHost(articulation_root: str, robot: str = "fr3", mode: str = "torque", gains: dict | None = None, rate: float | None = None, barriers: dict | None = None, q_rest: list | None = None, logger=None)` with public `holder: TargetHolder` (the ROS side's `put`), `model: Model`, `mode`, `diagnostics = {"stale": bool, "infeasible": int, "refused": str | None}`, `every: int` (after attach), `close()`. The caller must keep the host referenced: `register_callback` holds bound methods by weak reference.

- [ ] **Step 1: Write the failing tests**

`guide_core/test/test_oscbf_host.py`:
```python
"""OscbfHost's per-substep logic against a fake articulation, with Isaac stubbed out."""

import importlib
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

from guide_motion import description
from guide_motion.model import Model
from guide_motion.targets import POSE, Target

ROBOT = "fr3"
STUBS = ("isaacsim", "isaacsim.core", "isaacsim.core.experimental", "isaacsim.core.experimental.prims",
         "isaacsim.core.simulation_manager")
DT = 1.0 / 120.0


class Arr:
    """What the experimental Articulation's getters return: .numpy() of shape (1, ...)."""

    def __init__(self, a):
        self.a = np.asarray(a, np.float32)[None]
        self.shape, self.device = self.a.shape, "cpu"

    def numpy(self):
        return self.a


def usd_frame_mass_properties(model):
    """The description's URDF inertials as PhysX reports them: in USD body frames."""
    import xml.etree.ElementTree as ET

    raw = {}
    for link in ET.fromstring(model.urdf_text).iter("link"):
        inertial = link.find("inertial")
        if inertial is None or link.get("name") not in model.usd_to_urdf:
            continue
        i = {k: float(v) for k, v in inertial.find("inertia").attrib.items()}
        raw[link.get("name")] = (float(inertial.find("mass").get("value")), np.array(inertial.find("origin").get("xyz").split(), float),
                                 np.array([[i["ixx"], i["ixy"], i["ixz"]], [i["ixy"], i["iyy"], i["iyz"]], [i["ixz"], i["iyz"], i["izz"]]]))
    moved = description.to_urdf_frames({n: v[0] for n, v in raw.items()}, {n: v[1] for n, v in raw.items()},
                                       {n: v[2] for n, v in raw.items()}, {n: np.linalg.inv(model.usd_to_urdf[n]) for n in raw})
    return moved


class FakeArm:
    def __init__(self, model, q, scale_M=1.0):
        self.model = model
        self.dof_names = model.arm_joints + [model.gripper_joint, model.gripper_joint.replace("1", "2")]
        self.props = usd_frame_mass_properties(model)
        self.link_names = list(self.props)
        self.q = np.r_[q, 0.0, 0.0]
        self.scale_M = scale_M
        self.efforts, self.velocity_targets, self.finger_targets, self.modes = [], [], [], []

    def get_link_masses(self):
        return Arr([self.props[n][0] for n in self.link_names])

    def get_link_coms(self):
        return Arr([self.props[n][1] for n in self.link_names]), None

    def get_link_inertias(self):
        return Arr([self.props[n][2].reshape(9) for n in self.link_names])

    def get_mass_matrices(self):
        M = np.zeros((9, 9))
        M[:7, :7] = self.scale_M * np.asarray(self.model.manipulator.mass_matrix(self.q[:7]))
        return Arr(M)

    def get_dof_gravity_compensation_forces(self):
        return Arr(np.r_[np.asarray(self.model.manipulator.gravity_vector(self.q[:7])), 0, 0])

    def get_dof_positions(self):
        return Arr(self.q)

    def get_dof_limits(self):
        lower, upper = np.r_[self.model.lower, 0.0, 0.0], np.r_[self.model.upper, 0.04, 0.04]
        return Arr(lower), Arr(upper)

    def get_dof_velocities(self):
        return Arr(np.zeros(9))

    def switch_dof_control_mode(self, mode, dof_indices):
        self.modes.append((mode, list(dof_indices)))

    def set_dof_efforts(self, u, dof_indices):
        self.efforts.append(np.array(u[0]))

    def set_dof_velocity_targets(self, u, dof_indices):
        self.velocity_targets.append(np.array(u[0]))

    def set_dof_position_targets(self, x, dof_indices):
        self.finger_targets.append((float(x[0][0]), list(dof_indices)))


class FakeController:
    made = 0

    def __init__(self, model, mode, *args):
        self.model, self.mode, self.events, self.targets = model, mode, {"infeasible": 0}, []
        FakeController.made += 1

    def warm_up(self):
        return 0.0

    def step(self, target, q, qd):
        self.targets.append(target)
        return np.full(7, 0.5)


@pytest.fixture
def host_module(monkeypatch):
    before = dict(sys.modules)
    for name in STUBS:
        sys.modules[name] = MagicMock()
    sys.modules.pop("guide_core.control.oscbf_host", None)
    module = importlib.import_module("guide_core.control.oscbf_host")
    monkeypatch.setattr(module, "Controller", FakeController)
    monkeypatch.setattr(module, "enable_compilation_cache", lambda: None)
    yield module
    for name in set(sys.modules) - set(before):
        if name.startswith(("isaacsim", "guide_core.control")):
            del sys.modules[name]
    sys.modules.update(before)


def make_host(module, monkeypatch, mode="torque", rate=None, **arm):
    model = Model.load(ROBOT)
    fake = FakeArm(Model.load(ROBOT), model.home, **arm)
    monkeypatch.setattr(module, "Articulation", lambda path: fake)
    host = module.OscbfHost("/Scene_0/fr3", robot=ROBOT, mode=mode, rate=rate)
    return host, fake


def run(host, n):
    for _ in range(n):
        host._on_step(DT, None)


def test_no_target_holds_the_pose_with_the_arm_in_effort_mode(host_module, monkeypatch):
    host, arm = make_host(host_module, monkeypatch)
    run(host, 3)
    assert arm.modes == [("effort", list(range(7)))]
    assert host.diagnostics["stale"] and len(arm.efforts) == 3
    held = host._controller.targets
    assert held[0].mode == POSE and all(t is held[0] for t in held)
    np.testing.assert_allclose(held[0].position, host.model.ee_pose(host.model.home)[0], atol=1e-6)


def test_a_fresh_target_is_used_until_it_goes_stale(host_module, monkeypatch):
    host, arm = make_host(host_module, monkeypatch)
    target = Target(position=np.array([0.4, 0.0, 0.4]))
    host.holder.put(target)
    run(host, 30)  # 0.25 s
    assert host._controller.targets[-1] is target and not host.diagnostics["stale"]
    run(host, 2)
    assert host.diagnostics["stale"] and host._controller.targets[-1] is not target


def test_velocity_mode_runs_once_per_frame_and_stops_when_stale(host_module, monkeypatch):
    host, arm = make_host(host_module, monkeypatch, mode="velocity", rate=60)
    host.holder.put(Target(position=np.array([0.4, 0.0, 0.4])))
    run(host, 20)
    assert arm.modes == [("velocity", list(range(7)))] and host.every == 2
    assert len(host._controller.targets) == 10 and len(arm.velocity_targets) == 10
    run(host, 40)
    assert host.diagnostics["stale"] and np.all(arm.velocity_targets[-1] == 0)


def test_gripper_width_is_one_finger_target_written_once(host_module, monkeypatch):
    host, arm = make_host(host_module, monkeypatch)
    host.holder.put(Target(position=np.array([0.4, 0.0, 0.4]), gripper_width=0.08))
    run(host, 5)
    assert arm.finger_targets == [(pytest.approx(0.04), [7])]


def test_a_dynamics_mismatch_refuses_the_robot(host_module, monkeypatch):
    host, arm = make_host(host_module, monkeypatch, scale_M=1.05)
    run(host, 5)
    assert "dynamics check failed" in host.diagnostics["refused"]
    assert arm.modes == [] and arm.efforts == []


def test_a_stop_attaches_again_on_the_next_play_without_recompiling(host_module, monkeypatch):
    host, arm = make_host(host_module, monkeypatch)
    run(host, 2)
    made = FakeController.made
    host._on_stop(None)
    run(host, 2)
    assert len(arm.modes) == 2 and FakeController.made == made  # same masses: same compiled step


def test_a_target_sent_before_the_first_substep_is_used(host_module, monkeypatch):
    host, arm = make_host(host_module, monkeypatch)
    target = Target(position=np.array([0.4, 0.0, 0.4]))
    host.holder.put(target)
    run(host, 1)
    assert host._controller.targets == [target]


def test_a_gripper_width_outside_the_hand_is_clamped(host_module, monkeypatch):
    host, arm = make_host(host_module, monkeypatch)
    host.holder.put(Target(position=np.array([0.4, 0.0, 0.4]), gripper_width=0.2))
    run(host, 2)
    assert arm.finger_targets == [(pytest.approx(0.04), [7])]


def test_tensor_view_io_uses_the_views_device(host_module, monkeypatch):
    writes = []
    fake_wp = SimpleNamespace(int32="int32", float32="float32",
                              array=lambda data, dtype, device: ("index", device),
                              from_numpy=lambda data, dtype, device: writes.append(device) or ("data", device))
    monkeypatch.setattr(host_module, "wp", fake_wp)
    host, arm = make_host(host_module, monkeypatch)
    on_gpu = SimpleNamespace(shape=(1, 9), device="cuda:0", numpy=lambda: arm.get_dof_positions().numpy())
    arm._physics_articulation_view = SimpleNamespace(
        get_dof_positions=lambda: on_gpu,
        get_dof_velocities=arm.get_dof_velocities,
        set_dof_actuation_forces=lambda data, index: writes.append(("forces", data, index)),
    )
    run(host, 3)
    assert writes[0] == "cuda:0" and writes[1] == ("forces", ("data", "cuda:0"), ("index", "cuda:0"))
    assert arm.efforts == []  # the API setter was not used


def test_an_extra_physx_link_folds_into_its_parent(host_module):
    host = host_module.OscbfHost.__new__(host_module.OscbfHost)
    host.model = Model.load(ROBOT)
    names = ["fr3_hand", "camera"]
    T_camera = np.eye(4)
    T_camera[:3, 3] = [0.0, 0.0, 0.05]
    host._view = SimpleNamespace(
        shared_metatype=SimpleNamespace(link_names=names, link_parents=["fr3_link7", "fr3_hand"]),
        get_link_transforms=lambda: Arr([[0, 0, 0, 0, 0, 0, 1], [0, 0, 0.05, 0, 0, 0, 1]]),
    )
    (parent, T), = host._extra_links(names).values()
    assert parent == "fr3_hand"
    np.testing.assert_allclose(T, host.model.usd_to_urdf["fr3_hand"] @ T_camera, atol=1e-7)
```

- [ ] **Step 2: Run them to see them fail**

Run (sourced shell): `cd ~/ros2_ws/src/guide && PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 ~/ros2_ws/.venv/bin/python -m pytest -q guide_core/test/test_oscbf_host.py`
Expected: 10 errors, `ModuleNotFoundError: No module named 'guide_core.control'`. (pytest imports the source tree through `guide_core/test/conftest.py`, so no build is needed for the tests.)

- [ ] **Step 3: Implement**

`guide_core/guide_core/control/oscbf_host.py`:
```python
"""guide_motion's OSCBF controller inside Isaac's physics step, for one robot (S2b spec 4.3).

Before each physics substep (SimulationManager PHYSICS_PRE_STEP, main thread): read q and
qd, take the latest target from a single-slot holder that GUIDE's ROS thread writes, step
the controller, write joint efforts (torque) or velocity targets (velocity). Nothing goes
over ROS per substep; states return over the existing per-frame /joint_states.
"""

import numpy as np
import warp as wp
from isaacsim.core.experimental.prims import Articulation
from isaacsim.core.simulation_manager import SimulationEvent, SimulationManager
from scipy.spatial.transform import Rotation

from guide_motion.control import Controller, enable_compilation_cache
from guide_motion.model import Model
from guide_motion.targets import Target, TargetHolder

DYNAMICS_TOL = 1e-3  # controller M and g vs PhysX, relative (S2b spec 4.2.5)


class OscbfHost:
    def __init__(self, articulation_root: str, robot: str = "fr3", mode: str = "torque", gains=None,
                 rate=None, barriers=None, q_rest=None, logger=None):
        self.holder = TargetHolder(stale_after=0.25)
        self.mode, self.rate, self._logger = mode, rate, logger
        self.diagnostics = {"stale": True, "infeasible": 0, "refused": None}
        self.model = Model.load(robot)
        self._settings = (mode, gains, barriers, q_rest)
        enable_compilation_cache()
        self._controller = Controller(self.model, *self._settings)
        self._log("info", f"{robot} {mode}: OSCBF step compiled in {self._controller.warm_up():.1f} s")
        self._arm = Articulation(articulation_root)
        self._attached, self._t, self._k = False, 0.0, 0
        self._u, self._hold, self._width = None, None, None
        self._ids = [
            SimulationManager.register_callback(self._on_step, SimulationEvent.PHYSICS_PRE_STEP),
            SimulationManager.register_callback(self._on_stop, SimulationEvent.SIMULATION_STOPPED),
        ]

    def _log(self, level, message):
        if self._logger is not None:
            getattr(self._logger, level)(f"[oscbf] {message}")

    def _on_stop(self, event) -> None:
        self._attached = False  # PhysX rebuilds the articulation (gains, modes) on the next play

    def _on_step(self, dt, context) -> None:
        if self.diagnostics["refused"] or not (self._attached or self._attach(dt)):
            return
        self._t += dt
        self._k += 1
        if self._k % self.every:  # velocity targets persist between controller steps; efforts do not
            if self.mode == "torque":
                self._write(self._u)
            return
        q, qd = self._read()
        target, stale = self.holder.get(self._t)
        if stale != self.diagnostics["stale"]:
            self._log("warning" if stale else "info", "no target for 0.25 s: holding the pose" if stale else "targets resumed")
        self.diagnostics["stale"] = stale
        if not stale:
            self._hold = None
        elif self._hold is None:
            position, rotation = self.model.ee_pose(q)
            self._hold = Target(position=position, rotation=rotation)
        if stale and self.mode == "velocity":
            u = np.zeros(len(q))
        else:
            u = self._controller.step(self._hold or target, q, qd)
        self.diagnostics["infeasible"] = self._controller.events["infeasible"]
        self._u = u
        self._write(u)
        width = (self._hold or target).gripper_width
        if np.isfinite(width) and width != self._width:
            self._width = width  # one finger travels half the opening; clamped to its joint range
            finger = float(np.clip(width / 2.0, *self._finger_range))
            self._arm.set_dof_position_targets(np.array([[finger]]), dof_indices=[self._finger])

    def _attach(self, dt) -> bool:
        """First substep after play: mass properties from PhysX, the dynamics check, control modes."""
        arm = self._arm
        dofs = list(arm.dof_names)
        self._arm_dofs = [dofs.index(j) for j in self.model.arm_joints]
        self._finger = dofs.index(self.model.gripper_joint)
        lower, upper = arm.get_dof_limits()
        self._finger_range = (float(lower.numpy()[0][self._finger]), float(upper.numpy()[0][self._finger]))
        self.every = max(1, round(1.0 / (dt * self.rate))) if self.rate else 1
        view = getattr(arm, "_physics_articulation_view", None)  # private: 0.2 ms cheaper per call
        self._view = view if view is not None and hasattr(view, "set_dof_actuation_forces") else None
        names = list(arm.link_names)
        masses = arm.get_link_masses().numpy()[0]
        coms = arm.get_link_coms()[0].numpy()[0]
        inertias = arm.get_link_inertias().numpy()[0].reshape(-1, 3, 3)
        properties = [dict(zip(names, values)) for values in (masses, coms, inertias)]
        if self.model.set_mass_properties(*properties, extra_links=self._extra_links(names)):
            self._controller = Controller(self.model, *self._settings)
            self._log("warning", f"PhysX mass properties differ from the description: recompiled in {self._controller.warm_up():.1f} s")
        q, _ = self._read_api()
        M = arm.get_mass_matrices().numpy()[0][np.ix_(self._arm_dofs, self._arm_dofs)]
        g = arm.get_dof_gravity_compensation_forces().numpy()[0][self._arm_dofs]
        man = self.model.manipulator
        error_M = np.linalg.norm(np.asarray(man.mass_matrix(q)) - M) / np.linalg.norm(M)
        error_g = np.linalg.norm(np.asarray(man.gravity_vector(q)) - g) / np.linalg.norm(g)
        if max(error_M, error_g) > DYNAMICS_TOL:
            self.diagnostics["refused"] = (
                f"dynamics check failed: M {error_M:.1e}, g {error_g:.1e} relative > {DYNAMICS_TOL}; PhysX links {names}"
            )
            self._log("error", f"robot not registered for OSCBF control, it stays position-driven: {self.diagnostics['refused']}")
            return False
        arm.switch_dof_control_mode("effort" if self.mode == "torque" else "velocity", dof_indices=self._arm_dofs)
        if self._view is not None:
            sample = self._view.get_dof_positions()  # its device: "cpu", or the GPU with GPU physics
            self._buffer, self._device = np.zeros(sample.shape, np.float32), sample.device
            self._index = wp.array([0], dtype=wp.int32, device=self._device)
        self._t, self._k, self._hold, self._width = 0.0, 0, None, None
        self._attached = True
        self._log("info", f"attached: {self.mode}, controller every {self.every} substep(s), {'tensor view' if self._view else 'API'} I/O")
        return True

    def _extra_links(self, names):
        """PhysX links the description lacks (a camera mount, say) as {name: (parent, T_parent_urdf_child)}."""
        extra = [n for n in names if n not in self.model.usd_to_urdf]
        if not extra or self._view is None:
            return None
        parents = dict(zip(self._view.shared_metatype.link_names, self._view.shared_metatype.link_parents))
        poses = self._view.get_link_transforms().numpy().reshape(-1, len(names), 7)[0]
        T = {}
        for name, (x, y, z, qx, qy, qz, qw) in zip(names, poses):
            T[name] = np.eye(4)
            T[name][:3, :3] = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()
            T[name][:3, 3] = [x, y, z]
        to_urdf = lambda name: self.model.usd_to_urdf.get(name, np.eye(4))  # noqa: E731
        return {n: (parents[n], to_urdf(parents[n]) @ np.linalg.inv(T[parents[n]]) @ T[n]) for n in extra}

    def _read(self):
        if self._view is None:
            return self._read_api()
        q = self._view.get_dof_positions().numpy()[0, self._arm_dofs]
        qd = self._view.get_dof_velocities().numpy()[0, self._arm_dofs]
        return q.astype(np.float64), qd.astype(np.float64)

    def _read_api(self):
        q = self._arm.get_dof_positions().numpy()[0][self._arm_dofs]
        qd = self._arm.get_dof_velocities().numpy()[0][self._arm_dofs]
        return q.astype(np.float64), qd.astype(np.float64)

    def _write(self, u) -> None:
        if self._view is None:
            setter = self._arm.set_dof_efforts if self.mode == "torque" else self._arm.set_dof_velocity_targets
            setter(np.asarray(u)[None], dof_indices=self._arm_dofs)
            return
        self._buffer[0, self._arm_dofs] = u
        data = wp.from_numpy(self._buffer, dtype=wp.float32, device=self._device)
        setter = self._view.set_dof_actuation_forces if self.mode == "torque" else self._view.set_dof_velocity_targets
        setter(data, self._index)

    def close(self) -> None:
        for uid in self._ids:
            SimulationManager.deregister_callback(uid)
        self._ids = []
```
Notes for the implementer: `Target` objects are truthy, so `self._hold or target` picks the latched hold when stale and the fresh target otherwise. The raw writes cover all DOFs: the fingers get effort 0 or velocity target 0, which their position drives already had (as in `run.py`).

- [ ] **Step 4: Run them to see them pass**

Run: `cd ~/ros2_ws/src/guide && PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 ~/ros2_ws/.venv/bin/python -m pytest -q guide_core/test/test_oscbf_host.py`
Expected: `10 passed` in a few seconds (the fake controller compiles nothing; the dynamics check uses OSCBF's model eagerly).

- [ ] **Step 5: Commit**

```bash
cd ~/ros2_ws/src/guide
git add guide_core/guide_core/control/oscbf_host.py guide_core/test/test_oscbf_host.py
git commit -m "feat(guide_core): OSCBF host in the physics pre-step, tensor-view I/O with API fallback"
```

---

### Task 3: Configuration and wiring (`init.yaml` → host → ROS subscription)

**Files:**
- Modify: `guide_core/guide_core/scene/scene_orchestrator.py` (`create_robot_graphs`, lines 245–260)
- Modify: `guide_core/guide_core/core/commands/_cmd_robot.py` (`_cmd_create_robot_control`, lines 68–103)
- Modify: `guide_core/guide_core/core/commands/_cmd_simulator.py` (new `_attach_oscbf`; the robot loop of `_cmd_register_scene`, lines 92–100)
- Modify: `guide_core/guide_core/core/guide_simulator.py` (new `motion_hosts`)
- Modify: `guide_core/guide_core/ros/guide_ros.py` (imports; `_register_callback`; new `_subscribe_motion_target`; new `target_from_msg`)
- Modify: `guide_tasks/block_bin/config/init.yaml` (`robots.franka`)
- Test: `guide_core/test/test_command_fixes.py` (append three tests)

**Interfaces:**
- Consumes: Task 1's `MotionTarget`; Task 2's `OscbfHost` (`holder`, `model.base_link`); Plan A's `Target`, `POSE`, `TargetHolder`.
- Produces:
  - each robot dict from `SceneOrchestrator.create_robot_graphs()` gains `"controller": str` (default `"moveit"`) and `"oscbf": dict` (default `{}`), read from `robots.<name>.controller` / `robots.<name>.oscbf` (`robot`, `mode`, `rate`, `gains`, `barriers`).
  - `_cmd_create_robot_control(..., subscribe: bool = True)`.
  - `_attach_oscbf(self, scene_id, articulation_root, namespace, robot) -> None`: adds the host to `runtime._motion_hosts[scene_id]` and sets `host.topic = f"{namespace}/motion_target"` (e.g. `/Sim_0/Scene_0/franka/motion_target`).
  - `GUIDESimulator.motion_hosts(scene_id) -> list`.
  - `GUIDEROS2Interface._subscribe_motion_target(host) -> None`; module function `target_from_msg(msg: MotionTarget) -> Target`.

- [ ] **Step 1: Write the failing tests**

Append to `guide_core/test/test_command_fixes.py`:
```python
def test_robot_graphs_carry_the_motion_layer(isaac_import):
    orchestrator = isaac_import("guide_core.scene.scene_orchestrator").SceneOrchestrator
    config = {"robots": {"franka": {"path": "/fr3", "default_joint_states": [0.0] * 9,
                                    "controller": "oscbf", "oscbf": {"mode": "velocity"}},
                         "other": {"path": "/ur", "default_joint_states": [0.0] * 6}}}

    franka, other = orchestrator.create_robot_graphs(SimpleNamespace(_config=config))

    assert (franka["controller"], franka["oscbf"]) == ("oscbf", {"mode": "velocity"})
    assert (other["controller"], other["oscbf"]) == ("moveit", {})


def test_an_oscbf_robot_gets_a_host_and_no_joint_command_subscriber(command_module, monkeypatch):
    module = command_module("_cmd_simulator")
    made = []

    class FakeHost:
        def __init__(self, *args, **kwargs):
            made.append((args, kwargs))

    monkeypatch.setitem(sys.modules, "guide_core.control.oscbf_host", SimpleNamespace(OscbfHost=FakeHost))
    robots = [{"namespace": "/franka", "articulation_root": "/fr3", "path": "/franka",
               "default_joint_states": list(range(9)), "controller": "oscbf",
               "oscbf": {"robot": "fr3", "mode": "velocity", "rate": 60}}]
    manager = SimpleNamespace(
        get_scene_usd_path=lambda i: None, get_scene_robot_graphs=lambda i: robots,
        get_scene_camera_graphs=lambda i: None, _scenes=[SimpleNamespace(replicator_files=None)],
    )
    controls = []
    runtime = host(
        _simulator=SimpleNamespace(_add_scene_to_manager=lambda p: (0, (0, 0, 0), {}), _scene_manager=manager, _sim_path="/Sim_0"),
        _cmd_add_scene=lambda **kwargs: None,
        _cmd_set_world_poses=lambda **kwargs: None,
        _cmd_create_robot_control=lambda **kwargs: controls.append(kwargs),
        _cmd_create_tf_graph=lambda **kwargs: None,
    )

    module._cmd_register_scene(runtime, "block_bin")

    assert controls[0]["subscribe"] is False
    (args, kwargs), = made
    assert args == ("/Scene_0/fr3",) and (kwargs["robot"], kwargs["mode"], kwargs["rate"]) == ("fr3", "velocity", 60)
    assert kwargs["q_rest"] == list(range(7))
    assert runtime._motion_hosts[0][0].topic == "/Sim_0/Scene_0/franka/motion_target"


def test_motion_targets_are_subscribed_depth_one_and_converted(isaac_import):
    from guide_motion.targets import TargetHolder
    from guide_msgs.msg import MotionTarget

    ros = isaac_import("guide_core.ros.guide_ros")
    subscriptions = []
    me = SimpleNamespace(_logger=MagicMock(), _reentrant_group=None,
                         create_subscription=lambda *args, **kwargs: subscriptions.append(args))
    target_host = SimpleNamespace(topic="/Sim_0/Scene_0/franka/motion_target", holder=TargetHolder(),
                                  model=SimpleNamespace(base_link="fr3_link0"))

    ros.GUIDEROS2Interface._subscribe_motion_target(me, target_host)

    (msg_type, topic, put, qos), = subscriptions
    assert msg_type is MotionTarget and topic == target_host.topic and qos.depth == 1
    msg = MotionTarget(mode=MotionTarget.MODE_POSE, frame_id="fr3_link0", gripper_width=0.08)
    msg.pose.position.x, msg.pose.orientation.z, msg.pose.orientation.w = 0.4, 1.0, 0.0  # 180 deg about z
    put(msg)
    target, stale = target_host.holder.get(0.0)
    assert not stale and target.position[0] == 0.4 and target.gripper_width == 0.08
    assert abs(target.rotation[0, 0] + 1.0) < 1e-12
    put(MotionTarget(frame_id="Scene_0"))  # not the base frame: dropped
    broken = MotionTarget(frame_id="fr3_link0")
    broken.pose.position.x, broken.pose.orientation.w = float("nan"), 1.0
    put(broken)  # a NaN from a buggy publisher: dropped
    assert target_host.holder.get(0.0)[0] is target
```

- [ ] **Step 2: Run them to see them fail**

Run (sourced, Task 1 built): `cd ~/ros2_ws/src/guide && PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 ~/ros2_ws/.venv/bin/python -m pytest -q guide_core/test/test_command_fixes.py -k "motion or oscbf"`
Expected: 3 failures: `KeyError: 'controller'`, `KeyError: 'subscribe'`, and `AttributeError: type object 'GUIDEROS2Interface' has no attribute '_subscribe_motion_target'`.

- [ ] **Step 3: The robot dicts carry the motion layer**

In `guide_core/guide_core/scene/scene_orchestrator.py`, `create_robot_graphs`, replace
```python
                    "default_joint_states": default_joint_states,
                }
```
with
```python
                    "default_joint_states": default_joint_states,
                    # Motion layer (S2b): "moveit" (the joint_command graph) or "oscbf"
                    "controller": data.get("controller", "moveit"),
                    "oscbf": data.get("oscbf", {}),
                }
```

- [ ] **Step 4: The joint-state graph can leave `joint_command` out**

In `guide_core/guide_core/core/commands/_cmd_robot.py`, `_cmd_create_robot_control`, replace
```python
    default_joint_states: list[float] | None = None,
) -> None:
```
with
```python
    default_joint_states: list[float] | None = None,
    subscribe: bool = True,
) -> None:
```
and
```python
    js_graph._subscriber = True
    js_graph._sub_move_robot = True
```
with
```python
    # controller: oscbf drives the arm from the physics step; no joint_command into the drives
    js_graph._subscriber = subscribe
    js_graph._sub_move_robot = subscribe
```

- [ ] **Step 5: `register_scene` attaches a host per oscbf robot**

In `guide_core/guide_core/core/commands/_cmd_simulator.py`, before `def _cmd_register_scene(self, package_name: str)`, add:
```python
def _attach_oscbf(self, scene_id: int, articulation_root: str, namespace: str, robot: dict) -> None:
    """S2b: guide_motion's OSCBF controller in the physics step drives this robot.

    Compiles at registration (seconds; the persistent JAX cache makes restarts quicker). The
    GUIDE node subscribes `<namespace>/motion_target` for it after the scene is registered.
    """
    from guide_core.control.oscbf_host import OscbfHost

    settings = robot.get("oscbf", {})
    host = OscbfHost(
        articulation_root,
        robot=settings.get("robot", "fr3"),
        mode=settings.get("mode", "torque"),
        gains=settings.get("gains"),
        rate=settings.get("rate"),
        barriers=settings.get("barriers"),
        # nullspace posture: the configuration every episode starts from
        q_rest=list(robot.get("default_joint_states", []))[:7] or None,
        logger=self._logger,
    )
    host.topic = f"{namespace}/motion_target"
    if not hasattr(self, "_motion_hosts"):
        self._motion_hosts = {}
    self._motion_hosts.setdefault(scene_id, []).append(host)  # also keeps it alive: callbacks hold a weakref
```
and in `_cmd_register_scene` replace
```python
        for robot in robots:
            self._cmd_create_robot_control(
                namespace=f'{simulator._sim_path}{scene_path}{robot.get("namespace", "/robot")}',
                articulation_root=f'{scene_path}{robot.get("articulation_root", "/robot")}',
                path=f'{scene_path}/Graph{robot.get("path", "/robot")}_control_graph',
                default_joint_states=np.array(robot.get("default_joint_states", [])),
            )
```
with
```python
        for robot in robots:
            oscbf = robot.get("controller", "moveit") == "oscbf"
            self._cmd_create_robot_control(
                namespace=f'{simulator._sim_path}{scene_path}{robot.get("namespace", "/robot")}',
                articulation_root=f'{scene_path}{robot.get("articulation_root", "/robot")}',
                path=f'{scene_path}/Graph{robot.get("path", "/robot")}_control_graph',
                default_joint_states=np.array(robot.get("default_joint_states", [])),
                subscribe=not oscbf,
            )
            if oscbf:
                _attach_oscbf(
                    self,
                    id,
                    f'{scene_path}{robot.get("articulation_root", "/robot")}',
                    f'{simulator._sim_path}{scene_path}{robot.get("namespace", "/robot")}',
                    robot,
                )
```
(The command registry binds every function of a `_cmd*.py` module to the runtime, `_attach_oscbf` included; calling it as a module function keeps it testable on a fake runtime.)

- [ ] **Step 6: The simulator hands its hosts to the ROS side**

In `guide_core/guide_core/core/guide_simulator.py`, add this method to `GUIDESimulator`, right above `def reset_scene(self, scene_id: int) -> bool:`:
```python
    def motion_hosts(self, scene_id: int) -> list:
        """The scene's OSCBF hosts (robots with controller: oscbf), for the ROS side to feed."""
        return getattr(self._runtime, "_motion_hosts", {}).get(scene_id, [])
```

- [ ] **Step 7: GUIDE's ROS node subscribes the targets at depth 1**

In `guide_core/guide_core/ros/guide_ros.py`:
- replace `from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup` with
```python
import numpy as np
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup
```
- replace `from rclpy.node import Node` with
```python
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile
from scipy.spatial.transform import Rotation
```
- replace `from guide_core.types.geometry import Pose` with
```python
from guide_core.types.geometry import Pose
from guide_motion.targets import POSE, Target
from guide_msgs.msg import MotionTarget
```
- in `_register_callback`, replace `            id, offset = self._backend.register_scene(path)` with
```python
            id, offset = self._backend.register_scene(path)
            for host in self._backend.motion_hosts(id):
                self._subscribe_motion_target(host)
```
- before `    def _pose_request_callback(` add the method
```python
    def _subscribe_motion_target(self, host) -> None:
        """Depth 1, KEEP_LAST: the newest target wins (the OmniGraph subscriber served the oldest
        of 10, 110 ms stale at 100 Hz). Each message is one lock-protected slot write."""

        def put(msg: MotionTarget) -> None:
            if msg.frame_id and msg.frame_id != host.model.base_link:
                self._logger.warning(
                    f"{host.topic}: target in '{msg.frame_id}', expected '{host.model.base_link}'; dropped",
                    throttle_duration_sec=5.0,
                )
                return
            target = target_from_msg(msg)
            if not all(np.all(np.isfinite(v)) for v in (target.position, target.rotation, target.twist, target.joints)):
                self._logger.warning(f"{host.topic}: non-finite target dropped", throttle_duration_sec=5.0)
                return
            host.holder.put(target)

        self.create_subscription(
            MotionTarget, host.topic, put, QoSProfile(depth=1, history=HistoryPolicy.KEEP_LAST),
            callback_group=self._reentrant_group,
        )

```
- before `def launch_ros_interface(node: GUIDEROS2Interface):` add
```python
def target_from_msg(msg: MotionTarget) -> Target:
    """guide_msgs/MotionTarget -> guide_motion Target (the pose is read for MODE_POSE only)."""
    p, o, v, w = msg.pose.position, msg.pose.orientation, msg.twist.linear, msg.twist.angular
    rotation = Rotation.from_quat([o.x, o.y, o.z, o.w]).as_matrix() if msg.mode == POSE else np.eye(3)
    return Target(
        mode=int(msg.mode),
        position=np.array([p.x, p.y, p.z]),
        rotation=rotation,
        twist=np.array([v.x, v.y, v.z, w.x, w.y, w.z]),
        joints=np.asarray(msg.joints, float),
        gripper_width=float(msg.gripper_width),
    )


```
(`gripper_width` NaN is allowed through: it means "leave the gripper as it is".)

- [ ] **Step 8: Document the switch in block_bin's `init.yaml`**

In `guide_tasks/block_bin/config/init.yaml`, under `robots: franka:`, replace the end of `default_joint_states`
```yaml
      - 0.785398
      - 0.0
      - 0.0
```
with
```yaml
      - 0.785398
      - 0.0
      - 0.0
    # Motion layer (S2b). moveit: move_group + ros2_control over joint_command (the default).
    # oscbf: guide_motion's OSCBF controller in Isaac's physics step, fed MotionTargets by
    # ROS2Robot; the block_bin bring-up then starts no move_group. Under oscbf the arm's drives
    # are effort (torque) or velocity drives, so config/reset.yaml's drive-target homing does
    # not move it: home with a GUIDE-EX move instead.
    controller: moveit
    oscbf:
      robot: fr3       # guide_motion/guide_motion/robots/<robot>
      mode: torque     # torque (rate 120: every substep) | velocity (rate 60: once per frame)
      rate: 120
      gains: {}        # overrides of guide_motion.control.GAINS[mode]
      barriers: {}     # overrides of guide_motion.control.BARRIERS
```

- [ ] **Step 9: Run the tests and build**

```bash
cd ~/ros2_ws/src/guide && PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 ~/ros2_ws/.venv/bin/python -m pytest -q guide_core/test \
  --ignore-glob='*test_flake8.py' --ignore-glob='*test_pep257.py' --ignore-glob='*test_copyright.py'
cd ~/ros2_ws && colcon build --packages-select guide_core block_bin && source install/setup.zsh
```
Expected: every guide_core test passes (the prototype: 153), including the three new ones; `Finished <<< guide_core`, `Finished <<< block_bin`.

- [ ] **Step 10: Commit**

```bash
cd ~/ros2_ws/src/guide
git add guide_core/guide_core/scene/scene_orchestrator.py guide_core/guide_core/core/commands/_cmd_robot.py \
  guide_core/guide_core/core/commands/_cmd_simulator.py guide_core/guide_core/core/guide_simulator.py \
  guide_core/guide_core/ros/guide_ros.py guide_tasks/block_bin/config/init.yaml guide_core/test/test_command_fixes.py
git commit -m "feat(guide_core): controller: oscbf in init.yaml attaches the OSCBF host and its depth-1 target topic"
git status --porcelain   # guide_core/config/init.yaml still ' M', unstaged
```

---

### Task 4: Headless Isaac checks — hold, square, determinism

**Files:**
- Create: `guide_core/test/isaac_oscbf_check.py` (not `test_*`: pytest does not collect it)
- Create (results): `docs/design/s2b_oscbf_isaac_checks.json`

**Interfaces:**
- Consumes: Task 2's `OscbfHost` (`holder.put`, `_u`, `_hold`, `diagnostics`); Plan A's `Model.load`, `description.ROBOTS`, `description.ROBOTS_DIR / "panda" / "panda.usda"`.
- Produces: `isaac_oscbf_check.py --robot {fr3,panda} --mode {torque,velocity,baseline} --check {hold,square,rtf} [--baseline JSON] --out JSON` → JSON with `pass`, and for hold `hold_mm_mean`, `hold_mm_max`, `stale`; for square `path_mm_mean`, `path_mm_peak` and `<out>.q.npy`; for rtf `frame_ms_mean`, `added_ms`; always `nan`, `u_over_limit`, `infeasible`, `refused` for controller runs. Exit code 1 on a failed threshold.

- [ ] **Step 1: Write the check**

`guide_core/test/isaac_oscbf_check.py`:
```python
"""Headless Isaac checks for the OSCBF host (S2b spec 7). Run by hand, one Isaac at a time:

    OMNI_KIT_ACCEPT_EULA=YES env -u CUDA_DEVICE_ORDER ~/ros2_ws/.venv/bin/python \\
        ~/ros2_ws/src/guide/guide_core/test/isaac_oscbf_check.py --robot fr3 --mode torque --check hold --out /tmp/x.json

hold:   no targets for 5 s, so the host's stale hold: mean tool error over the last 4 s <= 0.1 mm.
square: 2 s hold, then a 10 cm square at 10 cm/s as a 20 Hz zero-order hold: mean path deviation
        on lap 2 <= 3 mm (torque) / 2 mm (velocity); no NaN; |command| <= the URDF limits. Saves q.
rtf:    frame time over the square; --mode baseline records an awake arm without a controller,
        a controller run given --baseline must add <= 2.5 ms (torque) / 1.2 ms (velocity).
Writes --out JSON, exits 1 when a threshold fails. Not collected by pytest (no test_ prefix).
"""

import argparse
import json
import sys
import time
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--robot", choices=["fr3", "panda"], required=True)
parser.add_argument("--mode", choices=["torque", "velocity", "baseline"], required=True)
parser.add_argument("--check", choices=["hold", "square", "rtf"], required=True)
parser.add_argument("--baseline", help="rtf: the baseline run's JSON")
parser.add_argument("--out", required=True)
args = parser.parse_args()
assert args.mode != "baseline" or args.check == "rtf", "the baseline exists for the rtf check only"

from isaacsim import SimulationApp  # noqa: E402

# GUIDE's startup settings (config/init.yaml), headless, rendering on the A4000.
app = SimulationApp({"headless": True, "renderer": "RaytracedLighting", "anti_aliasing": 0,
                     "width": 640, "height": 360, "active_gpu": 1, "multi_gpu": False})

import numpy as np  # noqa: E402
import omni.replicator.core as rep  # noqa: E402
from isaacsim.core.api import World  # noqa: E402  (deprecated, but it is what GUIDE steps with)
from isaacsim.core.experimental.prims import Articulation  # noqa: E402
from isaacsim.core.experimental.utils import stage as stage_utils  # noqa: E402
from isaacsim.core.simulation_manager import SimulationEvent, SimulationManager  # noqa: E402
from isaacsim.storage.native import get_assets_root_path  # noqa: E402
from pxr import UsdLux  # noqa: E402

from guide_core.control.oscbf_host import OscbfHost  # noqa: E402
from guide_motion import description  # noqa: E402
from guide_motion.model import Model  # noqa: E402
from guide_motion.targets import Target  # noqa: E402

PHYSICS_HZ, FRAME_HZ, TARGET_HZ = 120, 60, 20.0
HOLD_S, LAP_S, SETTLE_FRAMES = 2.0, 4.0, 30
SQUARE = np.array([[0, 0, 0], [0, 0.1, 0], [0.1, 0.1, 0], [0.1, 0, 0], [0, 0, 0]], float)  # +y, +x, -y, -x
LIMIT = {"hold": 0.1, "square": {"torque": 3.0, "velocity": 2.0}, "rtf": {"torque": 2.5, "velocity": 1.2}}


def square_offset(t):
    """The start for HOLD_S, then the square at 10 cm/s, sampled at 20 Hz and held."""
    if t < HOLD_S:
        return np.zeros(3)
    k = int((t - HOLD_S) * TARGET_HZ)
    s = (k / TARGET_HZ) % LAP_S / LAP_S * 4
    i = int(s)
    return SQUARE[i] + (s - i) * (SQUARE[i + 1] - SQUARE[i])


world = World(stage_units_in_meters=1.0, physics_dt=1.0 / PHYSICS_HZ, rendering_dt=1.0 / FRAME_HZ)
world.get_physics_context().enable_gpu_dynamics(False)
world.scene.add_default_ground_plane()
UsdLux.DistantLight.Define(world.stage, "/World/light").CreateIntensityAttr(3000)
source = (str(description.ROBOTS_DIR / "panda" / "panda.usda") if args.robot == "panda"
          else get_assets_root_path() + description.ROBOTS["fr3"]["usd"])
stage_utils.add_reference_to_stage(source, "/World/robot")
camera = rep.create.camera(position=(1.6, 1.2, 1.1), look_at=(0.3, 0.0, 0.3))
annotator = rep.AnnotatorRegistry.get_annotator("rgb")
annotator.attach([rep.create.render_product(camera, (640, 480))])  # one camera: the measurement's light load

model = Model.load(args.robot)
arm = Articulation("/World/robot")
arm.set_sleep_thresholds(0.0)  # awake in every run: a sleeping arm's physics is free and fakes the baseline
world.reset()
dofs = list(arm.dof_names)
ARM = [dofs.index(j) for j in model.arm_joints]
home = np.zeros(arm.num_dofs)
home[ARM] = model.home
arm.set_dof_positions(home[None])
arm.set_dof_position_targets(home[None])
for _ in range(SETTLE_FRAMES):  # position-driven settle at home, as GUIDE's scene is before registration
    world.step(render=True)
    annotator.get_data()

host = None
if args.mode != "baseline":  # registration: compiles (the persistent cache makes reruns quicker)
    host = OscbfHost("/World/robot", robot=args.robot, mode=args.mode,
                     rate=PHYSICS_HZ if args.mode == "torque" else FRAME_HZ, q_rest=model.home)
p0, R0 = model.ee_pose(arm.get_dof_positions().numpy()[0][ARM].astype(float))

log = []
if args.check != "rtf":  # a per-substep read would bill the RTF run

    def on_post_step(dt, context):
        q = arm.get_dof_positions().numpy()[0][ARM].astype(float)
        log.append((q, np.full(len(ARM), np.nan) if host._u is None else np.array(host._u, float)))

    SimulationManager.register_callback(on_post_step, SimulationEvent.PHYSICS_POST_STEP)

frames = int(5.0 * FRAME_HZ) if args.check == "hold" else int((HOLD_S + 2 * LAP_S) * FRAME_HZ)
frame_ms, last_k = [], -1
for i in range(frames):
    t = i / FRAME_HZ
    k = int(t * TARGET_HZ)
    if host is not None and args.check != "hold" and k != last_k:
        host.holder.put(Target(position=p0 + square_offset(t), rotation=R0))  # GUIDE's ROS thread does this
        last_k = k
    start = time.perf_counter()
    world.step(render=True)
    annotator.get_data()
    frame_ms.append(1e3 * (time.perf_counter() - start))

result = {"robot": args.robot, "mode": args.mode, "check": args.check, "frames": frames}
ok = True
if args.check == "rtf":
    ft = np.array(frame_ms[30:])
    result.update(frame_ms_mean=float(ft[ft < 40.0].mean()), spikes_over_40ms=int((ft >= 40.0).sum()))
    if args.baseline:
        result["added_ms"] = result["frame_ms_mean"] - json.loads(Path(args.baseline).read_text())["frame_ms_mean"]
        ok = result["added_ms"] <= LIMIT["rtf"][args.mode]
else:
    q, u = np.array([e[0] for e in log]), np.array([e[1] for e in log])
    p = np.array([model.ee_pose(x)[0] for x in q])
    t = np.arange(len(q)) / PHYSICS_HZ
    limit = model.max_effort if args.mode == "torque" else model.max_velocity
    result.update(nan=bool(np.isnan(q).any() or np.isnan(u[1:]).any()), u_over_limit=float(np.nanmax(np.abs(u) / limit)),
                  infeasible=host.diagnostics["infeasible"], refused=host.diagnostics["refused"])
    ok = not result["nan"] and result["u_over_limit"] <= 1.0 + 1e-6 and not result["refused"]
    if args.check == "hold":
        error = 1e3 * np.linalg.norm(p - host._hold.position, axis=1)[t >= 1.0]
        result.update(hold_mm_mean=float(error.mean()), hold_mm_max=float(error.max()), stale=host.diagnostics["stale"])
        ok = ok and result["hold_mm_mean"] <= LIMIT["hold"] and result["stale"]
    else:
        path = np.array([p0 + square_offset(HOLD_S + x) for x in np.linspace(0, LAP_S, 4000, endpoint=False)])
        lap2 = t >= HOLD_S + LAP_S
        deviation = 1e3 * np.linalg.norm(p[lap2, None] - path[None], axis=2).min(axis=1)
        result.update(path_mm_mean=float(deviation.mean()), path_mm_peak=float(deviation.max()))
        ok = ok and result["path_mm_mean"] <= LIMIT["square"][args.mode]
        np.save(Path(args.out).with_suffix(".q.npy"), q)
result["pass"] = bool(ok)
Path(args.out).write_text(json.dumps(result, indent=1))
print("RESULT", json.dumps(result), flush=True)
app.close()
sys.exit(0 if ok else 1)
```
The scene, render load, home settle and the square are `run.py`'s; the host replaces `run.py`'s inline controller. The spec's thresholds are means: "hold 0.03 mm" and "path deviation 1.7 mm mean, 4.6 mm peak" in spec 5 are means and peaks over the same windows, and the peak is reported, not gated.

- [ ] **Step 2: Run hold and square for both robots and modes**

From a sourced shell (guide_core built in Task 3), one run at a time:
```bash
pgrep -af "kit/kit|guide_core"   # must print nothing
mkdir -p /tmp/s2b_checks && cd ~/ros2_ws
for robot in fr3 panda; do for mode in torque velocity; do for check in hold square; do
  OMNI_KIT_ACCEPT_EULA=YES env -u CUDA_DEVICE_ORDER .venv/bin/python src/guide/guide_core/test/isaac_oscbf_check.py \
    --robot $robot --mode $mode --check $check --out /tmp/s2b_checks/${robot}_${mode}_${check}.json 2>&1 | grep -E "RESULT|\[oscbf\]|Error"
done; done; done
```
Expected: every `RESULT` line has `"pass": true` and `"refused": null`; for hold `"stale": true` and `hold_mm_mean` ≤ 0.1 (measured on the Panda with matched inertials: 0.03 mm torque, 0.09 mm velocity); for square `path_mm_mean` ≤ 3.0 (torque, measured 1.7) / ≤ 2.0 (velocity, measured 0.7); `"nan": false`. The `[oscbf]` lines show `compiled in … s` and `attached: …, tensor view I/O`. A refusal or a failed threshold: read the `[oscbf]` error line first (it names M/g and the PhysX links) and stop and report; do not loosen a threshold.

- [ ] **Step 3: Determinism — the same run twice**

```bash
cd ~/ros2_ws
OMNI_KIT_ACCEPT_EULA=YES env -u CUDA_DEVICE_ORDER .venv/bin/python src/guide/guide_core/test/isaac_oscbf_check.py \
  --robot fr3 --mode torque --check square --out /tmp/s2b_checks/fr3_torque_square_rerun.json | grep RESULT
.venv/bin/python -c "import numpy as np; a = np.load('/tmp/s2b_checks/fr3_torque_square.q.npy'); b = np.load('/tmp/s2b_checks/fr3_torque_square_rerun.q.npy'); n = min(len(a), len(b)); print(len(a), len(b), float(np.abs(a[:n] - b[:n]).max()))"
```
Expected: equal lengths and a max difference ≤ 1e-6 rad. If it is larger, rerun the velocity square twice the same way to tell the controller's run-to-run spread from PhysX's, and report both numbers.

- [ ] **Step 4: Record and commit**

```bash
cd ~/ros2_ws && .venv/bin/python -c "
import glob, json; from pathlib import Path
runs = {Path(f).stem: json.loads(Path(f).read_text()) for f in sorted(glob.glob('/tmp/s2b_checks/*.json'))}
Path('src/guide/docs/design/s2b_oscbf_isaac_checks.json').write_text(json.dumps(runs, indent=1)); print(len(runs), all(r['pass'] for r in runs.values()))"
cd ~/ros2_ws/src/guide
git add guide_core/test/isaac_oscbf_check.py docs/design/s2b_oscbf_isaac_checks.json
git commit -m "test(guide_core): headless OSCBF host checks: hold, 20 Hz square, determinism"
```
Expected: `9 True`.

---

### Task 5: RTF check and a GUIDE smoke test

**Files:**
- Modify (results): `docs/design/s2b_oscbf_isaac_checks.json`
- Modify, temporarily (reverted in Step 5): `guide_tasks/block_bin/config/init.yaml`

**Interfaces:**
- Consumes: Task 4's script; Task 3's wiring; Task 1's message.
- Produces: the RTF numbers (`added_ms` per mode) and the evidence that GUIDE registers block_bin's FR3 under `controller: oscbf` and moves it on a `MotionTarget` from the ROS CLI.

- [ ] **Step 1: Baseline and controller frame times**

```bash
cd ~/ros2_ws && C=src/guide/guide_core/test/isaac_oscbf_check.py
OMNI_KIT_ACCEPT_EULA=YES env -u CUDA_DEVICE_ORDER .venv/bin/python $C --robot fr3 --mode baseline --check rtf --out /tmp/s2b_checks/fr3_baseline_rtf.json | grep RESULT
for mode in torque velocity; do
  OMNI_KIT_ACCEPT_EULA=YES env -u CUDA_DEVICE_ORDER .venv/bin/python $C --robot fr3 --mode $mode --check rtf \
    --baseline /tmp/s2b_checks/fr3_baseline_rtf.json --out /tmp/s2b_checks/fr3_${mode}_rtf.json | grep RESULT
done
```
Expected: `added_ms` ≤ 2.5 (torque) and ≤ 1.2 (velocity), `"pass": true`. The measurement's raw-I/O numbers with OSCBF's 7-sphere model were +1.67 / +0.96 ms; this plan's self-collision barrier makes the step about 2× more expensive standalone (Plan A, spec question 6). If torque fails, rerun with `barriers: {min_group_gap: 3}`: add `barriers={"min_group_gap": 3}` to the `OscbfHost(...)` call in the script for that run. Report both numbers and stop — choosing the default is the user's call. Repeat the three runs once (the spikes spread RTF by ±3 %) and keep the second set if they differ by more than 0.3 ms.

- [ ] **Step 2: Register block_bin with `controller: oscbf`**

Temporarily set `controller: oscbf` under `robots: franka:` in `guide_tasks/block_bin/config/init.yaml` (do not touch `guide_core/config/init.yaml`: GUIDE starts with the user's local copy of it, which may open a Kit window), then:
```bash
cd ~/ros2_ws && colcon build --packages-select block_bin && source install/setup.zsh
export CYCLONEDDS_URI=file://$HOME/ros2_ws/install/guide_core/share/guide_core/config/cyclonedds_localhost.xml
pgrep -af "isaac|kit/kit|guide_core"   # nothing
ros2 launch guide_core bringup.launch.py > /tmp/s2b_checks/guide.log 2>&1 &
until grep -q "Running simulation loop" /tmp/s2b_checks/guide.log; do sleep 5; done
ros2 service call /Sim_0/Register guide_msgs/srv/RegisterScene "{path: 'block_bin'}"
grep "\[oscbf\]" /tmp/s2b_checks/guide.log
```
Expected: `success=True`; `[oscbf] fr3 torque: OSCBF step compiled in … s`, then `[oscbf] attached: torque, controller every 1 substep(s), tensor view I/O`, possibly preceded by `PhysX mass properties differ from the description: recompiled` (the wrist camera mount). No `robot not registered for OSCBF control` line: if there is one, it names M/g and the PhysX links; stop and report.

- [ ] **Step 3: Move it from the ROS CLI**

```bash
timeout 4 ros2 topic pub -r 20 /Sim_0/Scene_0/franka/motion_target guide_msgs/msg/MotionTarget \
  "{mode: 2, frame_id: fr3_link0, joints: [0.3, -0.785398, 0.0, -2.35619, 0.0, 1.5708, 0.785398], gripper_width: 0.08}"
ros2 topic echo --once /Sim_0/Scene_0/franka/joint_states | grep -A3 "^position"
grep -c "no target for 0.25 s" /tmp/s2b_checks/guide.log
```
Expected: `position:` starts with `- 0.29…` / `- 0.30…` (joint 1 followed the JOINTS target) and the fingers report about 0.04; at least one "no target" line (the stale hold after `timeout` stopped the publisher; one at start-up too).

- [ ] **Step 4: Record**

Append the RTF runs to the results and commit:
```bash
cd ~/ros2_ws && .venv/bin/python -c "
import glob, json; from pathlib import Path
out = Path('src/guide/docs/design/s2b_oscbf_isaac_checks.json'); runs = json.loads(out.read_text())
runs.update({Path(f).stem: json.loads(Path(f).read_text()) for f in sorted(glob.glob('/tmp/s2b_checks/*_rtf.json'))})
out.write_text(json.dumps(runs, indent=1)); print({k: v.get('added_ms') for k, v in runs.items() if k.endswith('rtf')})"
```

- [ ] **Step 5: Stop GUIDE, revert the switch, commit**

```bash
pkill -INT -f "guide_core" ; sleep 10; pgrep -af "isaac|kit/kit|guide_core"   # nothing left
```
Set `controller: moveit` back in `guide_tasks/block_bin/config/init.yaml` (the committed default until S2b is accepted), rebuild it, and check the file is unchanged against the last commit:
```bash
cd ~/ros2_ws && colcon build --packages-select block_bin
cd ~/ros2_ws/src/guide && git diff --stat guide_tasks/block_bin/config/init.yaml   # empty
git add docs/design/s2b_oscbf_isaac_checks.json
git commit -m "test(guide_core): OSCBF host RTF against the awake-arm baseline"
git status --porcelain   # only guide_core/config/init.yaml ' M'
```

---

## Self-review (run by the plan's author)

- **Spec coverage.** 4.4 message → Task 1. 4.3 configuration → Task 3 (`init.yaml`, `create_robot_graphs`); registration (description, mass properties, control-mode switch, compilation cache, warm-up) → Task 2 (`__init__`, `_attach`); per-substep read / holder / step / write → Task 2 (`_on_step`); rates → Task 2 (`every`); tensor view with API fallback → Task 2 (`_view`); depth-1 `KEEP_LAST` subscriber on GUIDE's ROS thread, one lock-protected write → Task 3 (`_subscribe_motion_target`, `TargetHolder.put`). Section 3's "states return over /joint_states" → the graph keeps publishing (Task 3, `subscribe=False` only drops the subscriber). Section 6 rules → per-substep traffic: none; persistent cache and warm-up at registration (Task 2); physics 120 Hz (unchanged); the RTF check → Task 5. Section 7 headless checks → Tasks 4–5. Section 8: stale target → hold plus a flag (Task 2); infeasible / non-finite and clamping → Plan A's `Controller`, counted in `diagnostics["infeasible"]` (Task 2); description validation → refusal with the check named (Task 2).
- **Placeholders.** None: every code step has the code; every run its command and expected output.
- **Type consistency.** `OscbfHost(articulation_root, robot, mode, gains, rate, barriers, q_rest, logger)`, `.holder`, `.model.base_link`, `.topic`, `.diagnostics` and `target_from_msg` are used with the same names in Tasks 2–5 and in Plan C; `MotionTarget` field names match spec 4.4 and Plan C's publisher.
- **Review Focus.** Each line has its test in Task 2 or 3.
