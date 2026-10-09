# Postmortem: ROS 2 service replies hang after a MoveIt motion

**Status:** Fixed 2026-07-20. **Symptom class:** "after N iterations the sim services
stop returning" / "`PoseRequest` never comes back". Seen on both Humble and Jazzy.

## Symptom

The solver calls a GUIDE service (`/Sim_*/PoseRequest`, `/CollisionRequest`, …) and the
call never returns — it eventually hits the `callService` timeout (`irob_lerobot_ros`).
The **server side is healthy**: it receives the request, computes the answer, and returns
a valid response. The **client side** never completes the request future
(`future.done() == False`), and the client's background executor sits idle in `rclpy_wait`.

Characteristically: the **first** service call to GUIDE in an episode (e.g. `Randomize`,
`start_recording`) works, and the first call **after a MoveIt arm/gripper motion**
(e.g. `PoseRequest`) hangs. "After many iterations, not a set amount" is the same bug —
it's a race, not a counter.

## Root cause

`pymoveit2` blocked in its wait loops with `rclpy.spin_once(self._node, timeout_sec=1.0)`
(5× in `moveit2.py`, 1× in `gripper_command.py`). But in GUIDE that node
(`ROS2Robot.node`) is **already** being spun by a background `MultiThreadedExecutor` in
`ROS2Robot._ros_thread`.

`rclpy.spin_once(node)` adds the node to the **global** executor and spins it. So during a
MoveIt motion, **two executors call `rclpy_wait` on the same node's entity handles
concurrently**. That is undefined behaviour in rclpy and corrupts the wait set
(`IndexError: wait set index too big` in `rclpy/event_handler.py`). After the corruption,
the node **silently stops delivering service-client replies** — hence every later
`PoseRequest`/`CollisionRequest` on that node hangs.

This is exactly what the old `@executor_safe` decorator was papering over (it detached the
node from its executor around calls). Removing that decorator exposed the real bug.

## The fix

`modules/pymoveit2/pymoveit2/utils.py` — new `spin_or_wait(node)`:

```python
def spin_or_wait(node, timeout_sec=1.0):
    executor = node.executor
    if executor is not None and getattr(executor, "_is_spinning", False):
        time.sleep(0.005)                 # external executor already services callbacks
    else:
        rclpy.spin_once(node, timeout_sec=timeout_sec)  # standalone pymoveit2 use
```

All six `rclpy.spin_once(self._node, …)` call sites in `moveit2.py` and
`gripper_command.py` now call `spin_or_wait(self._node)`.

**Why `_is_spinning`, not just `is not None`:** `Executor.remove_node()` does **not** reset
`node.executor`, so a standalone `spin_once` leaves `node.executor` dangling (pointing at
the global executor). Checking only `is not None` would then make standalone pymoveit2 stop
self-spinning and deadlock. `_is_spinning` is `True` only while an executor is actively
inside `spin()/spin_once()`, which cleanly distinguishes the persistent background MTE from
a dangling reference.

**Rebuild:** `colcon build --packages-select pymoveit2`.

## If it regresses

1. Confirm the shape: server logs the request served, client `future.done()` is `False`,
   client executor idle in `rclpy_wait`. If so it's this class of bug (reply not delivered),
   **not** DDS size, recorder, or camera load — those were ruled out (see below).
2. Grep for any new `rclpy.spin_once(` / `rclpy.spin(` on a node that an external executor
   also spins. Any node created by `ROS2Robot` and added to `_ros_thread`'s MTE must never
   be `spin_once`'d elsewhere. Route such waits through `pymoveit2.utils.spin_or_wait`.
3. Reproduce in isolation (no Isaac needed) — a node spun by a background MTE, plus a
   `spin_once` burst on the same node from another thread, then a service call: the call
   hangs with `future.done()==False`. Guarding the `spin_once` restores it. (This is how the
   root cause was confirmed.)

## Diagnosis method (for reference)

Confirmed by minimal, Isaac-free repros that **refuted** the wrong leads before landing on
the cause: a duplicate service client (works, 5/5), an 819 MB reliable DDS flood
(works, 8/8), the `guide_msgs/srv/Pose` type itself (works, 5/5). The concurrent-`spin_once`
repro deterministically reproduced the exact symptom (0/5 replies after the burst; 5/5 with
the guard).

## Related changes made during this investigation (not the root-cause fix)

Kept:
- `guide_ex/.../steps/simulation/isaac/prim.py` — `GetPrimPose`/`IsPrimClashing` checked
  `robot.node.pose`/`robot.node.collision` (always `None`) and created an **unused second
  client** on the same service each run; now check/reuse `robot.pose`/`robot.collision` with
  the node's registered callback group. Latent bug, proven harmless either way, kept as
  hygiene.
- `irob_lerobot_ros/ros2robot.py` — `callService` uses `call_async` + `Event` wait with a
  timeout/retry (replaces the `@executor_safe` executor juggling); `send_action` propagates
  `wait_until_executed()` failure. Robustness improvements, kept.
- `scene_orchestrator.py` — `publish_camera_topics` init-config flag gates the ROS camera
  publisher graphs (off for generation, on for inference). Kept (requested feature).

Speculative, kept but revisit if issues appear:
- `scene_recorder.py` / `scene_manager.py` — the recorder frame handoff is non-blocking
  (`put_nowait`, drops on a full queue instead of blocking physics) and
  `image_writer_threads=4`. These were made while (wrongly) suspecting the recorder; they
  are harmless but if you ever see dropped-frame warnings or dataset gaps, this is where to
  look.

Removed after the fix was confirmed: all diagnostic instrumentation added during the hunt
(`[STALL-WATCHDOG]`/`_inflight_cmd` in `runtime.py`, `[STALL-TRACE]` in `guide_ros.py` and
`ros2robot.py`, `[REC-TRACE]` in `scene_orchestrator.py`) plus pre-existing debug scaffolding
(`[DEBUG_FREEZE]` in `runtime.py`/`_cmd_robot.py`, `[GUIDE-TRACE]` in `guide_ros.py`/
`guide_simulator.py`).
