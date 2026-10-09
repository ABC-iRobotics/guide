# TODO

Deferred work, newest first. Feature-specific lists live next to their design notes
(e.g. [docs/design/zoned-randomization-todo.md](docs/design/zoned-randomization-todo.md)).

## Several robots in one scene, each with its own base

**Now:** the recorder writes one cartesian robot per scene (`dataset.cartesian_velocity_robot`,
else the first robot with an end effector): `observation.state` x..wz is its end effector as
its own base sees it (`robots.<name>.base_name`, `SceneOrchestrator.record_step`).

**Wanted:** every robot in a scene recorded the same way, each relative to its own base.

Notes for whoever implements it:
- Per-robot feature names (e.g. `<robot>.x` … `<robot>.wz`) in `observation.state` and
  `action`, and per-robot `_last_recorded_obs_pose` for the deltas.
- `base_views` / `ee_views` are already per robot; `record_step` picks only one of them.
- Inference must read each robot's pose in its own base frame (block_bin_eval's
  `BASE_OFFSET` still assumes the scene frame of the datasets recorded before this change).
