# TODO

Deferred work, newest first. Feature-specific lists live next to their design notes
(e.g. [docs/design/zoned-randomization-todo.md](docs/design/zoned-randomization-todo.md)).

## Record each robot's data relative to its own base

**Now:** the recorder stores the end-effector pose in the **scene frame**: the world pose minus
the scene's placement (`SceneOrchestrator.record_step`, `_offset` from
`SceneManager.add_scene`). The FR3 base sits at about (-0.307, 0.000, 0.013) in that frame
(the launch places it at `xyz="-0.3 0 0"` under `Scene_i`).

**Wanted:** the origin of each robot's data is **that robot's base** (`fr3_link0`); with
several robots in a scene, each one's data is relative to its own base.

Notes for whoever implements it:
- Express the pose in the base frame with the full transform (rotation too), not a
  translation: a second robot may be mounted rotated. Rotate the deltas (`action` x..wz)
  into the same frame.
- `record_step` records one cartesian robot (`dataset.cartesian_velocity_robot`); several
  robots need per-robot feature names (e.g. `<robot>.x`).
- Read the base pose from the articulation root at registration, once per robot, and write it
  into `guide_info.json` so the frame is documented in the dataset.
- Inference must use the same frame (block_bin_eval's `BASE_OFFSET` assumes the scene frame).
