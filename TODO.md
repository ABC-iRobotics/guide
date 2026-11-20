# TODO

Deferred work, newest first. Feature-specific lists live next to their design notes
(e.g. [docs/design/zoned-randomization-todo.md](docs/design/zoned-randomization-todo.md)).

## Inference in each robot's own base frame

The recorder writes each cartesian robot's pose in its own base frame (`dataset.cartesian_robots`,
README "Coordinates"). Inference must read it the same way: block_bin_eval's `BASE_OFFSET` still
assumes the scene frame of the datasets recorded before the base-frame change.
