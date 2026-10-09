# Block Bin Task

GUIDE demonstration task: **put a block in a bin**. Each episode draws one of four blocks
(red, yellow, green, blue) and a side (left, right); the task is `Put the <colour> block in
the <side> bin.` (`block_bin.scene.Scene.randomize_preprocess`). A Franka FR3 picks the block
up and drops it into that bin through MoveIt 2.

## Files

| Path | Contents |
|---|---|
| `block_bin/scene.py` | `Scene`: draws the colour and the side, answers Randomize with the target block, the goal bin (`/bin_0` left, `/bin_1` right) and the task, makes the drawn block the zone target, and narrows the success check to that block in that bin. |
| `block_bin/solve_task.py` | The GUIDE-EX tree (`PickAndPlace` TASK: Unclutch, StartRecording, GetTargetPose, Pick, Place, Return, CheckSuccess, StopRecording, with a regrasp and a retreat-via-rest recovery) and the `generate_demonstration` service. |
| `config/init.yaml` | USD, robot, cameras (top, base, wrist: RGB, depth, instance segmentation), dataset features, 10 fps. |
| `config/randomize.yaml` | Replicator YAML: the blocks uniform over the table with any yaw, the bins fixed, a 0.1 m zone grid (5 x 4 = 20 zones) for the drawn block. |
| `config/reset.yaml` | Replicator YAML for the Reset service: blocks in a row, bins, arm and fingers at home. |
| `config/success.yaml` | `is_prim_contained`: the block inside the bin. |
| `launch/bringup.launch.py` | Per scene (`num_env`, default 1): MoveIt (`franka_fr3_moveit_config`'s `guide_moveit.launch.py`) and the `solve_task` node under the `.venv` Python. |
| `assets/` | `block_bin.usd`, the scene, and the RealSense D435 camera mount it references. |

## Usage

```bash
ros2 launch guide_core bringup.launch.py
ros2 service call /Sim_0/Register guide_msgs/srv/RegisterScene "{path: 'block_bin'}"
ros2 launch block_bin bringup.launch.py
ros2 service call /Sim_0/Scene_0/generate_demonstration guide_msgs/srv/Demonstration \
  "{path: '', zones: [-1], counts: [5]}"
```

`zones: [-1]` records `counts[0]` successful episodes in each of the 20 zones, `zones: []`
draws the block anywhere; an empty `path` writes under `~/dataset`.

Policy-evaluation tools for this task are in the separate `block_bin_eval` package, outside
this repository.

## Maintainer
András Makány (makany.andras@uni-obuda.hu)
