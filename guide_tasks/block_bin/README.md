# Block Bin Task

GUIDE demonstration task: **put a block in a bin**. Each episode draws one of four blocks
(red, yellow, green, blue) and a side (left, right); the task is `Put the <colour> block in
the <side> bin.` (`block_bin.scene.Scene.randomize_preprocess`). A Franka FR3 picks the block
up and drops it into that bin, moved by the OSCBF controller (the default; `controller: moveit` plans through MoveIt 2).

## Files

| Path | Contents |
|---|---|
| `block_bin/scene.py` | `Scene`: draws the colour and the side, answers Randomize with the target block, the goal bin (`/bin_0` left, `/bin_1` right) and the task, makes the drawn block the zone target, and narrows the success check to that block in that bin. |
| `block_bin/solve_task.py` | Hooks for the shared solver `guide_ex.recipe.solve` (it runs the episodes, the CLI and `generate_demonstration`): where a planned miss lands, the recipe values (gripper joint, home, the camera transition's hold, the misses) and the context (block, bin, rest pose). Every other episode runs without motion noise; the misses stay, and recoveries are stripped (Contact closes kept) when an episode plans none. |
| `config/recipes/Episode.yaml` | The tree: guide_ex's `PickAndPlace` TASK (`guide_ex/guide_ex/subtasks/` and `steps/`: Unclutch, StartEpisode, GetTargetPose, WaitForCameras, Pick, Place, Return, CheckSuccess, StopEpisode, with a regrasp and a retreat-via-rest recovery) with block_bin's 28 cm lift and -90° regrasp turn as overrides. |
| `config/variants/` | `--recipe` names: `flawless` (no misses), `light_fail` (pre-grasp hover beside the block, the descent corrects it), `retry` (air grasp, PickRecovery retries), `retry_mixed` (1 or 2 of those). A comma list is taken in turn. |
| `config/init.yaml` | USD, robot, cameras (top, base, wrist: RGB, depth, instance segmentation), dataset features, 10 fps, `dataset.segmentation` (ADE20K colours: blocks `box`, bins `basket`, the robot `computer`, on black), and `dataset.camera_transition`: every episode opens with every camera on `base`'s pose, easing onto its own mount over 2 s while the arm holds still; those 20 frames carry the subtask `Startup`. |
| `config/randomize.yaml` | Replicator YAML: the blocks uniform over the table with any yaw, the bins fixed, a 0.1 m zone grid (3 x 4 = 12 zones) for the drawn block. |
| `config/reset.yaml` | Replicator YAML for the Reset service: blocks in a row, bins, arm and fingers at home. |
| `config/success.yaml` | `is_prim_contained`: the block inside the bin. |
| `launch/bringup.launch.py` | Per scene (`num_env`, default 1): MoveIt (`franka_fr3_moveit_config`'s `guide_moveit.launch.py`, only for `controller: moveit`) and the `solve_task` node under the `.venv` Python, given the controller from `config/init.yaml`; WaitForCameras (hold read from `dataset.camera_transition` and `dataset.fps` there) holds the arm for the duration plus one frame period. |
| `assets/` | `block_bin.usd`, the scene, and the RealSense D455 camera mount it references (`RealSenseD455_Camera_Mount.usd`, from the `.stl` beside it; black matte: NVIDIA vMaterials_2 `Plastic_Standardized_Surface_Finish_V33`, loaded by URL, NVIDIA Omniverse License). |

## Usage

```bash
ros2 launch guide_core bringup.launch.py
ros2 service call /Sim_0/Register guide_msgs/srv/RegisterScene "{path: 'block_bin'}"
ros2 launch block_bin bringup.launch.py
ros2 service call /Sim_0/Scene_0/generate_demonstration guide_msgs/srv/Demonstration \
  "{path: '', zones: [-1], counts: [5]}"
```

Solver options (`ros2 run block_bin solve_task -- ...`): `--recipe NAME[,NAME]`, `--motion-noise [JSON]` (perturbed demonstrations), `--seed-base`, `--perturb-all`. Each recorded dataset gets a `<path>.provenance.jsonl` (recipe, misses, failed nodes, moves).

`zones: [-1]` records `counts[0]` successful episodes in each of the 12 zones, `zones: []`
draws the block anywhere; an empty `path` writes under `~/dataset`.

Policy-evaluation tools for this task are in the separate `block_bin_eval` package, outside
this repository.

## Maintainer
András Makány (makany.andras@uni-obuda.hu)
