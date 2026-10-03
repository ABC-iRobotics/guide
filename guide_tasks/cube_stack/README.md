# Cube Stack Task

GUIDE demonstration task: **stack the cubes**. The four cubes of the block_bin scene are
built into one tower, in a random order, one cube on top of the other. Each episode is
recorded as a LeRobot dataset episode with the task prompt (`Stack the cubes.`) on every
frame and the subtask being worked on (`Put the red cube on the blue cube.`) as LeRobot's
subtask annotation.

## Scene

- **USD:** block_bin's, loaded from its install (`usd_path: package://block_bin/...`), so
  the table, FR3, cameras, lights and cubes are exactly block_bin's.
- **Bins:** parked out of every camera's view each episode (`config/randomize.yaml`): on
  the room floor 3 m behind the base camera, outside the top and base cameras' frustums
  and ~2.7 m from anywhere the wrist camera goes. They are dynamic rigid bodies, so they
  rest on the floor instead of floating. They stay in `dataset.tracked_objects`, so a bin
  colour in an instance mask would show one in view (none does).
- **Cubes:** uniform over x [0.10, 0.30], y [-0.25, 0.25] (scene frame; the robot base is
  at x = -0.3, so 40-60 cm in front of it), any yaw; a layout with two cubes closer than
  12 cm is redrawn.
- **Order:** a seeded draw over all 24 permutations, recorded with the episode's other
  drawn values in `meta/guide_episodes.jsonl`.

## Assumptions

1. *All* cubes in view form one tower. With four cubes on the table, "Stack the cubes."
   only has one meaning if every cube is stacked; each subtask puts one cube on top of
   the other.
2. Any order is physically valid: the cubes are identical 5 cm cubes of equal mass, so
   all 24 orders are drawn uniformly, and every pair appears in both directions.
3. The tower is built where the bottom cube spawned; no step moves the base.
4. A cube is *on* another when it is centred within 2 cm and its centre is 5 cm ± 1 cm
   higher (world z). The tree's progress check and the scene's success check share this
   test (`guide_ex.utility.stacking.is_on_top`).
5. Cubes have 4-fold symmetry about z, so a grasp never turns the wrist more than 45°;
   a cube that toppled onto another face is still grasped from its flattest axis.
6. The open gripper needs ~9 cm around a cube (8 cm opening plus fingers), so spawned
   cubes keep 12 cm between centres (~9% of uniform draws pass; redrawn otherwise).
7. Everything is reachable top-down without folding the arm onto itself: 30 cm from the
   base, a top-down reach to the table brings the forearm into the shoulder (MoveIt
   reports `fr3_link1`-`fr3_link5` contact), so cubes spawn 40-60 cm out.

## Tree

```
StackingDemonstration (PROCEDURE)
  Unclutch, LocateScene, AnnounceFirstSubtask, StartRecording
  StackCubes (TASK, loop until the tower is done)
    NextCube (SUBTASK, condition: done?)
      MeasureTower, TowerProgress     -> done | top, support, prompt
      else PutOn (SUBTASK)            "Put the red cube on the blue cube."
        AnnounceSubtask, LocateCube, Pick, CarryToSupport, Release
  GoHome, CheckSuccess, StopRecording (saved only on success)
```

## Recovery routes

The five failures judged most likely, most likely first:

| # | Failure | Detected by | Route |
|---|---------|-------------|-------|
| 1 | Grasp misses (block_bin's only failure mode in 300 episodes) | finger/cube contact after the lift (`Pick` condition) | `Regrasp`: open, measure the cube again, grasp the other pair of faces, retry `Pick` (2x) |
| 2 | An arm motion fails: no path, an aborted trajectory, or a straight-line path MoveIt can only partly compute (executed, it stopped short or ended in self-collision) | the move fails; `MoveToCartesianPose` refuses a path under 95% before moving | `*ViaRest`: detour through the rest pose (joint-space plan), retry the move (2x per move) |
| 3 | The cube slips out on the way to the tower | finger/cube contact above the tower (`CarryToSupport` condition) | `RepickDropped`: pick it up again from wherever it landed (2x) |
| 4 | A placed cube does not stay, or the tower is knocked | every loop pass measures the whole tower | the loop itself: rebuild from the lowest layer out of place (3 spare passes) |
| 5 | Anything beyond that (cube off the table, retries used up) | the tree fails | the episode is discarded; generation redraws a layout and tries again (8 attempts, then the run stops) |

Cubes spawned too close to grasp are prevented rather than recovered (assumption 6).
`test/test_solve_task.py` runs the real tree against a fake world through each route.

## Usage

```bash
ros2 launch guide_core bringup.launch.py
ros2 service call /Sim_0/Register guide_msgs/srv/RegisterScene "{path: 'cube_stack'}"
ros2 launch cube_stack bringup.launch.py
ros2 service call /Sim_0/Scene_0/generate_demonstration guide_msgs/srv/Demonstration \
  "{path: '~/dataset/cube_stack', zones: [], counts: [5]}"
```

## Dataset

Per camera (top, base, wrist): RGB, depth (`*_depth`, uint16 mm) and instance
segmentation (`*_instance`, one fixed colour per tracked object, legend in
`meta/guide_info.json`). Each frame's `task` is `Stack the cubes.`; the subtasks are
`style: subtask` rows of the `language_persistent` column, each active from its frame
until the next, e.g. read with
`lerobot.datasets.language_render.active_at(t, persistent=row, style="subtask")`.

## Maintainer
András Makány (makany.andras@uni-obuda.hu)
