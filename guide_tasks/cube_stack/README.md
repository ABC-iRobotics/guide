# Cube Stack Task

GUIDE demonstration task: **stack the cubes**. The four cubes of the block_bin scene are
built into one tower, in a random order, one cube on top of the other. Each episode is
recorded as a LeRobot dataset episode with the task prompt (`Stack the cubes.`) on every
frame and the subtask being worked on (`Put the red cube on the blue cube.`) as LeRobot's
subtask annotation.

## Scene

- **USD:** block_bin's, loaded from its install (`usd_path: package://block_bin/...`), so
  the table, FR3, cameras, lights and cubes are exactly block_bin's.
- **Randomization:** `config/randomize.yaml` and `config/reset.yaml` are Replicator YAML
  (guide_core `replicator_guide`, `docs/design/replicator-front-end.md`); `success.yaml`
  stays on the instruction executor.
- **Cubes:** uniform over scene x [0.10, 0.30], y [-0.25, 0.25] (40-60 cm in front of the
  robot base, which is at x = -0.3), any yaw; a layout with two cubes closer than 12 cm is
  redrawn. The file gives the region in the frame of `/blocks`, which block_bin.usd turns
  -90° about z (local x = -scene y, local y = scene x).
- **Starting cube on a grid:** the tower's bottom cube is the zone target. Its region is
  cut into 0.1 m cells, 5 x 2 = 10 zones; a request names zones (`zones: [0, 6]`, or
  `[-1]` for every zone) and the bottom cube, so the tower, is drawn inside that cell.
  `zones: []` draws it anywhere in the region.
- **Order:** a seeded draw over all 24 permutations, recorded with the episode's other
  drawn values in `meta/guide_episodes.jsonl`.
- **Bins:** not part of the task. Each episode they are parked on the room floor 3 m
  behind the base camera (they are dynamic bodies) and hidden. Hiding is what keeps them
  out of every frame: the wrist camera looks forward along the table, and from the floor
  spot alone it saw a bin in 20 of 1597 frames. They stay in `dataset.tracked_objects`,
  so a bin colour in an instance mask would show one.

## Assumptions

1. *All* cubes in view form one tower. With four cubes on the table, "Stack the cubes."
   only has one meaning if every cube is stacked; each subtask puts one cube on top of
   the other.
2. Any order is physically valid: the cubes are identical 5.15 cm cubes of equal mass, so
   all 24 orders are drawn uniformly, and every pair appears in both directions.
3. The tower is built where the bottom cube spawned; no step moves the base.
4. A cube is *on* another when it sits one cube edge (5.15 cm) above it, within 2 cm
   sideways and 1 cm in height (world z). The tree's progress check (`ChainLength`) and
   the scene's success check share this test (`guide_ex.utility.pose.is_at_offset`).
5. Cubes have 4-fold symmetry about z, so a grasp never turns the wrist more than 45°;
   a cube that toppled onto another face is still grasped from its flattest axis.
6. The open gripper needs ~9 cm around a cube (8 cm opening plus fingers), so spawned
   cubes keep 12 cm between centres (~9% of uniform draws pass; redrawn otherwise).
7. Everything is reachable top-down without folding the arm onto itself: 30 cm from the
   base, a top-down reach to the table brings the forearm into the shoulder (MoveIt
   reports `fr3_link1`-`fr3_link5` contact). From the home joints a straight descent is
   feasible over the whole 40-60 cm region (mapped with MoveIt's IK and Cartesian path).

## Tree

Built from reusable GUIDE-EX nodes only; nothing in it is specific to cubes or towers:

```
StackingDemonstration (PROCEDURE)
  Unclutch, LocateScene, AnnounceFirstSubtask, StartRecording
  StackCubes (TASK, loop until the tower is done)
    NextCube (SUBTASK, condition: done?)
      MeasureTower (GetPrimPoses), TowerHeight (ChainLength)   -> built, done
      else PutOn (SUBTASK)    "Put the red cube on the blue cube."
        NextTop, NextSupport, NextPrompt (GetItem: order[built], order[built - 1], ...)
        AnnounceSubtask (SetSubtaskPrompt), LocateCube, Pick, CarryToSupport, Release
  GoHome, CheckSuccess, StopRecording (saved only on success)
```

Rest, detours and the final home are the home *joint configuration*
(`MoveToJointConfiguration`): a 7-DoF arm reaches a pose in many postures and each
Cartesian move keeps the current one, so postures drift; a rest *pose* lets the planner
pick any of them. One episode's last pick failed six times in a row that way.

## Recovery routes

The five failures judged most likely, most likely first:

| # | Failure | Detected by | Route |
|---|---------|-------------|-------|
| 1 | Grasp misses (block_bin's only failure mode in 300 episodes) | finger/cube contact after the lift (`Pick` condition) | `Regrasp`: open, measure the cube again, grasp the other pair of faces, retry `Pick` (2x) |
| 2 | An arm motion fails: no path, an aborted trajectory, or a straight-line path MoveIt can only partly compute (executed, it stopped short or ended in self-collision) | the move fails; `MoveToCartesianPose` refuses a path under 95% before moving | `*ViaHome`: detour through the home joints, retry the move (2x per move) |
| 3 | The cube does not get to the tower: it slips out, or the arm finds no way there | finger/cube contact above the tower (`CarryToSupport` condition), or the carry's own detours run out | `RepickDropped`: set it down where it was picked (opening anywhere else drops it from height), pick it up again from wherever it is (2x) |
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
  "{path: '~/dataset/cube_stack', zones: [0, 6, 2, 8, 4], counts: [1]}"
```

## Dataset

Per camera (top, base, wrist): RGB, depth (`*_depth`, uint16 mm) and instance
segmentation (`*_instance`, one fixed colour per tracked object, legend in
`meta/guide_info.json`). Each frame's `task` is `Stack the cubes.`; the subtasks are
`style: subtask` rows of the `language_persistent` column, each active from its frame
until the next, e.g. read with
`lerobot.datasets.language_render.active_at(t, persistent=row, style="subtask")`.
`meta/guide_episodes.jsonl` holds each episode's zone, seed and drawn values.

## Maintainer
András Makány (makany.andras@uni-obuda.hu)
