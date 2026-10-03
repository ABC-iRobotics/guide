"""The real episode tree against a fake world: wiring, prompts and every recovery route.

Each motion, gripper, sensing and recording node is replaced by a few lines of fake
physics (cubes follow a closed gripper, fall onto whatever is below when let go), so
the tree runs exactly as built for Isaac, minus Isaac.
"""
from types import SimpleNamespace

import numpy as np
import pytest
from cube_stack import solve_task as st
from cube_stack.scene import subtask_prompts

from guide_core.types.geometry import Point, Pose
from guide_ex.core.states import DemoStatus, ExecutionResult
from guide_ex.steps.end_effector.gripper_control import SetGripperState
from guide_ex.steps.manipulation.cartesian_move import MoveToCartesianPose
from guide_ex.steps.simulation.isaac.prim import GetPrimPose, IsPrimClashing
from guide_ex.steps.simulation.success import IsTaskSuccessful
from guide_ex.utility import recording
from guide_ex.utility.stacking import is_on_top
from guide_ex.utility.wait import WaitForSeconds

COLOURS = ["blue", "red", "green", "yellow"]  # this episode's order, bottom first
ORDER = [f"/blocks/{c}_block" for c in COLOURS]
PERFECT = ExecutionResult(DemoStatus.PERFECT)


class World:
    def __init__(self, misses=0, slips=0, slides=0, failing_moves=()):
        spots = [(0.0, -0.2), (0.15, 0.2), (-0.1, 0.1), (0.2, -0.05)]
        self.cubes = {p: np.array([x, y, 0.025]) for p, (x, y) in zip(ORDER, spots)}
        self.tcp, self.held = np.array([0.0, 0.0, 0.5]), None
        self.misses, self.slips, self.slides = misses, slips, slides
        self.failing_moves = list(failing_moves)
        self.prompts, self.moves, self.saved = [], [], None

    def settle(self, cube):
        """Let go of `cube`: it lands on the highest cube under it, or the table."""
        x, y = self.cubes[cube][:2]
        if self.slides:  # it slides off whatever it was put on
            self.slides -= 1
            x += 0.08
        below = [c[2] for p, c in self.cubes.items() if p != cube and np.hypot(*(c[:2] - (x, y))) < 0.03]
        self.cubes[cube] = np.array([x, y, max(below, default=-0.025) + 0.05])

    def tower(self):
        poses = [Pose(position=Point(self.cubes[p])) for p in ORDER]
        return all(is_on_top(u, lo, 0.05, 0.02, 0.01) for lo, u in zip(poses, poses[1:]))


@pytest.fixture
def world(monkeypatch):
    w = World()

    def get_pose(self, robot, sim_namespace, scene_namespace, prim_path):
        position = w.cubes[prim_path] if prim_path else np.zeros(3)
        return ExecutionResult(DemoStatus.PERFECT, outputs={"pose": Pose(position=Point(position))})

    def move(self, robot, target_pose, speed=1.0, cartesian=False):
        w.moves.append(self.name)
        if self.name in w.failing_moves:
            w.failing_moves.remove(self.name)
            return ExecutionResult(DemoStatus.FAILURE, error_message="no path")
        w.tcp = target_pose.position.to_numpy()
        if w.held:
            w.cubes[w.held] = w.tcp - (0, 0, st.GRASP)
            if w.slips and self.name == "MoveOverSupport":
                w.slips -= 1
                w.cubes[w.held] = w.cubes[w.held] + (0.1, 0.0, 0.0)
                w.settle(w.held)
                w.held = None
        return PERFECT

    def grip(self, robot, gripper_goal_pos):
        (width,) = gripper_goal_pos.values()
        if width == st.CLOSED:
            under = [p for p, c in w.cubes.items() if np.allclose(c + (0, 0, st.GRASP), w.tcp)]
            if under and not w.misses:
                w.held = under[0]
            w.misses = max(0, w.misses - 1)
        elif w.held:
            w.settle(w.held)
            w.held = None
        return PERFECT

    def clashing(self, robot, sim_namespace, scene_namespace, prim1_path, prim2_path):
        return ExecutionResult(DemoStatus.PERFECT, outputs={"has_collided": w.held == prim2_path})

    def prompt(self, robot, sim_namespace, scene_id, prompt, timeout_sec=30.0):
        w.prompts.append(prompt)
        return PERFECT

    def stop(self, robot, sim_namespace, scene_id, save_episode=True, timeout_sec=60.0):
        w.saved = save_episode
        return PERFECT

    patches = {
        GetPrimPose: get_pose,
        MoveToCartesianPose: move,
        SetGripperState: grip,
        IsPrimClashing: clashing,
        WaitForSeconds: lambda self, seconds, timer=None: PERFECT,
        recording.SetSubtaskPrompt: prompt,
        recording.StartRecording: lambda self, *a, **k: PERFECT,
        recording.StopRecording: stop,
        IsTaskSuccessful: lambda self, *a, **k: ExecutionResult(
            DemoStatus.PERFECT, outputs={"success": w.tower(), "reason": ""}
        ),
    }
    for cls, run in patches.items():
        monkeypatch.setattr(cls, "run", run)
    return w


def run(world):
    robot = SimpleNamespace(config=SimpleNamespace(gripper_joint_names=["fr3_finger_joint1"]))
    prompts = subtask_prompts(COLOURS)
    tree = st.build_tree(robot, prompts, len(ORDER))
    return tree.execute(
        {
            "robot": robot,
            "sim_namespace": "/Sim_0",
            "scene_namespace": "/Scene_0",
            "scene_id": 0,
            "scene_path": "",
            "robot_prim": "/fr3/fr3_rightfinger",
            "rest_pose": Pose(position=Point([0.0, 0.0, 0.5])),
            "dataset_path": "",
            "order": ORDER,
            "prompts": prompts,
        }
    )


def test_builds_the_tower_announcing_each_subtask(world):
    result = run(world)

    assert result.status == DemoStatus.PERFECT
    assert world.tower() and world.saved is True
    prompts = subtask_prompts(COLOURS)
    assert world.prompts == [prompts[0], *prompts]


def test_route_1_a_missed_grasp_is_regrasped(world):
    world.misses = 1

    assert run(world).status == DemoStatus.PERFECT
    assert world.tower() and world.moves.count("MoveToGrasp") == 4


def test_route_2_a_failed_move_detours_via_rest(world):
    world.failing_moves = ["LowerOntoSupport", "MoveOverCube"]

    assert run(world).status == DemoStatus.PERFECT
    assert world.tower()
    assert {"LowerViaRestToRest", "OverCubeViaRestToRest"} <= set(world.moves)


def test_route_3_a_cube_dropped_on_the_way_is_picked_up_again(world):
    world.slips = 1

    assert run(world).status == DemoStatus.PERFECT
    assert world.tower() and world.moves.count("MoveToGrasp") == 4


def test_route_4_a_placement_that_slides_off_is_redone(world):
    world.slides = 2

    assert run(world).status == DemoStatus.PERFECT
    assert world.tower() and world.moves.count("LowerOntoSupport") == 5


def test_route_5_beyond_recovery_the_episode_fails(world):
    world.misses = 99

    assert run(world).status == DemoStatus.FAILURE
    assert world.saved is None  # never stopped by the tree: generation discards it
