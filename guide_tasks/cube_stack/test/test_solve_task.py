"""The real episode tree against a fake world: wiring, prompts and every recovery route.

Each motion, gripper, sensing and recording node is replaced by a few lines of fake
physics (cubes follow a closed gripper, fall onto whatever is below when let go), so
the tree runs exactly as built for Isaac, minus Isaac.
"""

from types import SimpleNamespace

import numpy as np
import pytest
from cube_stack import solve_task as st
from cube_stack.scene import plan

from guide_core.types.geometry import Point, Pose
from guide_ex.core.states import DemoStatus, ExecutionResult
from guide_ex.steps.end_effector.gripper_control import SetGripperState
from guide_ex.steps.manipulation.cartesian_move import MoveToCartesianPose
from guide_ex.steps.manipulation.joint_move import MoveToJointConfiguration
from guide_ex.steps.simulation.isaac.prim import GetPrimPose, IsPrimClashing
from guide_ex.steps.simulation.success import IsTaskSuccessful
from guide_ex.utility import recording
from guide_ex.utility.pose import is_at_offset
from guide_ex.utility.wait import WaitForSeconds

COLOURS = ["blue", "red", "green", "yellow"]  # this episode's order, bottom first
ORDER = [f"/blocks/{c}_block" for c in COLOURS]
PERFECT = ExecutionResult(DemoStatus.PERFECT)


class World:
    def __init__(self):
        spots = [(0.0, -0.2), (0.15, 0.2), (-0.1, 0.1), (0.2, -0.05)]
        self.cubes = {p: np.array([x, y, 0.025]) for p, (x, y) in zip(ORDER, spots)}
        self.tcp, self.held = np.array([0.0, 0.0, 0.5]), None
        self.misses = self.slips = self.slides = 0
        self.failing_moves = []
        self.prompts, self.moves, self.saved = [], [], None
        self.scattered = 0  # cubes let go of from height

    def landing(self, cube, x, y):
        below = [
            c[2] for p, c in self.cubes.items() if p != cube and np.hypot(*(c[:2] - (x, y))) < 0.03
        ]
        return max(below, default=-0.025) + 0.05

    def settle(self, cube):
        """Let go of `cube`: it lands on the highest cube under it, or the table. Slid
        off, or dropped from more than 2 cm, it ends up 8 cm aside."""
        x, y, z = self.cubes[cube]
        if z - self.landing(cube, x, y) > 0.02:
            self.scattered += 1
        if self.slides or z - self.landing(cube, x, y) > 0.02:
            self.slides = max(0, self.slides - 1)
            x += 0.08
        self.cubes[cube] = np.array([x, y, self.landing(cube, x, y)])

    def tower(self):
        poses = [Pose(position=Point(self.cubes[p])) for p in ORDER]
        return all(is_at_offset(u, lo, (0, 0, 0.05), (0.02, 0.02, 0.01)) for lo, u in zip(poses, poses[1:]))


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

    def home(self, robot, target_configuration, speed=1.0):
        w.moves.append(self.name)
        w.tcp = np.array([0.0, 0.0, 0.5])
        if w.held:
            w.cubes[w.held] = w.tcp - (0, 0, st.GRASP)
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

    def prompt(self, robot, sim_namespace, scene_id, task="", subtask="", timeout_sec=30.0):
        w.prompts += [(level, p) for level, p in (("task", task), ("subtask", subtask)) if p]
        return PERFECT

    def stop(self, robot, sim_namespace, scene_id, save_episode=True, timeout_sec=60.0):
        w.saved = save_episode
        return PERFECT

    patches = {
        GetPrimPose: get_pose,
        MoveToCartesianPose: move,
        MoveToJointConfiguration: home,
        SetGripperState: grip,
        IsPrimClashing: clashing,
        WaitForSeconds: lambda self, seconds, timer=None: PERFECT,
        recording.SetPrompt: prompt,
        recording.StartRecording: lambda self, *a, **k: PERFECT,
        recording.StopRecording: stop,
        IsTaskSuccessful: lambda self, *a, **k: ExecutionResult(
            DemoStatus.PERFECT, outputs={"success": w.tower(), "reason": ""}
        ),
    }
    for cls, run in patches.items():
        monkeypatch.setattr(cls, "run", run)
    return w


ROBOT = SimpleNamespace(config=SimpleNamespace(gripper_joint_names=["fr3_finger_joint1"]))


def run(world):
    drawn = plan(COLOURS)
    tree = st.build_tree(ROBOT, drawn)
    return tree.execute(st.episode_context(ROBOT, "/Sim_0", 0, drawn))


def test_builds_the_tower_announcing_each_layer(world):
    result = run(world)

    assert result.status == DemoStatus.PERFECT
    assert world.tower() and world.saved is True
    drawn = plan(COLOURS)
    tasks, sub = drawn["tasks"], drawn["subtasks"]
    # PutOn sets its task and first subtask in one call; Pick then announces itself (a
    # no-op here, but it is what puts the pick back after a set-down recovery).
    placements = [
        prompt
        for k in range(len(tasks))
        for prompt in (("task", tasks[k]), ("subtask", sub["pick"][k]),
                       ("subtask", sub["pick"][k]), ("subtask", sub["place"][k]))
    ]
    assert world.prompts == [
        ("task", tasks[0]), ("subtask", sub["pick"][0]),  # before the first frame
        *placements,
        ("task", "Stack the cubes."), ("subtask", "Return home."),  # the procedure closes
    ]


def test_the_layers_are_guide_exs():
    """The procedure stacks, each placement is a task, the pick and the place its subtasks."""
    root = st.build_tree(ROBOT, plan(COLOURS))
    found = {}

    def walk(node):
        found[getattr(node, "name", None)] = getattr(node, "level", None)
        for child in [*getattr(node, "children", []), getattr(node, "true_branch", None),
                      getattr(node, "false_branch", None), *getattr(node, "fallbacks", {}).values()]:
            if child is not None:
                walk(child)

    walk(root)
    L = st.Layer
    assert {k: found[k] for k in ("StackingDemonstration", "BuildTower", "NextCube", "PutOn", "Pick",
                                  "Place", "Regrasp", "RepickDropped", "Finish", "LocateCube",
                                  "CarryToSupport", "Release")} == {
        "StackingDemonstration": L.PROCEDURE, "BuildTower": L.PROCEDURE, "NextCube": L.TASK,
        "PutOn": L.TASK, "Pick": L.SUBTASK, "Place": L.SUBTASK, "Regrasp": L.SUBTASK,
        "RepickDropped": L.SUBTASK, "Finish": L.SUBTASK, "LocateCube": L.SEQUENCE,
        "CarryToSupport": L.SEQUENCE, "Release": L.SEQUENCE,
    }


def test_route_1_a_missed_grasp_is_regrasped(world):
    world.misses = 1

    assert run(world).status == DemoStatus.PERFECT
    assert world.tower() and world.moves.count("MoveToGrasp") == 4


def test_route_2_a_failed_move_detours_through_home(world):
    world.failing_moves = ["LowerOntoSupport", "MoveOverCube"]

    assert run(world).status == DemoStatus.PERFECT
    assert world.tower()
    assert {"LowerViaHomeToHome", "OverCubeViaHomeToHome"} <= set(world.moves)


def test_route_3_a_cube_dropped_on_the_way_is_picked_up_again(world):
    world.slips = 1

    assert run(world).status == DemoStatus.PERFECT
    assert world.tower() and world.moves.count("MoveToGrasp") == 4


def test_route_3_with_no_way_to_the_tower_the_cube_is_set_down_not_dropped(world):
    world.failing_moves = ["MoveOverSupport"] * 3  # the move and both its detours

    assert run(world).status == DemoStatus.PERFECT
    assert world.tower() and "SetDown" in world.moves
    assert world.scattered == 0


def test_route_4_a_placement_that_slides_off_is_redone(world):
    world.slides = 2

    assert run(world).status == DemoStatus.PERFECT
    assert world.tower() and world.moves.count("LowerOntoSupport") == 5


def test_route_5_beyond_recovery_the_episode_fails(world):
    world.misses = 99

    assert run(world).status == DemoStatus.FAILURE
    assert world.saved is None  # never stopped by the tree: generation discards it


@pytest.mark.parametrize("trouble", [{}, {"misses": 1}, {"slips": 1}, {"slides": 2}])
def test_the_oracle_names_the_task_and_subtask_the_demonstration_announces(world, monkeypatch, trouble):
    """Asked from the scene alone, the oracle agrees with the tree at both levels: at every
    announcement (re-announcements after a regrasp or a cube that slid off included) and
    all the while a cube is carried. The set-down recovery is a decision the scene does
    not show; the oracle is not asked to see it."""
    from cube_stack.oracle import SceneOracle

    for name, value in trouble.items():
        setattr(world, name, value)
    drawn = plan(COLOURS)
    oracle = SceneOracle(ROBOT, "/Sim_0", 0, drawn)
    announced, carried = [], []
    announce, move = recording.SetPrompt.run, MoveToCartesianPose.run

    def announce_and_ask(self, robot, sim_namespace, scene_id, task="", subtask="", timeout_sec=30.0):
        for level, prompt in (("task", task), ("subtask", subtask)):
            if prompt and prompt not in drawn["subtasks"]["set_down"]:
                announced.append((prompt, oracle()[level]))
        return announce(self, robot, sim_namespace, scene_id, task, subtask, timeout_sec)

    def move_and_ask(self, robot, target_pose, speed=1.0, cartesian=False):
        result = move(self, robot, target_pose, speed, cartesian)
        subtask = next((p for level, p in reversed(world.prompts) if level == "subtask"), None)
        # The lift ends Pick: Place is announced right after it, with no motion between.
        if world.held and self.name != "LiftCube" and subtask not in drawn["subtasks"]["set_down"]:
            carried.append((subtask, oracle()["subtask"]))
        return result

    grip = SetGripperState.run

    def grip_and_ask(self, robot, gripper_goal_pos):
        result = grip(self, robot, gripper_goal_pos)
        if world.held:  # just closed on it, still on the table: that is the pick
            carried.append((world.prompts[-1][1], oracle()["subtask"]))
        return result

    monkeypatch.setattr(recording.SetPrompt, "run", announce_and_ask)
    monkeypatch.setattr(MoveToCartesianPose, "run", move_and_ask)
    monkeypatch.setattr(SetGripperState, "run", grip_and_ask)

    assert run(world).status == DemoStatus.PERFECT
    assert announced and all(said == asked for said, asked in announced)
    assert carried and all(said == asked for said, asked in carried)
    said = {s for s, _ in carried}
    assert said >= set(drawn["subtasks"]["place"]) and said & set(drawn["subtasks"]["pick"])
    assert oracle() == {"task": "Stack the cubes.", "subtask": "Return home."} and oracle.done
