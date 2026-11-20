"""The real episode tree against a fake world: robot routing, prompts, the misses and their recoveries.

Motion, jaw, sensing and recording nodes are replaced by fake physics in the scene frame: two PSMs
on tilted bases (poses in and out in each one's base, as GetPrimPose and the moves use them), a block
that follows the jaws holding it and sits on the peg top when lowered off the peg's axis.
"""
from types import SimpleNamespace

import numpy as np
import pytest
from peg_transfer import solve_task as st
from peg_transfer import targets as tg
from peg_transfer.scene import BLOCK_HALF_H, PEG_H, draw_plan, seated
from scipy.spatial.transform import Rotation as R

from guide_core.types.geometry import Point, Pose, Rotation
from guide_core.types.randomization import RandomizationRecord, Randomizer, SeedTree
from guide_ex.core.states import DemoStatus, ExecutionResult
from guide_ex.steps.end_effector.gripper_control import SetGripperState
from guide_ex.steps.manipulation.cartesian_move import MoveToCartesianPose
from guide_ex.steps.manipulation.joint_move import MoveToJointConfiguration
from guide_ex.steps.simulation.isaac.prim import GetPrimPose, IsPrimClashing
from guide_ex.steps.simulation.success import IsTaskSuccessful
from guide_ex.utility import recording
from guide_ex.utility.wait import WaitForSeconds

PERFECT = ExecutionResult(DemoStatus.PERFECT)
BASES = {  # davinci_sim's RCMs, each base tilted toward the board
    "psm1": Pose(Point([-0.13, 0.82, 0.95]), Rotation(R.from_euler("y", -30, degrees=True))),
    "psm2": Pose(Point([0.13, 0.82, 0.95]), Rotation(R.from_euler("y", 30, degrees=True))),
    "ecm": Pose(Point([0.0, 0.9, 1.0])),
}
HOME = Pose(Point([0.0, 0.0, -0.177]))  # the tip at the home joints, base frame
MARKER = Pose(Point([0.005, 0.72, 0.85]), Rotation(R.from_euler("x", 20, degrees=True)))


def make_plan(seed, giver):
    for s in range(seed, seed + 100):
        rng, _ = SeedTree.create(1).generator(s)
        plan = draw_plan(Randomizer(rng, RandomizationRecord(seed=s)))
        if plan["giver"] == giver:
            return plan


class World:
    def __init__(self, plan):
        sx = -1 if plan["giver"] == "psm1" else 1  # the giver's pegs are on its side
        self.pegs = {plan["source_peg"]: np.array([sx * 0.043, 0.69, 0.80 + PEG_H / 2]),
                     plan["target_peg"]: np.array([-sx * 0.043, 0.75, 0.80 + PEG_H / 2])}
        self.plan, self.target = plan, self.pegs[plan["target_peg"]]
        self.block = self.pegs[plan["source_peg"]] + [0, 0, BLOCK_HALF_H - PEG_H / 2]
        self.yaw = R.from_euler("z", plan["yaw"][plan["layout"].index(plan["colour"])], degrees=True)
        self.tips = {k: BASES[k] * HOME for k in ("psm1", "psm2")}
        self.holders, self.prompts, self.calls, self.saved = set(), [], [], None

    def world(self, prim):
        if prim == self.plan["block"]:
            return Pose(Point(self.block), Rotation(self.yaw))
        if prim in self.pegs:
            return Pose(Point(self.pegs[prim]))
        return MARKER if prim == "/PegTransfer/handover" else Pose()  # /PegTransfer: identity

    def grip_point(self, tip):
        return tip.position.to_numpy() - tip.orientation.to_numpy() @ tg.GRIP  # the held block's centre

    def move_tip(self, arm, tip):
        self.tips[arm] = tip
        if arm in self.holders:
            centre = self.grip_point(tip)
            off_axis = np.hypot(*(centre[:2] - self.target[:2])) > 0.0025
            on_top = self.target[2] + PEG_H / 2 + BLOCK_HALF_H
            if off_axis and np.hypot(*(centre[:2] - self.target[:2])) < 0.0065 and centre[2] < on_top:
                centre[2] = on_top  # the hole missed the peg: the block rests on its top
            self.block = centre


@pytest.fixture
def world(monkeypatch):
    holder = {}

    def arm(robot):
        return robot.id

    def get_pose(self, robot, sim_namespace, scene_namespace, prim_path):
        w = holder["w"]
        w.calls.append((self.name, arm(robot)))
        return ExecutionResult(DemoStatus.PERFECT, outputs={"pose": BASES[arm(robot)].inv() * w.world(prim_path)})

    def move(self, robot, target_pose, speed=1.0, cartesian=False, perturb=False, precise_end=True, jitter=None, aim=False):
        w = holder["w"]
        w.calls.append((self.name, arm(robot)))
        assert 0.05 <= speed <= 0.2, (self.name, speed)
        w.move_tip(arm(robot), BASES[arm(robot)] * target_pose)
        return PERFECT

    def joints(self, robot, target_configuration, speed=1.0):
        w = holder["w"]
        w.calls.append((self.name, arm(robot)))
        if arm(robot) != "ecm":
            w.move_tip(arm(robot), BASES[arm(robot)] * HOME)
        return PERFECT

    def grip(self, robot, gripper_goal_pos):
        w = holder["w"]
        w.calls.append((self.name, arm(robot)))
        ((joint, q),) = gripper_goal_pos.items()
        assert joint == "psm_tool_gripper1_joint" and q in (-0.5, 0.0)  # 1 rad open via description.json's scale
        if q == 0.0 and np.linalg.norm(w.grip_point(w.tips[arm(robot)]) - w.block) < 0.005:
            w.holders.add(arm(robot))
        elif q != 0.0:
            w.holders.discard(arm(robot))
        return PERFECT

    def clashing(self, robot, sim_namespace, scene_namespace, prim1_path, prim2_path):
        w = holder["w"]
        held_by = "psm1" if prim1_path.startswith("/PSM1/") else "psm2"
        return ExecutionResult(DemoStatus.PERFECT, outputs={"has_collided": prim2_path == w.plan["block"] and held_by in w.holders})

    def prompt(self, robot, sim_namespace, scene_id, task="", subtask="", timeout_sec=30.0):
        holder["w"].prompts += [(level, p) for level, p in (("task", task), ("subtask", subtask)) if p]
        return PERFECT

    def stop(self, robot, sim_namespace, scene_id, save_episode=True, timeout_sec=60.0):
        holder["w"].saved = save_episode
        return PERFECT

    def success(self, robot, sim_namespace, scene_id):
        w = holder["w"]
        ok = seated(Pose(Point(w.block)), Pose(Point(w.target))) and not w.holders
        return ExecutionResult(DemoStatus.PERFECT, outputs={"success": ok, "reason": "" if ok else "not seated"})

    for cls, run in {GetPrimPose: get_pose, MoveToCartesianPose: move, MoveToJointConfiguration: joints,
                     SetGripperState: grip, IsPrimClashing: clashing, WaitForSeconds: lambda self, seconds, timer=None: PERFECT,
                     recording.SetPrompt: prompt, recording.StartRecording: lambda self, *a, **k: PERFECT,
                     recording.StopRecording: stop, IsTaskSuccessful: success}.items():
        monkeypatch.setattr(cls, "run", run)
    return holder


ROBOTS = {k: SimpleNamespace(id=k) for k in ("psm1", "psm2", "ecm")}


def run(holder, plan, miss=None, recoveries=True):
    holder["w"] = w = World(plan)
    tree = st.build_tree(st.init_yaml(), plan, miss=miss)
    if not recoveries:
        st.without_recoveries(tree)
    result = tree.execute(st.episode_context(ROBOTS, "/Sim_0", 0))
    return w, result


@pytest.mark.parametrize("giver", ["psm1", "psm2"])
def test_both_directions_transfer_the_block_each_node_on_its_arm(world, giver):
    plan = make_plan(0, giver)
    w, result = run(world, plan)

    assert result.status == DemoStatus.PERFECT and result.outputs["task_success"] and w.saved
    receiver = plan["receiver"]
    by = dict(w.calls)  # the last arm each node ran on
    for name in ("LocateBlock", "MoveOverBlock", "MoveToGrasp", "CloseOnBlock", "Lift", "TurnLevel", "Give",
                 "GiverLetsGo", "GiverBacksOff", "GiverHome"):
        assert by[name] == giver, name
    for name in ("ReceiverOpens", "ApproachReceive", "MoveToReceive", "ReceiverCloses", "LocatePeg", "TurnDown",
                 "AlignOverPeg", "Lower", "Release", "Retreat", "ReceiverHome"):
        assert by[name] == receiver, name
    assert by["HoldCamera"] == "ecm"
    # arms move one at a time: the receiver's first move comes after the giver's last before it lets go
    names = [n for n, _ in w.calls]
    assert names.index("ApproachReceive") > names.index("Give") and names.index("GiverBacksOff") > names.index("ReceiverCloses")


def test_the_prompts_at_three_levels(world):
    plan = make_plan(0, "psm1")
    w, _ = run(world, plan)

    assert plan["task"] == "Transfer the block to the other side."  # the procedure, recorded by the scene
    assert [p for level, p in w.prompts if level == "task"] == plan["tasks"]
    subtasks = [p for level, p in w.prompts if level == "subtask"]
    assert list(dict.fromkeys(subtasks)) == list(st.SUBTASKS.values())


@pytest.mark.parametrize("stage,recovery", [("grasp", "BackOverBlock"), ("handover", "BackToReceivePre"),
                                            ("place", "LiftOffPegTop")])
def test_each_planned_miss_fails_once_and_its_recovery_finishes_the_transfer(world, stage, recovery):
    plan = make_plan(3, "psm2")
    miss, stages = st.planned_misses({"failures": {"counts": [1], "weights": [1.0]},
                                      "stages": {stage: st.variant("light_fail")["stages"][stage]}},
                                     plan["colour"], np.random.default_rng(0))
    w, result = run(world, plan, miss={**miss, "seed": 5})

    assert stages == [stage] and result.outputs["task_success"] and w.saved
    names = [n for n, _ in w.calls]
    assert names.count(recovery) == 1

    w, result = run(world, plan, miss={**miss, "seed": 5}, recoveries=False)
    assert result.status == DemoStatus.FAILURE and recovery not in [n for n, _ in w.calls]


def test_light_fail_plans_exactly_one_miss_and_retry_mixed_one_or_two():
    light, mixed = st.variant("light_fail"), st.variant("retry_mixed")
    assert st.variant("flawless") == {"miss": {}} and st.planned_misses(st.variant("flawless"), "red", None) == ({}, [])
    counts = set()
    for seed in range(60):
        rng = np.random.default_rng(seed)
        miss, stages = st.planned_misses(light, "red", rng)
        assert len(stages) == 1 and stages[0] in ("grasp", "handover", "place")
        assert miss == {t: {"at": {"red": s}} for t, s in light["stages"][stages[0]].items()}
        miss, stages = st.planned_misses(mixed, "red", rng)
        assert len(stages) == len(set(stages)) == len(miss) and set(miss) <= {"grasp", "receive", "place"}
        counts.add(len(stages))
    assert counts == {1, 2}


def test_without_recoveries_strips_every_fallback():
    tree = st.without_recoveries(st.build_tree(st.init_yaml(), make_plan(0, "psm1")))
    stack = [tree]
    while stack:
        node = stack.pop()
        assert not getattr(node, "fallbacks", None), node.name
        stack += [c for c in [*getattr(node, "children", []), getattr(node, "true_branch", None),
                              getattr(node, "false_branch", None)] if c is not None]


def test_the_ecm_hold_is_a_jittered_joint_target():
    from guide_core.scene.robot_profile import resolve

    entry = resolve(st.init_yaml()["robots"]["ecm"])
    for seed in range(20):
        hold = st.ecm_hold(entry, np.random.default_rng(seed))
        q = np.array(list(hold.values()))
        assert list(hold) == [f"{j}.pos" for j in entry["joint_names"]]
        assert np.all(np.abs(q - entry["default_joint_states"]) <= st.ECM_JITTER)
    assert st.ecm_hold(entry, None) == dict(zip(hold, map(float, entry["default_joint_states"])))


@pytest.mark.parametrize("recipe", ["flawless", "light_fail", "retry_mixed"])
def test_solve_task_randomizes_draws_the_variant_and_saves(world, recipe):
    import json

    plan = make_plan(7, "psm1")
    world["w"] = World(plan)
    logger = SimpleNamespace(info=lambda m: None, error=lambda m: None)
    robots = {k: SimpleNamespace(id=k, randomize="randomize", node=SimpleNamespace(get_logger=lambda: logger),
                                 callService=lambda client, request, **kw: SimpleNamespace(success=True, message=json.dumps(plan)),
                                 motion_episode=lambda *key, clean=False: None, sleep=lambda s: None)
              for k in ("psm1", "psm2", "ecm")}

    saved, info = st.solve_task(0, robots, "/Sim_0", st.init_yaml(), draw=3, scene_seed=11, recipe=recipe)

    assert saved and info["recipe"] == recipe and info["giver"] == "psm1"
    assert {"flawless": 0, "light_fail": 1}.get(recipe, len(info["misses"])) == len(info["misses"]) <= 2
