"""LIBERO-Goal's skills (S3 v2 G6a-d): Pull, Turn, Push, PutIn, PutOnTilted and the leaves under them."""
import numpy as np
import pytest
from scipy.spatial.transform import Rotation as R

from guide_core.types.geometry import Pose
from guide_ex.core.base_node import BaseNode
from guide_ex.core.states import DemoStatus, ExecutionResult, Layer
from guide_ex.recipe import load
from guide_ex.utility.pose import PoseDisplacement, PushPoses, RotateAbout

VALUES = {"robot": {"gripper_joint": "fr3_finger_joint1", "home": {f"fr3_joint{i}.pos": 0.0 for i in range(1, 8)}},
          "episode": {"camera_hold": False, "hold_seconds": 0.0}}


def pose(xyz, rot=None):
    return Pose.from_numpy(np.array(xyz, float), rot or R.identity())


def child(node, name):
    return next(c for c in node.children if c.name == name)


def test_displacement_reads_a_slide_and_a_hinge():
    out = PoseDisplacement().run(pose([0, -0.12, 0.3]), pose([0, 0, 0.3])).outputs
    assert np.isclose(out["distance"], 0.12) and np.isclose(out["angle"], 0.0)
    out = PoseDisplacement().run(pose([0, 0, 0], R.from_euler("z", -0.7)), pose([0, 0, 0], R.from_euler("z", 0.2))).outputs
    assert np.isclose(out["angle"], 0.9) and np.isclose(out["distance"], 0.0)


@pytest.mark.parametrize("direction", [1.0, -1.0])
def test_rotate_about_turns_by_the_signed_angle(direction):
    about = pose([1.0, 2.0, 0.0], R.from_euler("z", 0.3))
    out = RotateAbout().run(about * pose([0.06, 0, 0]), about, 0.15, direction).outputs["pose"]
    rel = (about.inv() * out)
    assert np.isclose(np.hypot(*rel.position.to_numpy()[:2]), 0.06)
    assert np.isclose(rel.orientation.to_scipy().as_rotvec()[2], direction * 0.15)


def test_push_poses_run_along_the_line_to_the_goal_with_the_hand_down():
    out = PushPoses().run(pose([0.5, 0.1, 0.02]), pose([0.5, 0.4, 0.02]), reach=0.1, gap=0.05, height=0.015).outputs
    start, end = out["start_pose"], out["end_pose"]
    np.testing.assert_allclose(start.position.to_numpy(), [0.5, -0.05, 0.035], atol=1e-9)
    np.testing.assert_allclose(end.position.to_numpy(), [0.5, 0.3, 0.035], atol=1e-9)
    hand = end.orientation.to_scipy().as_matrix()
    np.testing.assert_allclose(hand[:, 0], [0, 1, 0], atol=1e-9)  # x along the push
    np.testing.assert_allclose(hand[:, 2], [0, 0, -1], atol=1e-9)  # fingers down
    assert PushPoses().run(pose([0, 0, 0]), pose([0, 0, 0])).status == DemoStatus.FAILURE


def test_the_skills_load_at_subtask_and_name_their_context():
    for name, keys in {"Pull": {"part", "pull"}, "Turn": {"part", "turn", "direction"}, "Push": {"top", "support", "push"}}.items():
        tree = load({"ref": name}, values=VALUES)
        assert (tree.name, tree.level) == (name, Layer.SUBTASK)
        assert keys <= set(tree.dynamic_map) | set(tree.static_args)
        assert tree.children[0].dynamic_map["subtask"] == name.lower()


def test_the_handle_pose_has_the_hand_pointing_into_the_drawer_face_and_fingers_vertical():
    grasp = child(load({"ref": "Pull"}, values=VALUES), "HandlePose").static_args["r_pose"].orientation.to_scipy().as_matrix()
    np.testing.assert_allclose(grasp[:, 2], [0, 1, 0], atol=1e-9)  # approach: +y, into the face (opens toward -y)
    np.testing.assert_allclose(grasp[:, 1], [0, 0, 1], atol=1e-9)  # finger opening: up, across the horizontal bar


class Stub(BaseNode):
    """Stands in for a leaf that needs the simulator: `run` is the function given."""
    level = Layer.STEP

    def __init__(self, like, fn):
        super().__init__(like.node_type, like.name, like.dynamic_map, like.static_args, like.output_map)
        self.fn = fn

    def run(self, **kw):
        return ExecutionResult(status=DemoStatus.PERFECT, outputs=self.fn(**kw))


def test_pull_stops_when_the_drawer_has_slid_15_cm():
    """The hand backs off 3 cm a pass along the hand's -z (the drawer's -y), the stand-in drawer follows it."""
    loop = child(load({"ref": "Pull"}, values=VALUES), "PullUntilOpen")
    start = pose([0.6, 0.0, 0.3])
    goals = []

    def step(target_pose, **kw):
        goals.append(target_pose.position.to_numpy())
        return {}

    loop.children[1] = Stub(loop.children[1], step)
    loop.children[2] = Stub(loop.children[2], lambda **kw: {"part_now": pose(start.position.to_numpy() + [0, goals[-1][1] + 0.1, 0])})
    hand = Pose.from_numpy(np.array([0.6, -0.1, 0.3]), R.from_euler("ZX", [np.pi, np.pi / 2]))
    result = loop.run(robot=None, sim_namespace="/Sim_0", scene_namespace="/Scene_0", part="p", part_start=start, handle_pose=hand)

    assert result.status == DemoStatus.PERFECT and result.outputs["part_travel"] >= 0.15
    assert len(goals) in (5, 6)  # 5 x 3 cm, a 6th if the float sum falls a hair short of 15 cm
    np.testing.assert_allclose(goals[-1] - [0.6, -0.1, 0.3], [0, -0.03 * len(goals), 0], atol=1e-9)


@pytest.mark.parametrize("direction", [1.0, -1.0])
def test_turn_sweeps_until_the_knob_is_0_6_rad_round_in_either_direction(direction):
    loop = child(load({"ref": "Turn"}, values=VALUES), "TurnUntilOn")
    start = pose([0.4, 0.2, 0.1], R.from_euler("z", 0.5))
    hands = []

    def sweep(target_pose, **kw):
        hands.append(target_pose)
        return {}

    def knob_now(**kw):  # the knob follows the hand's turn about its axis
        turned = (start.inv() * hands[-1]).position.to_numpy()
        first = (start.inv() * hands[0]).position.to_numpy()
        return {"part_now": pose(start.position.to_numpy(), start.orientation.to_scipy() * R.from_euler(
            "z", np.arctan2(turned[1], turned[0]) - np.arctan2(first[1], first[0]) + direction * 0.15))}

    loop.children[1] = Stub(loop.children[1], sweep)
    loop.children[2] = Stub(loop.children[2], knob_now)
    rim = start * pose([0.062, 0.0, 0.02], R.from_euler("x", np.pi))
    result = loop.run(robot=None, sim_namespace="/Sim_0", scene_namespace="/Scene_0", part="p", part_start=start,
                      rim_pose=rim, direction=direction)

    assert result.status == DemoStatus.PERFECT and result.outputs["part_angle"] >= 0.6
    assert len(hands) in (4, 5)  # 4 x 0.15 rad, a 5th if the float sum falls a hair short of 0.6
    sweep_dir = np.sign((start.inv() * hands[-1]).position.to_numpy()[1])
    assert sweep_dir == direction


def test_put_in_and_put_on_tilted_are_put_on_with_their_overrides():
    put_in = load({"ref": "PutIn"}, values=VALUES)
    assert (put_in.name, put_in.level) == ("PutIn", Layer.TASK)
    carry = child(child(put_in, "Place"), "CarryToSupport")
    assert child(carry, "OverSupportPose").static_args["l_pose"].position.to_numpy()[2] == 0.20
    tilted = load({"ref": "PutOnTilted"}, values=VALUES)
    assert tilted.name == "PutOnTilted"
    carry = child(child(tilted, "Place"), "CarryToSupport")
    for name in ("FlattenSupport", "MinimizeSupportTurn"):  # identity: the tilt is kept
        assert child(carry, name).node_type == "TransformPose"
        np.testing.assert_allclose(child(carry, name).static_args["r_pose"].position.to_numpy(), 0.0)
    flat = child(child(load({"ref": "PutOn"}, values=VALUES), "Place"), "CarryToSupport")
    assert child(flat, "FlattenSupport").node_type == "ProjectRotationToBaseZ"
