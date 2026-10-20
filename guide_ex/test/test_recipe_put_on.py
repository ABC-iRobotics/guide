"""The library's PutOn (S3 v2 Plan C): announce, locate, pick, place on a support; no recording."""
import numpy as np

from guide_ex.core.states import Layer
from guide_ex.recipe import load
from guide_ex.recipe.registry import LIBRARY

PUT_ON = next(d for d in LIBRARY if d.name == "tasks") / "PutOn.yaml"
VALUES = {"robot": {"gripper_joint": "fr3_finger_joint1", "home": {f"fr3_joint{i}.pos": 0.0 for i in range(1, 8)}},
          "episode": {"camera_hold": False, "hold_seconds": 0.0}}
EPISODE = """
node_type: CompositeNode
name: Episode
level: PROCEDURE
dynamic_map: {robot: robot, sim_namespace: sim_namespace, scene_namespace: scene_namespace, scene_id: scene_id,
              robot_prim: robot_prim, target: target, top: top, support: support, pick: pick, place: place,
              rest_pose: rest_pose, scene_pose_inv: scene_pose_inv, dataset_path: dataset_path}
children:
  - ref: StartEpisode
  - ref: PutOn
  - ref: CheckSuccess
  - ref: StopEpisode
overrides:
  PutOn.GetTargetPose.TransformCubePoseForGripper:
    ref: TransformPose
    alias: TransformCubePoseForGripper
    dynamic_map: {l_pose: cube_pose}
    static_args: {r_pose: {position: [-0.05, 0.0, 0.03], orientation: {euler: [xz, 3.141592653589793, 1.5707963267948966]}}}
    output_map: {pose: cube_pose}
  PutOn.Place.CarryToSupport.SupportGripperDown:
    ref: TransformPose
    alias: SupportGripperDown
    dynamic_map: {l_pose: support_pose}
    static_args: {r_pose: {position: [0.0, 0.0, 0.02]}}
    output_map: {pose: support_pose}
  PutOn.Pick.CloseGripper:
    ref: SetGripperState
    alias: CloseGripper
    dynamic_map: {robot: robot}
    static_args: {gripper_goal_pos: {"${robot.gripper_joint}": 0.0}}
"""


def test_put_on_announces_locates_picks_and_places_on():
    tree = load(PUT_ON, values=VALUES)
    assert (tree.name, tree.level) == ("PutOn", Layer.TASK)
    assert [c.name for c in tree.children] == ["AnnouncePick", "GetTargetPose", "Pick", "Place"]
    assert tree.children[0].dynamic_map["subtask"] == "pick"
    assert {"target", "top", "support", "pick", "place", "scene_pose_inv", "rest_pose"} <= set(tree.dynamic_map)


def test_a_task_overrides_the_grasp_through_put_on(tmp_path):  # the shape guide_bench writes
    (tmp_path / "Episode.yaml").write_text(EPISODE)
    tree = load(tmp_path / "Episode.yaml", values=VALUES, search_paths=[tmp_path])
    put_on = next(c for c in tree.children if c.name == "PutOn")
    grasp = next(c for c in put_on.children[1].children if c.name == "TransformCubePoseForGripper")
    np.testing.assert_allclose(grasp.static_args["r_pose"].position.to_numpy(), [-0.05, 0.0, 0.03])
    close = next(c for c in put_on.children[2].children if c.name == "CloseGripper")
    assert close.static_args["gripper_goal_pos"] == {"fr3_finger_joint1": 0.0}
    carry = next(c for c in put_on.children[3].children if c.name == "CarryToSupport")
    down = next(c for c in carry.children if c.name == "SupportGripperDown")
    np.testing.assert_allclose(down.static_args["r_pose"].position.to_numpy(), [0.0, 0.0, 0.02])
