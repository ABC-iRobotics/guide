"""load(): a recipe's nodes, typed values, templates, overrides, `when`, and every load-time check."""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation as R

from guide_core.types.geometry import Pose
from guide_ex.core.composite_node import CompositeNode, RecoveryNode
from guide_ex.core.states import Layer
from guide_ex.recipe import RecipeError, default_registry, load
from guide_ex.recipe.values import convert
from guide_ex.utility.pose import TransformPose
from guide_ex.utility.wait import WaitForSeconds


def wait(alias, seconds=1.0, **extra):
    return {"ref": "WaitForSeconds", "alias": alias, "static_args": {"seconds": seconds}, **extra}


def seq(*children, name="Seq", level="SEQUENCE", **extra):
    return {"node_type": "CompositeNode", "name": name, "level": level, "children": list(children), **extra}


def test_a_leaf_gets_its_alias_and_maps():
    leaf = load({"ref": "TransformPose", "alias": "Lift", "dynamic_map": {"r_pose": "cube_pose"},
                 "static_args": {"l_pose": {"position": [0, 0, 0.1]}}, "output_map": {"pose": "lifted"}})

    assert isinstance(leaf, TransformPose) and leaf.name == "Lift" and leaf.level == Layer.UTILITY
    assert leaf.dynamic_map == {"r_pose": "cube_pose"} and leaf.output_map == {"pose": "lifted"}
    assert np.allclose(leaf.static_args["l_pose"].position.to_numpy(), [0, 0, 0.1])


def test_a_composite_gets_children_mode_and_branches():
    tree = load(seq(wait("A"), wait("B"), mode="condition", condition_expr="ok",
                    false_branch={"ref": "NodeException", "alias": "NotOk"}))

    assert isinstance(tree, CompositeNode) and tree.level == Layer.SEQUENCE
    assert [c.name for c in tree.children] == ["A", "B"]
    assert tree.mode.value == "condition" and tree.false_branch.name == "NotOk"


@pytest.mark.parametrize("orientation, expected", [
    ({"euler": ["z", 1.57]}, R.from_euler("z", 1.57)),
    ({"euler": ["xyz", 3.14, 0.0, 0.5]}, R.from_euler("xyz", [3.14, 0.0, 0.5])),
    ({"quat": [0.0, 0.0, 0.7071068, 0.7071068]}, R.from_quat([0.0, 0.0, 0.7071068, 0.7071068])),
    ({"rotvec": [0.0, 0.0, 1.0]}, R.from_rotvec([0.0, 0.0, 1.0])),
])
def test_an_orientation_becomes_a_pose(orientation, expected):
    pose = convert({"orientation": orientation})

    assert isinstance(pose, Pose)
    assert np.allclose(pose.orientation.to_scipy().as_matrix(), expected.as_matrix())
    assert np.allclose(pose.position.to_numpy(), 0.0)


def test_poses_convert_at_any_depth_and_nothing_else_does():
    value = convert({"poses": [{"position": [1, 2, 3]}], "gripper": {"fr3_finger_joint1": 0.04}})

    assert isinstance(value["poses"][0], Pose)
    assert value["gripper"] == {"fr3_finger_joint1": 0.04}


def test_static_args_named_like_pose_keys_stay_arguments():
    """The static_args mapping itself is never a pose: its keys are argument names."""
    leaf = load({"ref": "WaitForSeconds", "static_args": {"seconds": 0.5}})

    assert leaf.static_args == {"seconds": 0.5}


def test_a_bad_euler_is_refused_with_the_node():
    with pytest.raises(RecipeError, match=r"<dict>: <top>: euler is \[axes, one angle per axis\]"):
        load({"ref": "TransformPose", "dynamic_map": {"r_pose": "a"},
              "static_args": {"l_pose": {"orientation": {"euler": ["xy", 1.0]}}}})


def test_values_interpolate_in_values_and_in_mapping_keys():
    values = {"robot": {"gripper_joint": "panda_finger_joint1"}, "hold": 2.5}
    leaf = load({"ref": "SetGripperState", "dynamic_map": {"robot": "robot"},
                 "static_args": {"gripper_goal_pos": {"${robot.gripper_joint}": 0.04}}}, values=values)
    hold = load(wait("Hold", "${hold}"), values=values)

    assert leaf.static_args == {"gripper_goal_pos": {"panda_finger_joint1": 0.04}}
    assert hold.static_args == {"seconds": 2.5}


def test_a_missing_value_names_the_file(tmp_path):
    path = tmp_path / "Hold.yaml"
    path.write_text("ref: WaitForSeconds\nstatic_args: {seconds: '${episode.hold}'}\n")

    with pytest.raises(RecipeError, match=r"Hold\.yaml: .*episode"):
        load(path)


def test_when_false_omits_the_node():
    tree = load(seq(wait("A"), wait("B", when="${on}"), wait("C", when=True)), values={"on": False})

    assert [c.name for c in tree.children] == ["A", "C"]


def test_when_must_be_a_bool():
    with pytest.raises(RecipeError, match="`when` must be true or false"):
        load(seq(wait("A", when="yes please")))


def test_a_template_is_merged_with_the_keys_beside_its_ref(tmp_path):
    (tmp_path / "Settle.yaml").write_text("ref: WaitForSeconds\nalias: Settle\nstatic_args: {seconds: 1.0}\n")
    (tmp_path / "Twice.yaml").write_text(
        "node_type: CompositeNode\nname: Twice\nlevel: SEQUENCE\n"
        "children: [{ref: Settle}, {ref: Settle, alias: SettleLonger, static_args: {seconds: 2.0}}]\n")

    tree = load({"ref": "Twice", "alias": "TwiceAgain"}, search_paths=[tmp_path])

    assert tree.name == "TwiceAgain" and tree.node_type == "Twice"
    assert [(c.name, c.static_args["seconds"]) for c in tree.children] == [("Settle", 1.0), ("SettleLonger", 2.0)]


def test_an_override_replaces_a_nested_node_by_its_dot_path():
    inner = seq(wait("Detect"), wait("Move"), name="Locate")
    tree = load(seq(inner, name="Pick", level="SUBTASK",
                    overrides={"Locate.Detect": {"ref": "TransformPose", "alias": "Detect",
                                                 "dynamic_map": {"l_pose": "a", "r_pose": "b"}}}))

    assert isinstance(tree.children[0].children[0], TransformPose)
    assert isinstance(tree.children[0].children[1], WaitForSeconds)


def test_an_override_that_names_no_node_is_refused():
    with pytest.raises(RecipeError, match="override 'Locate.Typo' names no node below Pick"):
        load(seq(seq(wait("Detect"), name="Locate"), name="Pick", level="SUBTASK",
                 overrides={"Locate.Typo": wait("Typo")}))


def test_an_override_that_breaks_the_layers_is_refused():
    with pytest.raises(RecipeError, match=r"Outer\.Inner: child A \(SEQUENCE\) is not below SEQUENCE"):
        load(seq(seq(wait("A"), name="Inner"), name="Outer", level="SUBTASK",
                 overrides={"Inner.A": seq(wait("B"), name="A")}))


def test_a_recovery_node_takes_resume_target_and_max_retries():
    recovery = {"node_type": "RecoveryNode", "name": "Again", "level": "SEQUENCE", "resume_target": "A",
                "max_retries": 2, "children": [wait("Breathe")]}
    tree = load(seq(wait("A"), name="Task", level="SUBTASK", fallbacks={"A": recovery}))

    again = tree.fallbacks["A"]
    assert isinstance(again, RecoveryNode) and again.resume_target == "A" and again.max_retries == 2


@pytest.mark.parametrize("spec, message", [
    ({**wait("A"), "colour": "red"}, r"unknown key\(s\) \['colour'\]"),
    ({"ref": "NoSuchNode"}, "unknown ref 'NoSuchNode'"),
    ({"ref": "WaitForSeconds", "node_type": "CompositeNode"}, "exactly one of `ref` and `node_type`"),
    ({"alias": "Nameless"}, "exactly one of `ref` and `node_type`"),
    ({**wait("A"), "children": []}, r"WaitForSeconds is a leaf; .* not \['children'\]"),
    ({**wait("A"), "level": "STEP"}, "WaitForSeconds is UTILITY, not STEP"),
    ({"ref": "WaitForSeconds", "dynamic_map": {"seconds": "s", "secs": "s"}},
     r"WaitForSeconds\.run\(\) takes no \['secs'\]"),
    ({"ref": "TransformPose", "dynamic_map": {"l_pose": "a"}}, r"TransformPose\.run\(\) needs \['r_pose'\]"),
    (seq(wait("A"), level="LAYER9"), "`level` is one of"),
    (seq(wait("A"), name=None), "a CompositeNode needs a `name`"),
    (seq(wait("A"), node_type="Sequence"), "node_type is CompositeNode or RecoveryNode"),
    (seq(wait("A"), resume_target="A"), r"\['resume_target'\] belong to a RecoveryNode"),
    ({"node_type": "RecoveryNode", "name": "R", "level": "STEP", "children": []}, "needs a `resume_target`"),
    (seq(wait("A"), condition_expr="done ==", mode="condition"), "condition_expr 'done =='"),
    (seq(seq(wait("A"), name="Inner"), name="Outer"), "'Outer' \\(SEQUENCE\\) cannot contain child 'Inner'"),
    (seq(wait("A"), fallbacks={"B": {"node_type": "RecoveryNode", "name": "R", "level": "STEP",
                                     "resume_target": "A", "children": []}}),
     r"fallback for 'B', which is no child \(children: \['A'\]\)"),
    (seq(wait("A"), fallbacks={"A": {"node_type": "RecoveryNode", "name": "R", "level": "STEP",
                                     "resume_target": "Z", "children": []}}),
     "R resumes at 'Z', which is no child"),
    (seq(wait("A"), fallbacks={"A": wait("NotARecovery")}), "fallback 'A' must be a RecoveryNode"),
])
def test_a_bad_recipe_is_refused_naming_the_node(spec, message):
    with pytest.raises(RecipeError, match=message):
        load(spec)


def test_an_error_names_the_file_and_the_path(tmp_path):
    path = tmp_path / "Task.yaml"
    path.write_text("node_type: CompositeNode\nname: Task\nlevel: SUBTASK\n"
                    "children: [{node_type: CompositeNode, name: Inner, level: SEQUENCE, children: [{ref: Nope}]}]\n")

    with pytest.raises(RecipeError, match=r"Task\.yaml: Task\.children\[0\]\.Inner\.children\[0\]: unknown ref 'Nope'"):
        load(path)


def test_a_template_that_refs_itself_is_refused(tmp_path):
    (tmp_path / "Loop.yaml").write_text("node_type: CompositeNode\nname: Loop\nlevel: SEQUENCE\n"
                                        "children: [{ref: Loop}]\n")

    with pytest.raises(RecipeError, match="'Loop' refs itself"):
        load({"ref": "Loop"}, search_paths=[tmp_path])


def test_a_top_node_left_out_by_when_is_refused():
    with pytest.raises(RecipeError, match="the top node's `when` is false"):
        load(wait("A", when=False))


def test_euler_axes_must_be_a_string():
    """YAML 1.1 reads a bare on/off/yes/no as a bool: such axes are refused, not passed to scipy."""
    with pytest.raises(RecipeError, match=r"euler is \[axes, one angle per axis\]"):
        load({"ref": "TransformPose", "dynamic_map": {"r_pose": "a"},
              "static_args": {"l_pose": {"orientation": {"euler": [True, 1.0]}}}})


def test_two_tasks_recipes_never_meet(tmp_path):
    """Each task has an Episode.yaml and may name its templates alike: a load sees only its own."""
    for task, seconds in (("a", 1.0), ("b", 2.0)):
        (tmp_path / task).mkdir()
        (tmp_path / task / "Hold.yaml").write_text(f"ref: WaitForSeconds\nstatic_args: {{seconds: {seconds}}}\n")
        (tmp_path / task / "Episode.yaml").write_text("node_type: CompositeNode\nname: E\nlevel: SEQUENCE\n"
                                                      "children: [{ref: Hold}]\n")

    a = load(tmp_path / "a" / "Episode.yaml", search_paths=[tmp_path / "a"])
    b = load(tmp_path / "b" / "Episode.yaml", search_paths=[tmp_path / "b"])

    assert (a.children[0].static_args, b.children[0].static_args) == ({"seconds": 1.0}, {"seconds": 2.0})
    assert not {"Hold", "Episode"} & set(default_registry.names())
