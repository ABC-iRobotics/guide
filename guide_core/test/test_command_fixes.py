"""Runtime commands, service replies and the recorder, run against fakes with Isaac stubbed out.

Each test pins one defect that used to slip through because the code path needs a
running simulator to reach. These modules import Isaac at module level, so they are
imported under stubs, and every module imported meanwhile is dropped again afterwards
(a MagicMock left in sys.modules would follow every later test in the session).
"""

import importlib
import math
import sys
from types import MethodType, SimpleNamespace
from unittest.mock import MagicMock

import pytest

from guide_core.types.isaac_state import IsaacState

STUBBED = (
    "carb",
    "pxr",
    "omni",
    "omni.physx",
    "isaacsim",
    "isaacsim.core",
    "isaacsim.core.prims",
    "isaacsim.core.utils",
    "isaacsim.core.utils.bounds",
    "isaacsim.core.utils.prims",
    "isaacsim.core.utils.stage",
    "isaacsim.storage",
    "isaacsim.storage.native",
    "isaacsim.util",
    "isaacsim.util.clash_detection",
)


@pytest.fixture
def isaac_import():
    """``isaac_import("guide_core.x")`` imports a module with Isaac stubbed out."""
    before = dict(sys.modules)

    def load(name):
        for stub in STUBBED:
            sys.modules[stub] = MagicMock()
        sys.modules.pop(name, None)
        return importlib.import_module(name)

    yield load
    # Drop what was imported against the stubs; real libraries (numpy, rclpy) stay.
    stubbed_roots = {name.split(".")[0] for name in STUBBED} | {"guide_core"}
    for name in set(sys.modules) - set(before):
        if name.split(".")[0] in stubbed_roots:
            del sys.modules[name]
    sys.modules.update(before)


@pytest.fixture
def command_module(isaac_import):
    """``command_module("_cmd_x")`` imports guide_core.core.commands._cmd_x under stubs."""
    return lambda name: isaac_import(f"guide_core.core.commands.{name}")


def host(**attrs):
    """A stand-in IsaacSimRuntime: commands are bound under their own names."""
    return SimpleNamespace(_logger=MagicMock(), **attrs)


def test_is_prim_clashing_falls_back_to_the_bounding_box_command(command_module):
    clash = command_module("_cmd_clash")
    h = host(state=IsaacState.RUNNING, _cd=None, _scope="")
    setattr(h, "__init_clash_detector", lambda tolerance=0.0: None)  # leaves _cd None
    h._cmd_set_scope = lambda scope: None
    h._cmd_check_bounding_box_collision = MagicMock(return_value=True)

    assert clash._cmd_is_prim_clashing(h, "/Scene_0/blocks/red_block", scope="/Scene_0/bin_0")
    h._cmd_check_bounding_box_collision.assert_called_once()


def test_set_visibilities_spreads_one_bool_over_every_prim(command_module):
    prims_cmds = command_module("_cmd_prims")
    view = SimpleNamespace(count=3, set_visibilities=MagicMock())  # count is a property
    h = host()
    setattr(h, "__get_xform", lambda prim_path: view)

    prims_cmds._cmd_set_visibilities(h, "/Scene_0/blocks/.*", True)

    view.set_visibilities.assert_called_once_with([True, True, True])


def test_a_missing_scene_usd_fails_add_scene(command_module):
    stage = command_module("_cmd_stage")
    stage.is_file = lambda path: False
    h = host(state=IsaacState.READY, update=MagicMock())
    setattr(h, "__setup_stage", MethodType(getattr(stage, "__setup_stage"), h))

    with pytest.raises(FileNotFoundError):
        stage._cmd_add_scene(h, {"usd_path_absolute": "/nowhere/block_bin.usd"}, "/Scene_0")
    assert h.state == IsaacState.ERROR


def test_service_replies_carry_their_own_type_and_the_reason(isaac_import):
    from guide_msgs.srv import Attribute, Randomize, StopRecording

    ros = isaac_import("guide_core.ros.guide_ros").GUIDEROS2Interface
    scenes = SimpleNamespace(stop_recording=lambda id, save: True, wait_stop_recording_event=print)
    backend = SimpleNamespace(
        randomize_scene=MagicMock(side_effect=RuntimeError("no scene 3")),
        call=MagicMock(return_value=True),
        _scene_manager=scenes,
    )
    me = SimpleNamespace(_backend=backend, _logger=MagicMock())

    reply = ros._randomize_callback(me, Randomize.Request(id=3), None)
    assert (reply.success, reply.message) == (False, "no scene 3")

    reply = ros._attribute_request_callback(me, Attribute.Request(path="/a", attribute="b"), None)
    assert isinstance(reply, Attribute.Response) and reply.result == "True"

    reply = ros._stop_recording_callback(me, StopRecording.Request(id=0), None)
    assert reply.success and "reset" not in reply.message


def test_the_end_effector_is_recorded_relative_to_the_robot_base(isaac_import):
    """A base on Scene_1 (y = 10, z = 1 in the world), turned 90 deg about z: the data is the
    end effector as that base sees it, translation and rotation."""
    import numpy as np

    orchestrator = isaac_import("guide_core.scene.scene_orchestrator").SceneOrchestrator
    turn = np.array([[np.cos(np.pi / 4), 0.0, 0.0, np.sin(np.pi / 4)]])  # wxyz, +90 deg about z
    base = SimpleNamespace(get_world_poses=lambda: (np.array([[-0.3, 10.0, 1.0]]), turn))
    # 0.4 ahead of and 0.4 above the base: ahead is world +y once the base is turned.
    ee = SimpleNamespace(get_world_poses=lambda: (np.array([[-0.3, 10.4, 1.4]]), turn))
    scene = host(
        _config={"dataset": {}}, ee_views={"franka": ee}, base_views={"franka": base},
        prompts={"task": "", "subtask": ""}, task="",
    )
    scene._recorded_annotators = dict

    obs = orchestrator.record_step(scene, 0)["observation"]

    assert np.allclose([obs[k] for k in ("x", "y", "z")], [0.4, 0.0, 0.4])
    assert np.allclose([obs[k] for k in ("wx", "wy", "wz")], 0.0, atol=1e-9)


def test_robot_graphs_carry_the_motion_layer(isaac_import):
    orchestrator = isaac_import("guide_core.scene.scene_orchestrator").SceneOrchestrator
    config = {"robots": {"franka": {"path": "/fr3", "default_joint_states": [0.0] * 9,
                                    "controller": "oscbf", "oscbf": {"mode": "velocity"}},
                         "other": {"path": "/ur", "default_joint_states": [0.0] * 6}}}

    franka, other = orchestrator.create_robot_graphs(SimpleNamespace(_config=config))

    assert (franka["controller"], franka["oscbf"]) == ("oscbf", {"mode": "velocity"})
    assert (other["controller"], other["oscbf"]) == ("moveit", {})


def test_an_oscbf_robot_gets_a_host_and_no_joint_command_subscriber(command_module, monkeypatch):
    module = command_module("_cmd_simulator")
    made = []

    class FakeHost:
        def __init__(self, *args, **kwargs):
            made.append((args, kwargs))

    monkeypatch.setitem(sys.modules, "guide_core.control.oscbf_host", SimpleNamespace(OscbfHost=FakeHost))
    robots = [{"namespace": "/franka", "articulation_root": "/fr3", "path": "/franka",
               "default_joint_states": list(range(9)), "controller": "oscbf",
               "oscbf": {"robot": "fr3", "mode": "velocity", "rate": 60}}]
    manager = SimpleNamespace(
        get_scene_usd_path=lambda i: None, get_scene_robot_graphs=lambda i: robots,
        get_scene_camera_graphs=lambda i: None, _scenes=[SimpleNamespace(replicator_files=None)],
    )
    controls = []
    runtime = host(
        _simulator=SimpleNamespace(_add_scene_to_manager=lambda p: (0, (0, 0, 0), {}), _scene_manager=manager, _sim_path="/Sim_0"),
        _cmd_add_scene=lambda **kwargs: None,
        _cmd_set_world_poses=lambda **kwargs: None,
        _cmd_create_robot_control=lambda **kwargs: controls.append(kwargs),
        _cmd_create_tf_graph=lambda **kwargs: None,
    )

    module._cmd_register_scene(runtime, "block_bin")

    assert controls[0]["subscribe"] is False
    (args, kwargs), = made
    assert args == ("/Scene_0/fr3",) and (kwargs["robot"], kwargs["mode"], kwargs["rate"]) == ("fr3", "velocity", 60)
    assert kwargs["q_rest"] == list(range(7))
    assert runtime._motion_hosts[0][0].topic == "/Sim_0/Scene_0/franka/motion_target"


def test_motion_targets_are_subscribed_depth_one_and_converted(isaac_import):
    from guide_motion.targets import TargetHolder
    from guide_msgs.msg import MotionTarget

    ros = isaac_import("guide_core.ros.guide_ros")
    subscriptions = []
    me = SimpleNamespace(_logger=MagicMock(), _reentrant_group=None,
                         create_subscription=lambda *args, **kwargs: subscriptions.append(args))
    target_host = SimpleNamespace(topic="/Sim_0/Scene_0/franka/motion_target", holder=TargetHolder(),
                                  model=SimpleNamespace(base_link="fr3_link0"))

    ros.GUIDEROS2Interface._subscribe_motion_target(me, target_host)

    (msg_type, topic, put, qos), = subscriptions
    assert msg_type is MotionTarget and topic == target_host.topic and qos.depth == 1
    msg = MotionTarget(mode=MotionTarget.MODE_POSE, frame_id="fr3_link0", gripper_width=0.08)
    msg.pose.position.x, msg.pose.orientation.z, msg.pose.orientation.w = 0.4, 1.0, 0.0  # 180 deg about z
    put(msg)
    target, stale = target_host.holder.get(0.0)
    assert not stale and target.position[0] == 0.4 and target.gripper_width == 0.08
    assert abs(target.rotation[0, 0] + 1.0) < 1e-12
    put(MotionTarget(frame_id="Scene_0"))  # not the base frame: dropped
    broken = MotionTarget(frame_id="fr3_link0")
    broken.pose.position.x, broken.pose.orientation.w = float("nan"), 1.0
    put(broken)  # a NaN from a buggy publisher: dropped
    assert target_host.holder.get(0.0)[0] is target


def test_a_motion_target_with_a_zero_quaternion_is_dropped_with_a_warning(isaac_import):
    from guide_motion.targets import TargetHolder
    from guide_msgs.msg import MotionTarget

    ros = isaac_import("guide_core.ros.guide_ros")
    subscriptions = []
    me = SimpleNamespace(_logger=MagicMock(), _reentrant_group=None,
                         create_subscription=lambda *args, **kwargs: subscriptions.append(args))
    target_host = SimpleNamespace(topic="/Sim_0/Scene_0/franka/motion_target", holder=TargetHolder(),
                                  model=SimpleNamespace(base_link="fr3_link0"))
    ros.GUIDEROS2Interface._subscribe_motion_target(me, target_host)
    (_, _, put, _), = subscriptions

    msg = MotionTarget(mode=MotionTarget.MODE_POSE, frame_id="fr3_link0")
    msg.pose.orientation.w = 0.0  # (0, 0, 0, 0); ROS 2 defaults w to 1. scipy refuses a zero quaternion
    put(msg)

    assert target_host.holder.get(0.0) == (None, True)
    me._logger.warning.assert_called_once()


ARM_JOINTS = [f"fr3_joint{i}" for i in range(1, 8)]


def test_a_nan_gripper_width_passes_through_to_leave_the_gripper(isaac_import):
    """FIX B1: NaN means "leave the gripper" (Target's convention). rosidl cannot default the
    field to NaN, so publishers set it; the ROS boundary must hand it to the holder unchanged."""
    from guide_motion.targets import JOINTS, TargetHolder
    from guide_msgs.msg import MotionTarget

    ros = isaac_import("guide_core.ros.guide_ros")
    subscriptions = []
    me = SimpleNamespace(_logger=MagicMock(), _reentrant_group=None,
                         create_subscription=lambda *args, **kwargs: subscriptions.append(args))
    target_host = SimpleNamespace(topic="/Sim_0/Scene_0/franka/motion_target", holder=TargetHolder(),
                                  model=SimpleNamespace(base_link="fr3_link0", arm_joints=ARM_JOINTS))
    ros.GUIDEROS2Interface._subscribe_motion_target(me, target_host)
    (_, _, put, _), = subscriptions

    put(MotionTarget(mode=MotionTarget.MODE_JOINTS, frame_id="fr3_link0", joints=[0.1] * 7, gripper_width=float("nan")))

    target, stale = target_host.holder.get(0.0)
    assert not stale and target.mode == JOINTS and math.isnan(target.gripper_width)


def test_reset_homes_the_oscbf_robots_of_that_scene(command_module):
    module = command_module("_cmd_simulator")
    homed = []
    scene = SimpleNamespace(reset_fire=lambda: None)
    manager = SimpleNamespace(reset_preprocess=lambda i: [], reset_postprocess=lambda i, results: True,
                              _scenes=[scene, scene])
    runtime = host(
        _simulator=SimpleNamespace(_scene_manager=manager),
        _execute_instructions_directly=lambda instructions: [],
        _motion_hosts={0: [SimpleNamespace(home=lambda: homed.append(0))], 1: [SimpleNamespace(home=lambda: homed.append(1))]},
    )

    assert module._cmd_reset_scene(runtime, 1) is True
    assert homed == [1]


def motion_subscription(isaac_import):
    """GUIDE's motion_target callback for an FR3 host, on a fake node: (node, host, put)."""
    from guide_motion.targets import TargetHolder

    ros = isaac_import("guide_core.ros.guide_ros")
    subscriptions = []
    me = SimpleNamespace(_logger=MagicMock(), _reentrant_group=None,
                         create_subscription=lambda *args, **kwargs: subscriptions.append(args))
    target_host = SimpleNamespace(topic="/Sim_0/Scene_0/franka/motion_target", holder=TargetHolder(),
                                  model=SimpleNamespace(base_link="fr3_link0", arm_joints=ARM_JOINTS))
    ros.GUIDEROS2Interface._subscribe_motion_target(me, target_host)
    (_, _, put, _), = subscriptions
    return me, target_host, put


def test_a_joints_target_of_the_wrong_length_is_dropped_with_a_warning(isaac_import):
    """REVIEWER C B3: an empty (the message default) or wrong-length joints array would raise in
    Controller.step on every physics substep; it is dropped at the ROS boundary instead."""
    from guide_msgs.msg import MotionTarget

    me, target_host, put = motion_subscription(isaac_import)
    for n in (0, 3, 12):
        put(MotionTarget(mode=MotionTarget.MODE_JOINTS, frame_id="fr3_link0", joints=[0.1] * n))
    assert target_host.holder.get(0.0) == (None, True) and me._logger.warning.call_count == 3
    put(MotionTarget(mode=MotionTarget.MODE_JOINTS, frame_id="fr3_link0", joints=[0.1] * 7))
    assert len(target_host.holder.get(0.0)[0].joints) == 7


def test_a_motion_target_of_an_unknown_mode_is_dropped_with_a_warning(isaac_import):
    """REVIEWER C B3: Controller.step reads any mode it does not know as POSE, with the default
    position (the base origin) as the goal; such a message is dropped at the ROS boundary."""
    from guide_msgs.msg import MotionTarget

    me, target_host, put = motion_subscription(isaac_import)
    put(MotionTarget(mode=9, frame_id="fr3_link0"))
    assert target_host.holder.get(0.0) == (None, True)
    me._logger.warning.assert_called_once()


def test_reset_runs_reset_yaml_before_homing(command_module):
    """Ruling 2 pin (controller, B3b; FINAL-B I1/M2): reset.yaml places the objects and drives the fingers
    first, in either dialect (instructions, or the Replicator file reset_fire runs); the oscbf hosts are
    homed after both."""
    module = command_module("_cmd_simulator")
    calls = []
    manager = SimpleNamespace(reset_preprocess=lambda i: ["reset.yaml"], reset_postprocess=lambda i, results: True,
                              _scenes=[SimpleNamespace(reset_fire=lambda: calls.append("reset_fire"))])
    runtime = host(
        _simulator=SimpleNamespace(_scene_manager=manager),
        _execute_instructions_directly=lambda instructions: calls.extend(instructions) or [],
        _motion_hosts={0: [SimpleNamespace(home=lambda: calls.append("home"))]},
    )

    module._cmd_reset_scene(runtime, 0)

    assert calls == ["reset.yaml", "reset_fire", "home"]


def test_register_scene_reports_refused_robots_in_its_message(isaac_import):
    """FINAL-B M7: a robot refused at registration (missing overrides, unsafe q_rest) answered success=True
    with an empty message; the solver then timed out. The message names it; success keeps its meaning."""
    from guide_msgs.srv import RegisterScene

    ros = isaac_import("guide_core.ros.guide_ros")
    refused = SimpleNamespace(topic="/Sim_0/Scene_0/franka/motion_target", diagnostics={"refused": "q_rest breaks a barrier"})
    fine = SimpleNamespace(topic="/Sim_0/Scene_0/ur/motion_target", diagnostics={"refused": None})
    backend = SimpleNamespace(stop=lambda: None, play=lambda: None, call=lambda name: None,
                              register_scene=lambda path: (0, (0.0, 0.0, 0.0)), motion_hosts=lambda i: [refused, fine])
    me = SimpleNamespace(_backend=backend, _logger=MagicMock(), _has_clock=True, _subscribe_motion_target=lambda host: None)

    response = ros.GUIDEROS2Interface._register_callback(me, RegisterScene.Request(path="block_bin"), RegisterScene.Response())

    assert response.success and "/Sim_0/Scene_0/franka" in response.message and "q_rest breaks a barrier" in response.message
    assert "/ur" not in response.message
