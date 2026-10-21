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
    assert np.allclose([obs[k] for k in ("rx", "ry", "rz")], 0.0, atol=1e-9)


def test_record_step_names_pose_euler_joint_vel_for_and_extras(isaac_import):
    """the state's rotation vector rx ry rz, .vel/.eff measured in both, twist/wrench copied into state and action."""
    import numpy as np

    orchestrator = isaac_import("guide_core.scene.scene_orchestrator").SceneOrchestrator
    yaw = np.array([[np.cos(np.pi / 8), 0.0, 0.0, np.sin(np.pi / 8)]])  # wxyz, +45 deg about z
    ident = np.array([[1.0, 0.0, 0.0, 0.0]])
    ee = SimpleNamespace(get_world_poses=lambda: (np.array([[0.1, 0.2, 0.3]]), yaw))
    base = SimpleNamespace(get_world_poses=lambda: (np.zeros((1, 3)), ident))
    view = SimpleNamespace(get_joint_positions=lambda joint_indices: np.array([0.5]), get_joint_velocities=lambda joint_indices: np.array([1.5]),
                           get_measured_joint_efforts=lambda joint_indices: np.array([-2.5]), get_applied_action=lambda: None)
    obs_masks = {"franka": {"pos": {"indices": [0], "keys": ["j.pos"]}, "vel": {"indices": [0], "keys": ["j.vel"]}, "eff": {"indices": [0], "keys": ["j.eff"]}}}
    scene = host(
        _config={"dataset": {}}, ee_views={"franka": ee}, base_views={"franka": base}, robots_views={"franka": view},
        obs_masks=obs_masks, act_masks={"franka": {"vel": obs_masks["franka"]["vel"], "eff": obs_masks["franka"]["eff"]}},
        prompts={"task": "", "subtask": ""}, task="",
        _extras={"obs": ["vx", "fz"], "act": ["vx"]}, _record_extras=lambda: {"vx": 3.0, "fz": 7.0},
    )
    scene._recorded_annotators = dict

    frame = orchestrator.record_step(scene, 0)
    obs, act = frame["observation"], frame["action"]

    assert list(obs) == ["x", "y", "z", "rx", "ry", "rz", "j.pos", "j.vel", "j.eff", "vx", "fz"]
    assert list(act) == ["x", "y", "z", "rx", "ry", "rz", "j.vel", "j.eff", "vx"]
    assert np.allclose([obs["rx"], obs["ry"], obs["rz"]], [0.0, 0.0, np.pi / 4])  # rotation vector: 45 deg about z
    assert (obs["j.vel"], obs["j.eff"], obs["vx"], obs["fz"]) == (1.5, -2.5, 3.0, 7.0)
    assert (act["j.vel"], act["j.eff"], act["vx"]) == (1.5, -2.5, 3.0)


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

    config = {"obstacles": [{"path": "/bin_0"}]}
    parsed = []
    monkeypatch.setitem(sys.modules, "guide_core.control.oscbf_host", SimpleNamespace(
        OscbfHost=FakeHost, obstacle_config=lambda scene, oscbf: parsed.append((scene, oscbf)) or ([{"path": "/bin_0", "shape": "obb"}], 10.0)))
    robots = [{"namespace": "/franka", "articulation_root": "/fr3", "path": "/franka",
               "default_joint_states": list(range(9)), "controller": "oscbf",
               "oscbf": {"robot": "fr3", "mode": "velocity", "rate": 60}}]
    manager = SimpleNamespace(
        get_scene_usd_path=lambda i: None, get_scene_robot_graphs=lambda i: robots,
        get_scene_camera_graphs=lambda i: None, _scenes=[SimpleNamespace(replicator_files=None, _config=config, build_stage=lambda: None)],
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
    # init.yaml's obstacles: the scene's config and the robot's oscbf section, paths under this scene's root
    assert parsed == [(config, robots[0]["oscbf"])]
    assert kwargs["obstacles"] == [{"path": "/Scene_0/bin_0", "shape": "obb"}] and kwargs["obstacle_update_hz"] == 10.0


def test_motion_targets_are_subscribed_depth_one_and_converted(isaac_import):
    from guide_motion.targets import TargetHolder
    from guide_msgs.msg import MotionTarget

    ros = isaac_import("guide_core.ros.guide_ros")
    subscriptions = []
    me = SimpleNamespace(_logger=MagicMock(), _reentrant_group=None,
                         create_subscription=lambda *args, **kwargs: subscriptions.append(args))
    target_host = SimpleNamespace(topic="/Sim_0/Scene_0/franka/motion_target", holder=TargetHolder(), reset_time=None,
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
    target_host = SimpleNamespace(topic="/Sim_0/Scene_0/franka/motion_target", holder=TargetHolder(), reset_time=None,
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
    target_host = SimpleNamespace(topic="/Sim_0/Scene_0/franka/motion_target", holder=TargetHolder(), reset_time=None,
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
        _motion_hosts={0: [SimpleNamespace(home=lambda: homed.append(0), refresh_obstacles=lambda: None)],
                       1: [SimpleNamespace(home=lambda: homed.append(1), refresh_obstacles=lambda: None)]},
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
    target_host = SimpleNamespace(topic="/Sim_0/Scene_0/franka/motion_target", holder=TargetHolder(), reset_time=None,
                                  model=SimpleNamespace(base_link="fr3_link0", arm_joints=ARM_JOINTS))
    ros.GUIDEROS2Interface._subscribe_motion_target(me, target_host)
    (_, _, put, _), = subscriptions
    return me, target_host, put


def test_a_motion_target_stamped_at_or_before_the_last_reset_is_dropped(isaac_import):
    """FIX C2-2: a target already in flight when GUIDE's Reset homed the arm must not undo the
    homing. The Reset at 12.5 s of sim time, the /clock the solver stamps its targets from."""
    from builtin_interfaces.msg import Time
    from guide_msgs.msg import MotionTarget

    me, target_host, put = motion_subscription(isaac_import)
    target_host.reset_time = 12.5

    def stamped(sec, nanosec):
        return MotionTarget(mode=MotionTarget.MODE_JOINTS, frame_id="fr3_link0", joints=[0.1] * 7,
                            gripper_width=float("nan"), stamp=Time(sec=sec, nanosec=nanosec))

    put(stamped(12, 400_000_000))  # before the Reset
    put(stamped(12, 500_000_000))  # at it
    assert target_host.holder.get(0.0) == (None, True)
    me._logger.warning.assert_called_with(
        f"{target_host.topic}: target stamped before the last Reset: dropped", throttle_duration_sec=5.0)
    put(stamped(12, 500_000_001))  # after it
    assert target_host.holder.get(0.0)[0] is not None


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
        _motion_hosts={0: [SimpleNamespace(home=lambda: calls.append("home"), refresh_obstacles=lambda: calls.append("obstacles"))]},
    )

    module._cmd_reset_scene(runtime, 0)

    assert calls == ["reset.yaml", "reset_fire", "home", "obstacles"]  # the barriers follow the objects reset.yaml moved


def test_randomize_refreshes_the_obstacles_after_moving_the_objects(command_module):
    """The bins move on Randomize (block_bin's randomize.yaml): the OSCBF barriers read their new poses before
    the solver's next target. Only that scene's hosts."""
    module = command_module("_cmd_simulator")
    calls = []
    manager = SimpleNamespace(randomize_preprocess=lambda i, **kwargs: ["randomize.yaml"],
                              randomize_postprocess=lambda i, results: calls.append("postprocess") or "{}")
    runtime = host(
        _simulator=SimpleNamespace(_scene_manager=manager),
        _execute_instructions_directly=lambda instructions: calls.extend(instructions) or [],
        _motion_hosts={0: [SimpleNamespace(refresh_obstacles=lambda: calls.append("obstacles 0"))],
                       1: [SimpleNamespace(refresh_obstacles=lambda: calls.append("obstacles 1"))]},
    )

    module._cmd_randomize_scene(runtime, 1)

    assert calls == ["randomize.yaml", "obstacles 1", "postprocess"]


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


def test_a_pose_request_answers_in_the_requesting_robots_base_frame(isaac_import):
    """Ruling 4: GUIDE converts. A block on Scene_1 (world y + 10, z + 1), its robot's base at
    (-0.3, 10, 1.013) turned 90 deg about z: the answer is the block as that base sees it."""
    import numpy as np
    from guide_msgs.srv import Pose as PoseSrv
    from scipy.spatial.transform import Rotation as R

    from guide_core.types.geometry import Point, Pose, Rotation

    ros = isaac_import("guide_core.ros.guide_ros").GUIDEROS2Interface
    world = {
        "/Scene_1": Pose(Point([0.0, 10.0, 1.0])),
        "/Scene_1/fr3/fr3_link0": Pose(Point([-0.3, 10.0, 1.013]), Rotation(R.from_euler("z", 90, degrees=True))),
        # 0.4 ahead of the base (world +y once it is turned) and 0.4 above it
        "/Scene_1/blocks/red_block": Pose(Point([-0.3, 10.4, 1.413])),
    }
    config = {"robots": {"franka": {"path": "/fr3", "base_name": "fr3_link0"}}}
    backend = SimpleNamespace(
        call=lambda name, prim_path: world[prim_path],
        _scene_manager=SimpleNamespace(_scenes=[None, SimpleNamespace(_config=config)]),
    )
    me = SimpleNamespace(_backend=backend)

    def ask(frame_id):
        request = PoseSrv.Request(path="/Scene_1/blocks/red_block", frame_id=frame_id)
        return ros._pose_request_callback(me, request, None)

    reply = ask("fr3_link0")
    p, q = reply.pose.position, reply.pose.orientation
    assert reply.success and np.allclose([p.x, p.y, p.z], [0.4, 0.0, 0.4])
    assert np.allclose([q.x, q.y, q.z, q.w], [0.0, 0.0, -np.sqrt(0.5), np.sqrt(0.5)])  # the base's -90 deg
    p = ask("Scene_1").pose.position  # the scene's own frame, as a solver with frame_id Scene_<i> moves in
    assert np.allclose([p.x, p.y, p.z], [-0.3, 0.4, 0.413])
    p = ask("").pose.position  # no frame: world, as before
    assert np.allclose([p.x, p.y, p.z], [-0.3, 10.4, 1.413])
    reply = ask("ur_base_link")
    assert not reply.success and "ur_base_link" in reply.message


def test_a_pose_request_says_what_it_cannot_answer_in(isaac_import):
    """C3b review M2/M3: two robots sharing a base_name, a missing base prim and a path outside a
    scene are refused with a message that says so (get_world_poses answers None for a missing prim)."""
    from guide_msgs.srv import Pose as PoseSrv

    from guide_core.types.geometry import Point, Pose

    ros = isaac_import("guide_core.ros.guide_ros").GUIDEROS2Interface
    world = {"/Scene_0/blocks/red_block": Pose(Point([0.4, 0.0, 0.0])), "/World/light": Pose()}
    two = {"robots": {"left": {"path": "/fr3_left", "base_name": "fr3_link0"},
                      "right": {"path": "/fr3_right", "base_name": "fr3_link0"}}}
    scenes = [SimpleNamespace(_config=two)]
    backend = SimpleNamespace(call=lambda name, prim_path: world.get(prim_path), _scene_manager=SimpleNamespace(_scenes=scenes))
    me = SimpleNamespace(_backend=backend)

    def ask(path):
        return ros._pose_request_callback(me, PoseSrv.Request(path=path, frame_id="fr3_link0"), None)

    reply = ask("/Scene_0/blocks/red_block")
    assert not reply.success and reply.message == "Scene_0 has more than one robot with base_name 'fr3_link0'"
    scenes[0] = SimpleNamespace(_config={"robots": {"franka": {"path": "/fr3", "base_name": "fr3_link0"}}})
    assert ask("/Scene_0/blocks/red_block").message == "Scene_0 has no prim /Scene_0/fr3/fr3_link0"
    assert ask("/World/light").message == "PoseRequest path must be under /Scene_<i>/, got '/World/light'"


def test_an_arrangement_goes_from_the_service_to_the_scene(isaac_import, command_module):
    from guide_msgs.srv import Randomize

    ros = isaac_import("guide_core.ros.guide_ros").GUIDEROS2Interface
    backend = SimpleNamespace(randomize_scene=MagicMock(return_value="{}"))
    me = SimpleNamespace(_backend=backend, _logger=MagicMock())
    ros._randomize_callback(me, Randomize.Request(id=0, arrangement="demo_3"), None)
    ros._randomize_callback(me, Randomize.Request(id=0), None)
    assert [c.kwargs["arrangement"] for c in backend.randomize_scene.call_args_list] == ["demo_3", None]

    module = command_module("_cmd_simulator")
    seen = []
    manager = SimpleNamespace(randomize_preprocess=lambda i, **kwargs: seen.append(kwargs) or [],
                              randomize_postprocess=lambda i, results: "{}")
    runtime = host(_simulator=SimpleNamespace(_scene_manager=manager), _execute_instructions_directly=lambda i: [])
    module._cmd_randomize_scene(runtime, 0, arrangement="demo_3")
    assert seen[0]["arrangement"] == "demo_3"

def test_registration_builds_the_stage_before_anything_reads_it(command_module):
    """Profile robots and the scene recipe (build_stage) exist before the robot graphs, the cameras
    and randomize.yaml look for them."""
    module = command_module("_cmd_simulator")
    calls = []
    scene = SimpleNamespace(replicator_files=None, build_stage=lambda: calls.append("build_stage"))
    manager = SimpleNamespace(get_scene_usd_path=lambda i: None,
                              get_scene_robot_graphs=lambda i: calls.append("robots") or [],
                              get_scene_camera_graphs=lambda i: calls.append("cameras") or [], _scenes=[scene])
    runtime = host(
        _simulator=SimpleNamespace(_add_scene_to_manager=lambda p: (0, (0, 0, 0), {}), _scene_manager=manager, _sim_path="/Sim_0"),
        _cmd_add_scene=lambda **kwargs: calls.append("add_scene"),
        _cmd_set_world_poses=lambda **kwargs: calls.append("place"),
    )

    module._cmd_register_scene(runtime, "task")

    assert calls == ["add_scene", "place", "build_stage", "robots", "cameras"]
