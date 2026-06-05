from isaacsim.core.api.robots import Robot
from isaacsim.core.utils.types import ArticulationAction
from isaacsim.ros2.bridge.scripts.og_shortcuts.og_rtx_sensors import Ros2CameraGraph
from isaacsim.ros2.bridge.scripts.og_shortcuts.og_utils import (
    Ros2ClockGraph,
    Ros2JointStatesGraph,
    Ros2TfPubGraph,
)
from isaacsim.sensors.camera import Camera

# from isaacsim.sensors.camera import Camera
from guide_core.types.geometry import Pose
from guide_core.types.isaac_state import IsaacState

UNINITIALIZED = IsaacState.UNINITIALIZED
INITIALIZING = IsaacState.INITIALIZING
STOPPED = IsaacState.STOPPED
LOADING = IsaacState.LOADING
READY = IsaacState.READY
RUNNING = IsaacState.RUNNING
PAUSED = IsaacState.PAUSED
ERROR = IsaacState.ERROR
SHUTTING_DOWN = IsaacState.SHUTTING_DOWN


def _cmd_create_clock(self, namespace: str = "", path: str | None = None) -> None:

    assert self.state in [READY, PAUSED, STOPPED]

    clock = Ros2ClockGraph()
    clock._publisher = True
    clock._subscriber = False

    clock._node_namespace = namespace
    if path is not None:
        clock._og_path = path

    print("Creating clock")
    clock.make_graph()


def _cmd_create_robot_control(
    self,
    namespace: str = "",
    articulation_root: str = "",
    path: str | None = None,
    default_joint_states: list[float] | None = None,
) -> None:

    assert self.state in [READY, PAUSED, STOPPED]

    if not hasattr(self, "_robots"):
        self._robots = {}

    if articulation_root not in self._robots:
        try:
            robot = Robot(prim_path=articulation_root)
            self._world.scene.add(robot)
            self._robots[articulation_root] = robot
        except Exception as e:
            print(f"[DEBUG_FREEZE] Failed to add robot {articulation_root} to scene: {e}")

    js_graph = Ros2JointStatesGraph()
    js_graph._publisher = True
    js_graph._subscriber = True
    js_graph._sub_move_robot = True
    js_graph._node_namespace = namespace
    js_graph._art_root_path = articulation_root
    js_graph._pub_topic = "joint_states"
    js_graph._sub_topic = "joint_command"
    js_graph._og_path = path

    if default_joint_states is not None:
        js_graph._default_joint_states = default_joint_states

    js_graph.make_graph()


def _cmd_create_camera(
    self,
    pose: Pose | None = None,
    camera_path: str = "/Camera",
    path: str | None = None,
    width: int = 1920,
    height: int = 1080,
    frame: str = "sim_camera",
    namespace: str = "",
    topic: str = "/rgb",
):

    assert self.state in [READY, PAUSED, STOPPED]

    if pose is not None:
        camera = Camera(
            prim_path=camera_path,
            dt=self._dt,
            resolution=(width, height),
            position=pose.position.to_numpy(),
            orientation=pose.orientation.to_numpy_quat(),
        )

    cp = Ros2CameraGraph()
    if path is not None:
        cp._og_path = path
    cp._camera_prim = camera_path
    cp._frame_id = frame
    cp._node_namespace = namespace
    cp._rgb_topic = topic
    cp._depth_pub = False

    print("Creating camera")
    cp.make_graph()


def _cmd_create_tf_graph(
    self,
    prim: str,
    path: str | None = None,
    parent_prim: str = "/World",
    namespace: str = "",
):

    assert self.state in [READY, PAUSED, STOPPED]

    tf_g = Ros2TfPubGraph()
    if path is not None:
        tf_g._og_path = path
    tf_g._node_namespace = namespace
    tf_g._target_prim = prim
    tf_g._parent_prim = parent_prim

    print("Creating tf graph")
    tf_g.make_graph()


def _cmd_set_joint(
    self,
    articulation_root: str = "",
    joint_positions: list[float] | None = None,
    joint_velocities: list[float] | None = None,
    joint_efforts: list[float] | None = None,
    joint_indices: list[int] | None = None,
):

    assert self.state in [READY, RUNNING, PAUSED, STOPPED]

    if not hasattr(self, "_robots"):
        self._robots = {}

    if articulation_root not in self._robots:
        robot = Robot(prim_path=articulation_root)
        self._world.scene.add(robot)
        self._robots[articulation_root] = robot
    else:
        robot = self._robots[articulation_root]

    if not robot.is_initialized:
        try:
            robot.initialize()
        except Exception as e:
            self._logger.warning(f"Failed to initialize robot: {e}")

    if joint_positions is not None:
        # robot.set_joint_positions(positions=joint_positions, joint_indices=joint_indices)
        action = ArticulationAction(joint_positions=joint_positions, joint_indices=joint_indices)
        robot.apply_action(action)
    if joint_velocities is not None:
        # robot.set_joint_velocities(velocities=joint_velocities, joint_indices=joint_indices)
        action = ArticulationAction(joint_velocities=joint_velocities, joint_indices=joint_indices)
        robot.apply_action(action)
    if joint_efforts is not None:
        # robot.set_joint_efforts(efforts=joint_efforts, joint_indices=joint_indices)
        action = ArticulationAction(joint_efforts=joint_efforts, joint_indices=joint_indices)
        robot.apply_action(action)
