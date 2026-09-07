import omni.graph.core as og
from isaacsim.core.api.robots import Robot
from isaacsim.core.utils.types import ArticulationAction

# OmniGraph ROS 2 shortcut helpers. These moved across Isaac Sim versions:
#   4.5: isaacsim.ros2.bridge.scripts.og_shortcuts
#   5.x: isaacsim.ros2.bridge.impl.og_shortcuts
#   6.0: isaacsim.ros2.ui
from isaacsim.ros2.ui.og_rtx_sensors import Ros2CameraGraph
from isaacsim.ros2.ui.og_utils import (
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


def _finalize_graph(graph_window) -> None:
    """Build the OmniGraph, then discard the helper window so it never pops up.

    The Ros2*Graph classes (isaacsim.ros2.ui) are ``MenuHelperWindow`` /
    ``ui.Window`` subclasses meant for interactive menu use — instantiating one
    opens a window. GUIDE only needs the graph, so build it (``make_graph`` ->
    ``og.Controller``) and destroy the window. It is hidden first so it can't
    flash on the next render; ``make_graph`` no-ops gracefully if the graph
    already exists.
    """
    graph_window.make_graph()
    try:
        graph_window.visible = False
    except Exception:
        pass
    try:
        graph_window.destroy()
    except Exception:
        pass


def _cmd_create_clock(self, namespace: str = "", path: str | None = None) -> None:

    assert self.state in [READY, PAUSED, STOPPED]

    clock = Ros2ClockGraph()
    clock._publisher = True
    clock._subscriber = False

    clock._node_namespace = namespace
    if path is not None:
        clock._og_path = path

    print("Creating clock")
    _finalize_graph(clock)


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
            self._logger.error(f"Failed to add robot {articulation_root} to scene: {e}")

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

    _finalize_graph(js_graph)


def _set_render_resolution(graph_path: str, width: int, height: int) -> None:
    """Force a camera graph's render product to the configured resolution.

    ``Ros2CameraGraph`` builds its ``IsaacCreateRenderProduct`` node with only
    ``inputs:cameraPrim`` set, so the node keeps its own defaults --  1280x720, from
    ``OgnIsaacCreateRenderProduct.ogn``. Nothing downstream complains, and the result
    is that the published images do not match the ones a dataset was recorded from:
    ``scene_orchestrator.create_render_products`` builds the recorder's annotator
    render products at the size ``config/init.yaml`` asks for, so a dataset renders at
    640x480 while inference reads a 1280x720 topic. ``irob_lerobot_ros.ros2camera``
    then scales that to 853x480 and centre-crops to 640x480, keeping 75% of the
    horizontal field of view, and 16:9 against 4:3 costs another quarter vertically --
    about a 1.33x zoom into a framing the policy never trained on. Both halves of the
    pipeline have to render at the same resolution or the policy is fed a different
    camera than the one it learned from.

    Set after ``make_graph``: the node does not exist before it, and ``make_graph``
    also moves ``_og_path`` on to the next free path. Read back rather than assumed,
    because a node renamed in a future Isaac release would otherwise put us straight
    back to a silent mismatch.
    """
    for name, value in (("width", int(width)), ("height", int(height))):
        attribute_path = f"{graph_path}/RenderProduct.inputs:{name}"
        try:
            attribute = og.Controller.attribute(attribute_path)
            attribute.set(value)
            written = attribute.get()
        except Exception as error:  # node renamed, or the graph failed to build
            raise RuntimeError(
                f"Could not set '{attribute_path}'. The camera would publish at the "
                f"node's default 1280x720 instead of {width}x{height}, which silently "
                f"mismatches the resolution the recorder renders at."
            ) from error
        if written != value:
            raise RuntimeError(f"'{attribute_path}' kept {written} after being set to {value}.")


# One camera prim publishes up to two ROS 2 streams, laid out the way image_transport
# lays them out so a raw subscriber and a compressed one never collide on one topic
# name (two message types on one name is legal in DDS and unreadable in `ros2 topic`):
#
#   <topic>              sensor_msgs/Image           rgb8               encoding: rgb
#   <topic>/compressed   sensor_msgs/CompressedImage h264               encoding: rgb_h264
#   <topic>/depth        sensor_msgs/Image           32FC1, metres      depth: true
#
# ``irob_lerobot_ros.ros2camera`` discovers which of the first two the camera helper
# actually created rather than being told, so a policy does not have to be configured
# to match the simulator it happens to be pointed at.
def _rgb_topic_for(topic: str, encoding: str) -> str:
    return topic if encoding == "rgb" else f"{topic}/compressed"


def _depth_topic_for(topic: str) -> str:
    return f"{topic}/depth"


def _set_rgb_encoding(graph_path: str, encoding: str) -> None:
    """Switch the camera graph's RGB publisher from raw Image to H.264 CompressedImage.

    ``Ros2CameraGraph.make_graph`` hard-codes ``RGBPublish.inputs:type = "rgb"``
    (isaacsim/ros2/ui/og_rtx_sensors.py), which publishes ``sensor_msgs/Image``: 640x480
    rgb8 is 921 kB per message, per camera, per rendered frame. ``rgb_h264`` swaps the
    writer for ``ROS2PublishCompressedImage`` -- ``sensor_msgs/CompressedImage``,
    ``format: "h264"``, encoded on the GPU by NVENC -- for roughly 1% of the bytes.

    It does not reduce the renderer's work: the render product still renders every
    frame. This is a transport fix (it stops the image stream starving small service
    replies over localhost DDS), not a frame-budget one.

    Must run after ``make_graph`` and before the first tick: ``ROS2CameraHelper.compute``
    reads ``inputs:type`` once and latches ``state.initialized``. ``make_graph`` stops
    the timeline, so nothing has ticked yet. Read back for the same reason
    ``_set_render_resolution`` does -- a node renamed in a future Isaac release would
    otherwise leave the topic silently raw.
    """
    if encoding == "rgb":
        return

    attribute_path = f"{graph_path}/RGBPublish.inputs:type"
    try:
        attribute = og.Controller.attribute(attribute_path)
        attribute.set(encoding)
        written = attribute.get()
    except Exception as error:  # node renamed, or the graph failed to build
        raise RuntimeError(
            f"Could not set '{attribute_path}' to '{encoding}'. The camera would publish "
            f"raw sensor_msgs/Image on a topic nothing is subscribed to."
        ) from error
    if written != encoding:
        raise RuntimeError(f"'{attribute_path}' kept '{written}' after being set to '{encoding}'.")


def _cmd_create_camera(
    self,
    pose: Pose | None = None,
    camera_path: str = "/Camera",
    path: str | None = None,
    # Matches the caller's fallback in _cmd_simulator and the recorder's render
    # products: both halves must render at the same size (see _set_render_resolution).
    width: int = 640,
    height: int = 480,
    frame: str = "sim_camera",
    namespace: str = "",
    topic: str = "/rgb",
    # Which streams this camera publishes, and in what form. Resolved upstream by
    # SceneOrchestrator.resolve_cameras() from the task's config/init.yaml; the
    # defaults here are the ones a camera gets when the config says nothing.
    rgb: bool = True,
    depth: bool = False,
    encoding: str = "rgb",
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
    cp._rgb_pub = rgb
    cp._rgb_topic = _rgb_topic_for(topic, encoding)
    cp._depth_pub = depth
    cp._depth_topic = _depth_topic_for(topic)

    print("Creating camera")
    _finalize_graph(cp)
    # Both publishers hang off one IsaacCreateRenderProduct, so this sizes them both.
    # The graph is built with the node's own 1280x720 default until this runs.
    _set_render_resolution(cp._og_path, width, height)
    if rgb:
        _set_rgb_encoding(cp._og_path, encoding)


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
    _finalize_graph(tf_g)


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
