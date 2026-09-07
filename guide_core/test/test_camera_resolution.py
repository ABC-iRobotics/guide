# flake8: noqa: E402
"""The camera graph must publish at the resolution the recorder renders at.

``Ros2CameraGraph`` leaves its ``IsaacCreateRenderProduct`` node on the node's own
1280x720 default, while ``scene_orchestrator.create_render_products`` builds the
recorder's annotators at the size ``config/init.yaml`` asks for. A dataset recorded at
640x480 and an inference run reading a 1280x720 topic are two different cameras, and
nothing downstream reports it -- ``ros2camera`` quietly scales and centre-crops the
larger image, costing a quarter of the horizontal field of view and, at 16:9 against
4:3, another quarter vertically.
"""

import sys
from unittest.mock import MagicMock

import pytest


class FakeAttribute:
    """An OmniGraph attribute that actually remembers what it was set to."""

    def __init__(self, store, path, writable=True):
        self.store = store
        self.path = path
        self.writable = writable

    def set(self, value):
        if self.writable:
            self.store[self.path] = value

    def get(self):
        return self.store[self.path]


class FakeController:
    def __init__(self, store, missing=(), read_only=()):
        self.store = store
        self.missing = set(missing)
        self.read_only = set(read_only)

    def attribute(self, path):
        name = path.rsplit(":", 1)[-1]
        if name in self.missing:
            raise ValueError(f"no attribute {path}")
        self.store.setdefault(path, {"width": 1280, "height": 720}.get(name, "rgb"))
        return FakeAttribute(self.store, path, writable=name not in self.read_only)


@pytest.fixture
def commands():
    """``guide_core.core.commands._cmd_robot`` with Isaac stubbed out."""
    for name in (
        "isaacsim",
        "isaacsim.core",
        "isaacsim.core.api",
        "isaacsim.core.api.robots",
        "isaacsim.core.utils",
        "isaacsim.core.utils.types",
        "isaacsim.ros2",
        "isaacsim.ros2.ui",
        "isaacsim.ros2.ui.og_rtx_sensors",
        "isaacsim.ros2.ui.og_utils",
        "isaacsim.sensors",
        "isaacsim.sensors.camera",
        "omni",
        "omni.graph",
        "omni.graph.core",
    ):
        sys.modules.setdefault(name, MagicMock())
    sys.modules.pop("guide_core.core.commands._cmd_robot", None)
    import guide_core.core.commands._cmd_robot as module

    return module


def test_the_configured_resolution_reaches_the_render_product(commands):
    store = {}
    commands.og.Controller = FakeController(store)

    commands._set_render_resolution("/Scene_0/Graph/top_camera_graph", 640, 480)

    assert store == {
        "/Scene_0/Graph/top_camera_graph/RenderProduct.inputs:width": 640,
        "/Scene_0/Graph/top_camera_graph/RenderProduct.inputs:height": 480,
    }


def test_without_it_the_node_keeps_1280x720(commands):
    # The bug this guards: the node's own default, which is what the topic published
    # while every dataset was rendered at 640x480.
    store = {}
    controller = FakeController(store)
    controller.attribute("/g/RenderProduct.inputs:width")
    controller.attribute("/g/RenderProduct.inputs:height")

    assert list(store.values()) == [1280, 720]


def test_a_renamed_node_is_an_error_not_a_silent_default(commands):
    commands.og.Controller = FakeController({}, missing={"width"})

    with pytest.raises(RuntimeError, match="1280x720"):
        commands._set_render_resolution("/g", 640, 480)


def test_a_value_that_does_not_stick_is_an_error(commands):
    commands.og.Controller = FakeController({}, read_only={"height"})

    with pytest.raises(RuntimeError, match="height"):
        commands._set_render_resolution("/g", 640, 480)


def test_the_default_size_matches_the_callers_fallback(commands):
    # _cmd_simulator passes camera.get("width", 640) / camera.get("height", 480), and
    # the recorder renders at whatever init.yaml says. A signature default that
    # disagreed would reintroduce the same mismatch by another route.
    import inspect

    parameters = inspect.signature(commands._cmd_create_camera).parameters

    assert parameters["width"].default == 640
    assert parameters["height"].default == 480


def test_the_encoding_reaches_the_camera_helper(commands):
    # Ros2CameraGraph hard-codes RGBPublish.inputs:type = "rgb"; nothing in its API
    # exposes it, so a compressed stream exists only if this write lands.
    store = {}
    commands.og.Controller = FakeController(store)

    commands._set_rgb_encoding("/Scene_0/Graph/top_camera_graph", "rgb_h264")

    assert store == {"/Scene_0/Graph/top_camera_graph/RGBPublish.inputs:type": "rgb_h264"}


def test_raw_rgb_leaves_the_node_untouched(commands):
    store = {}
    commands.og.Controller = FakeController(store)

    commands._set_rgb_encoding("/g", "rgb")

    assert store == {}


def test_an_encoding_that_does_not_stick_is_an_error(commands):
    # A silently-raw topic is the failure this guards: the subscriber discovers the
    # wire format off the graph, so it would happily read raw images at 166 MB/s and
    # nothing would report that compression was asked for and never happened.
    commands.og.Controller = FakeController({}, read_only={"type"})

    with pytest.raises(RuntimeError, match="rgb_h264"):
        commands._set_rgb_encoding("/g", "rgb_h264")


def test_a_renamed_publisher_node_is_an_error_not_a_silent_raw_topic(commands):
    commands.og.Controller = FakeController({}, missing={"type"})

    with pytest.raises(RuntimeError, match="RGBPublish"):
        commands._set_rgb_encoding("/g", "rgb_h264")


def test_the_stream_defaults_match_the_callers_fallbacks(commands):
    # _cmd_simulator passes camera.get("rgb", True) / ("depth", False) / ("encoding",
    # "rgb"), and SceneOrchestrator.resolve_cameras resolves the same three. A
    # signature default that disagreed would make a camera's behaviour depend on which
    # of the two built the dict.
    import inspect

    parameters = inspect.signature(commands._cmd_create_camera).parameters

    assert parameters["rgb"].default is True
    assert parameters["depth"].default is False
    assert parameters["encoding"].default == "rgb"
