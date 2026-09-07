# flake8: noqa: E402
"""One camera, up to two streams, and the names they get.

``resolve_cameras`` is the single place that turns the task's ``cameras:`` block into a
plan -- which streams exist, what the dataset calls them, how they go over the wire --
so that the recorder's annotators and the ROS 2 publisher graphs cannot disagree about
a camera the way they used to about its resolution.

The naming rule is the part worth pinning down, because it is not "append the modality":
a camera whose only stream is depth keeps the plain key. The suffix disambiguates one
camera's two streams, and there is nothing to disambiguate when there is only one.
"""

import logging
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest


@pytest.fixture
def orchestrator():
    """``guide_core.scene.scene_orchestrator`` with Isaac stubbed out."""
    for name in ("omni", "omni.replicator", "omni.replicator.core"):
        sys.modules.setdefault(name, MagicMock())

    import guide_core.scene.scene_orchestrator as module

    return module


def plan(orchestrator, cameras, images, encoding=None):
    """Run ``resolve_cameras`` against a config, without building a whole scene."""
    config = {"cameras": cameras, "dataset": {"images": images}}
    if encoding is not None:
        config["camera_encoding"] = encoding
    scene = SimpleNamespace(
        _config=config,
        _logger=logging.getLogger("test"),
        DEPTH_SUFFIX=orchestrator.SceneOrchestrator.DEPTH_SUFFIX,
    )
    return {c["name"]: c for c in orchestrator.SceneOrchestrator.resolve_cameras(scene)}


CAMERA = {"path": "/top", "width": 640, "height": 480, "topic": "/cam_top"}


def test_rgb_only_camera_keeps_the_dataset_key(orchestrator):
    cameras = plan(orchestrator, {"top": CAMERA}, [{"top": "top"}])

    assert cameras["top"]["rgb_feature"] == "top"
    assert cameras["top"]["depth_feature"] is None


def test_both_streams_put_the_suffix_on_the_depth_one(orchestrator):
    cameras = plan(orchestrator, {"top": {**CAMERA, "depth": True}}, [{"top": "top"}])

    assert cameras["top"]["rgb_feature"] == "top"
    assert cameras["top"]["depth_feature"] == "top_depth"


def test_depth_only_camera_keeps_the_plain_key(orchestrator):
    # The suffix separates one camera's two streams. With one stream there is nothing
    # to separate, and a policy trained on `top` should not have to know the simulator
    # switched that camera from colour to depth.
    cameras = plan(orchestrator, {"top": {**CAMERA, "rgb": False, "depth": True}}, [{"top": "top"}])

    assert cameras["top"]["rgb_feature"] is None
    assert cameras["top"]["depth_feature"] == "top"


def test_the_feature_key_is_the_dataset_key_not_the_camera_name(orchestrator):
    cameras = plan(orchestrator, {"wrist": {**CAMERA, "depth": True}}, [{"hand": "wrist"}])

    assert cameras["wrist"]["rgb_feature"] == "hand"
    assert cameras["wrist"]["depth_feature"] == "hand_depth"


def test_a_camera_outside_dataset_images_is_published_but_not_recorded(orchestrator):
    cameras = plan(orchestrator, {"top": {**CAMERA, "depth": True}}, [])

    assert cameras["top"]["rgb_feature"] is None
    assert cameras["top"]["depth_feature"] is None
    assert cameras["top"]["rgb"] and cameras["top"]["depth"]


def test_defaults_are_rgb_on_depth_off_raw_encoding(orchestrator):
    cameras = plan(orchestrator, {"top": CAMERA}, [{"top": "top"}])

    assert cameras["top"]["rgb"] is True
    assert cameras["top"]["depth"] is False
    assert cameras["top"]["encoding"] == "rgb"


def test_encoding_reaches_every_camera(orchestrator):
    cameras = plan(orchestrator, {"top": CAMERA}, [{"top": "top"}], encoding="rgb_h264")

    assert cameras["top"]["encoding"] == "rgb_h264"


def test_a_camera_with_no_streams_is_dropped(orchestrator):
    cameras = plan(
        orchestrator, {"top": {**CAMERA, "rgb": False, "depth": False}}, [{"top": "top"}]
    )

    assert cameras == {}


def test_depth_is_uint16_millimetres_with_a_channel_axis(orchestrator):
    metres = np.array([[0.0, 1.5], [0.001, 12.0]], dtype=np.float32)

    depth = orchestrator.depth_to_uint16_mm(metres)

    assert depth.dtype == np.uint16
    assert depth.shape == (2, 2, 1)
    assert depth[..., 0].tolist() == [[0, 1500], [1, 12000]]


def test_misses_read_as_zero_not_as_maximum_range(orchestrator):
    # distance_to_image_plane returns +inf where the ray hit nothing. Saturating those
    # to 65.535 m would put a wall at maximum range wherever the sky is, and drag the
    # dataset's depth statistics with it.
    depth = orchestrator.depth_to_uint16_mm(np.array([[np.inf, np.nan, 100.0]], dtype=np.float32))

    assert depth[..., 0].tolist() == [[0, 0, 65535]]
