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
from pathlib import Path
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
        INSTANCE_SUFFIX=orchestrator.SceneOrchestrator.INSTANCE_SUFFIX,
    )
    # The real property, so the plan's fps and the recorder's fps cannot drift apart.
    scene.record_frequency = orchestrator.SceneOrchestrator.record_frequency.fget(scene)
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


def test_the_plan_carries_the_dataset_rate_to_the_publisher(orchestrator):
    # The topics publish at the rate the dataset was recorded at; the runtime turns it
    # into a frameSkipCount, because only it knows the render rate.
    cameras = plan(orchestrator, {"top": CAMERA}, [{"top": "top"}])
    assert cameras["top"]["fps"] == 10.0

    cameras = plan(orchestrator, {"top": CAMERA}, [{"top": "top"}])
    assert cameras["top"]["fps"] == 10.0


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


def frequency(orchestrator, config):
    """The rate SceneManager samples at and the camera topics are gated to."""
    return orchestrator.SceneOrchestrator.record_frequency.fget(SimpleNamespace(_config=config))


def test_the_capture_rate_comes_from_dataset_fps(orchestrator):
    assert frequency(orchestrator, {"dataset": {"fps": 20}}) == 20.0


def test_the_rate_and_the_recorded_fps_are_one_number(orchestrator):
    # The bug this closes: SceneManager sampled at a `getattr(scene,
    # "record_frequency", 10)` that nothing in the repo assigned, while the recorder
    # stamped `fps=30` into the dataset metadata. Every dataset written before this
    # claims a rate 3x what it was recorded at. Both sides now read one key with one
    # default, and this test fails if either grows its own literal again.
    import inspect

    from guide_core.scene.scene_recorder import DEFAULT_FPS, SceneRecorder

    assert frequency(orchestrator, {}) == float(DEFAULT_FPS)
    assert DEFAULT_FPS == 10, "10 Hz is what the recorder has always actually captured at"

    stamped = inspect.getsource(SceneRecorder._initialize_dataset)
    assert 'get("fps", DEFAULT_FPS)' in stamped
    assert 'get("fps", 30)' not in stamped


def semantics(orchestrator, config, scene_id=0):
    scene = SimpleNamespace(_config=config, _scene_id=scene_id, _logger=logging.getLogger("test"))
    return orchestrator.SceneOrchestrator.resolve_semantics(scene)


def test_semantic_patterns_are_scoped_to_the_scene(orchestrator):
    # The task config is written relative to the scene root so one config works in
    # whichever Scene_N it is instantiated into -- the same convention reset.yaml and
    # randomize.yaml use for prim_path.
    labels = semantics(
        orchestrator,
        {"dataset": {"tracked_objects": {"red_block": "/blocks/red_block"}}},
        scene_id=3,
    )

    assert labels == {"red_block": "/Scene_3/blocks/red_block"}


def test_a_task_without_semantics_labels_nothing(orchestrator):
    assert semantics(orchestrator, {}) == {}
    assert semantics(orchestrator, {"dataset": {}}) == {}
    assert semantics(orchestrator, {"dataset": {"tracked_objects": None}}) == {}


def test_instance_stream_gets_its_own_suffix(orchestrator):
    camera = {**CAMERA, "depth": True, "instance": True}
    cameras = plan(orchestrator, {"top": camera}, [{"top": "top"}])

    features = [cameras["top"][f] for f in ("rgb_feature", "depth_feature", "instance_feature")]
    assert features == ["top", "top_depth", "top_instance"]


def test_instance_frames_paint_tracked_objects_in_their_own_colour(orchestrator):
    scene = SimpleNamespace(
        instance_colors={"robot": (1, 1, 1), "bin_0": (2, 2, 2)}, segmentation=None
    )
    ids = np.array([[0, 1, 5], [7, 9, 9]], np.uint32)
    info = {
        "idToLabels": {
            "0": {"class": "BACKGROUND"},
            "1": {"class": "UNLABELLED"},
            "5": {"class": "robot, camera"},  # inherited labels arrive comma-joined
            "7": {"class": "bin_0"},
            "9": {"class": "table"},  # labelled, but not tracked
        }
    }

    image = orchestrator.SceneOrchestrator.colorize_instances(scene, {"data": ids, "info": info})

    assert image.dtype == np.uint8 and image.shape == (2, 3, 3)
    assert image[:, :, 0].tolist() == [[0, 0, 1], [2, 0, 0]]


# --- dataset.segmentation: tracked-object colours from an RGB or a named colour table ---

ADE20K = Path(__file__).parents[1] / "config" / "segmentation_colors" / "ade20k.csv"
TRACKED = {"blue_block": "/blocks/blue_block", "bin_0": "/bin_0", "robot": "/fr3"}


def test_a_table_name_matches_the_whole_cell_then_the_first_token_then_any_token(orchestrator):
    table = orchestrator.load_color_table(ADE20K)

    assert len(table) == 150
    # "box" is row 42's whole cell, though row 90 (television;...;idiot;box;...) has the token.
    assert orchestrator.lookup_color(table, "box") == (0, 255, 20)
    assert orchestrator.lookup_color(table, "basket") == (92, 255, 0)  # row 113's first token
    assert orchestrator.lookup_color(table, "computer") == (0, 255, 173)  # row 75's first token
    assert orchestrator.lookup_color(table, "handbasket") == (92, 255, 0)  # any token, one row


def test_an_ambiguous_or_unknown_name_is_an_error(orchestrator):
    table = orchestrator.load_color_table(ADE20K)

    with pytest.raises(ValueError, match="machine.*21.*75.*79.*108.*130"):
        orchestrator.lookup_color(table, "machine")
    with pytest.raises(ValueError, match="no row.*spaceship"):
        orchestrator.lookup_color(table, "spaceship")


def test_an_rgb_is_three_ints_in_0_to_255(orchestrator):
    assert orchestrator.as_rgb([255, 6, 82], "x") == (255, 6, 82)
    for bad in ([256, 0, 0], [-1, 0, 0], [1, 2], [1, 2, 3, 4], [1.0, 2, 3], [True, 0, 0], "red"):
        with pytest.raises(ValueError, match="x must be"):
            orchestrator.as_rgb(bad, "x")


def colors_of(orchestrator, config, task_dir="/nonexistent"):
    """instance_colors and the guide_info.json legend of a scene with this config."""
    scene = SimpleNamespace(
        _config=config,
        _scene_id=0,
        INSTANCE_PALETTE=orchestrator.SceneOrchestrator.INSTANCE_PALETTE,
        segmentation=orchestrator.segmentation_colors(config, task_dir),
    )
    scene.resolve_semantics = lambda: orchestrator.SceneOrchestrator.resolve_semantics(scene)
    scene.instance_colors = orchestrator.SceneOrchestrator.instance_colors.fget(scene)
    return scene.instance_colors, orchestrator.SceneOrchestrator.instance_legend.fget(scene)


def test_without_a_segmentation_section_the_colours_are_tab10_as_before(orchestrator):
    colors, legend = colors_of(orchestrator, {"dataset": {"tracked_objects": TRACKED}})

    assert colors == {
        "blue_block": (31, 119, 180), "bin_0": (255, 127, 14), "robot": (44, 160, 44)
    }
    assert legend is None


def test_segmentation_colors_paint_tracked_labels_from_a_table_or_an_rgb(orchestrator):
    config = {"dataset": {"tracked_objects": TRACKED, "segmentation": {
        "tables": {"ade20k": str(ADE20K)},
        "colors": {"blue_block": "ade20k:box", "robot": [255, 6, 82]},
        "background": [1, 2, 3],
    }}}

    colors, legend = colors_of(orchestrator, config)

    # bin_0 has no entry: it keeps its tab10 colour, and the legend says so.
    assert colors == {"blue_block": (0, 255, 20), "bin_0": (255, 127, 14), "robot": (255, 6, 82)}
    assert legend == {
        "background": [1, 2, 3],
        "colors": {
            "blue_block": {
                "path": "/blocks/blue_block", "source": "ade20k:box", "rgb": [0, 255, 20]
            },
            "bin_0": {"path": "/bin_0", "source": "tab10", "rgb": [255, 127, 14]},
            "robot": {"path": "/fr3", "source": [255, 6, 82], "rgb": [255, 6, 82]},
        },
    }


def test_the_background_defaults_to_black(orchestrator):
    config = {"dataset": {
        "tracked_objects": TRACKED, "segmentation": {"colors": {"robot": [9, 9, 9]}}
    }}

    assert orchestrator.segmentation_colors(config, "/nonexistent")["background"] == [0, 0, 0]


def test_a_bare_table_name_is_guide_cores_and_a_relative_path_is_the_tasks(orchestrator, tmp_path):
    (tmp_path / "config").mkdir()
    (tmp_path / "config" / "mine.csv").write_text(
        'Idx,Name,"Color_Code (R,G,B)"\n1,crate;box,"(1, 2, 3)"\n'
    )
    config = {"dataset": {"tracked_objects": TRACKED, "segmentation": {
        "tables": {"ade20k": "ade20k", "mine": "config/mine.csv"},  # installed guide_core copy
        "colors": {"blue_block": "ade20k:box", "bin_0": "mine:box"},
    }}}

    colors = orchestrator.segmentation_colors(config, tmp_path)["colors"]

    assert colors["blue_block"]["rgb"] == [0, 255, 20]
    assert colors["bin_0"]["rgb"] == [1, 2, 3]


@pytest.mark.parametrize("section, error", [
    ({"colors": {"robot": "nope:box"}}, "unknown table 'nope'"),
    ({"colors": {"robot": "box"}}, r"\[r, g, b\] or '<table>:<name>'"),
    ({"colors": {"crate": [1, 2, 3]}}, "'crate' is not a dataset.tracked_objects label"),
    ({"colors": {"robot": [1, 2, 300]}}, "colors.robot must be"),
    (
        {"colors": {"robot": "ade20k:machine"}, "tables": {"ade20k": str(ADE20K)}},
        "colors.robot.*several rows",
    ),
    ({"colors": {}, "background": [0, 0]}, "background must be"),
    ({"colors": {}, "tables": {"t": "missing.csv"}}, "tables.t: no file"),
    ({"colours": {}}, "takes tables, colors, background"),
])
def test_a_bad_segmentation_section_fails_the_registration(orchestrator, section, error):
    config = {"dataset": {"tracked_objects": TRACKED, "segmentation": section}}

    with pytest.raises(ValueError, match=error):
        orchestrator.segmentation_colors(config, "/nonexistent")


def test_the_background_colour_fills_every_pixel_no_tracked_object_claims(orchestrator):
    scene = SimpleNamespace(
        instance_colors={"robot": (1, 1, 1)}, segmentation={"background": [7, 8, 9]}
    )
    ids = np.array([[0, 1, 5]], np.uint32)
    info = {"idToLabels": {
        "0": {"class": "BACKGROUND"}, "1": {"class": "UNLABELLED"}, "5": {"class": "robot"}
    }}

    image = orchestrator.SceneOrchestrator.colorize_instances(scene, {"data": ids, "info": info})

    assert image.tolist() == [[[7, 8, 9], [7, 8, 9], [1, 1, 1]]]
