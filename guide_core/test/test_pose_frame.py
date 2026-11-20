import pytest

from guide_core.ros.guide_ros import pose_frame


class _Scene:
    _config = {"robots": {
        "psm1": {"path": "/psm1", "base_name": "psm_base_link"},
        "psm2": {"path": "/psm2", "base_name": "psm_base_link"},
        "fr3": {"path": "/fr3", "base_name": "fr3_link0"},
    }}


SCENES = [_Scene()]


def test_qualified_picks_robot():
    assert pose_frame(SCENES, "/Scene_0/x", "psm2/psm_base_link") == "/Scene_0/psm2/psm_base_link"


def test_bare_ambiguous_hints():
    with pytest.raises(ValueError, match="psm1/psm_base_link"):
        pose_frame(SCENES, "/Scene_0/x", "psm_base_link")


def test_unique_bare_resolves():
    assert pose_frame(SCENES, "/Scene_0/x", "fr3_link0") == "/Scene_0/fr3/fr3_link0"
