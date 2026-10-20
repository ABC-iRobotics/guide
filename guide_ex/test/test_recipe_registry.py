"""The recipe registry: @node leaves, YAML templates by file stem, clashes named with both sources."""

import subprocess
import sys

import pytest

from guide_ex.core.states import Layer
from guide_ex.recipe import RecipeError, RecipeTemplate, Registry, default_registry, node
from guide_ex.recipe.registry import load_defaults
from guide_ex.utility.pose import TransformPose
from guide_ex.utility.wait import WaitForSeconds

LEAVES = {
    "SetGripperState", "MoveToCartesianPose", "MoveWithCartesianVelocity", "MoveToJointConfiguration",
    "GetPrimPose", "GetPrimPoses", "IsPrimClashing", "IsTaskSuccessful", "GetItem", "NodeException",
    "TransformPose", "InvertPose", "ChainLength", "StartRecording", "PauseRecording", "StopRecording",
    "SetPrompt", "ProjectRotationToBaseZ", "ReduceRotationToSymmetry", "WaitForSeconds",
}


def test_every_leaf_registers_under_its_class_name():
    registry = load_defaults()

    assert LEAVES <= set(registry.names())
    assert registry.resolve("TransformPose") is TransformPose


def test_the_decorator_registers_and_can_set_the_level():
    from guide_ex.core.base_node import BaseNode

    try:
        @node(level=Layer.STEP, name="DecoratedForTest")
        class Decorated(BaseNode):
            def run(self):
                pass

        assert default_registry.resolve("DecoratedForTest") is Decorated
        assert Decorated.level == Layer.STEP
    finally:
        default_registry._entries.pop("DecoratedForTest", None)


def test_templates_register_by_file_stem(tmp_path):
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "Hold.yaml").write_text("ref: WaitForSeconds\nstatic_args: {seconds: 1.0}\n")
    registry = Registry()
    registry.add_search_path(tmp_path)
    registry.add_search_path(tmp_path)  # the same file again is no clash

    template = registry.resolve("Hold")
    assert isinstance(template, RecipeTemplate)
    assert template.spec == {"ref": "WaitForSeconds", "static_args": {"seconds": 1.0}}
    assert registry.names() == ["Hold"]


def test_a_clash_names_both_sources(tmp_path):
    (tmp_path / "WaitForSeconds.yaml").write_text("ref: TransformPose\n")
    registry = Registry()
    registry.register_class("WaitForSeconds", WaitForSeconds)

    both = r"'WaitForSeconds' is registered twice: guide_ex\.utility\.wait\.WaitForSeconds and .*WaitForSeconds\.yaml"
    with pytest.raises(RecipeError, match=both):
        registry.add_search_path(tmp_path)


def test_an_unknown_name_is_refused():
    with pytest.raises(RecipeError, match="unknown ref 'NoSuchNode'"):
        Registry().resolve("NoSuchNode")


def test_a_light_leaf_does_not_import_the_others():
    """cube_stack's scene imports guide_ex.utility.pose inside Isaac: it must not pull in
    lerobot, rclpy and the rest through the registry."""
    code = "import sys, guide_ex.utility.pose; print('lerobot' in sys.modules, 'rclpy' in sys.modules)"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout

    assert out.split() == ["False", "False"]


def test_malformed_yaml_is_a_recipe_error_naming_the_file(tmp_path):
    (tmp_path / "Broken.yaml").write_text("ref: [WaitForSeconds\nstatic_args: {seconds: 1.0\n")

    with pytest.raises(RecipeError, match=r"Broken\.yaml: "):
        Registry().add_search_path(tmp_path)
