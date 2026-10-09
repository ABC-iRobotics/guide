"""Runtime commands, service replies and the recorder, run against fakes with Isaac stubbed out.

Each test pins one defect that used to slip through because the code path needs a
running simulator to reach. These modules import Isaac at module level, so they are
imported under stubs, and every module imported meanwhile is dropped again afterwards
(a MagicMock left in sys.modules would follow every later test in the session).
"""

import sys
from types import MethodType, SimpleNamespace
from unittest.mock import MagicMock

import pytest

from guide_core.types.isaac_state import IsaacState

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


def test_shutdown_closes_isaac_even_when_stopping_fails(command_module):
    stage = command_module("_cmd_stage")
    h = host(
        state=IsaacState.ERROR,
        _cmd_stop=MagicMock(side_effect=RuntimeError("no world")),
        simulation_app=MagicMock(),
    )

    with pytest.raises(RuntimeError):
        stage._cmd_shutdown(h)

    h.simulation_app.close.assert_called_once()
    assert h.state == IsaacState.UNINITIALIZED
