"""Runtime command functions, run against a fake host with Isaac stubbed out.

Each test pins one defect that used to slip through because the code path needs a
running simulator to reach. The command modules import Isaac at module level, so they
are imported under stubs that are removed again afterwards (a MagicMock left in
sys.modules would follow every later test in the session).
"""

import importlib
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
def command_module():
    """``command_module("_cmd_x")`` imports guide_core.core.commands._cmd_x under stubs."""
    saved = {name: sys.modules.get(name) for name in STUBBED}
    loaded = []

    def load(name):
        for stub in STUBBED:
            sys.modules[stub] = MagicMock()
        full = f"guide_core.core.commands.{name}"
        sys.modules.pop(full, None)
        loaded.append(full)
        return importlib.import_module(full)

    yield load
    for name in loaded:
        sys.modules.pop(name, None)
    for name, module in saved.items():
        if module is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = module


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
