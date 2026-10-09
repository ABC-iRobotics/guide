import importlib
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

# Ensure the repository root (containing the guide_core
# package) is importable.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


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
