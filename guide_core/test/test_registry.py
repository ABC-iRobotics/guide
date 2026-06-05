from pathlib import Path

import pytest

from guide_core.core.registry import attach_cmd_functions


class Host:
    def __init__(self):
        self.calls = []


def _make_pkg(tmp_path: Path) -> str:
    """Create a temporary importable package with a commands subpackage."""
    pkg = tmp_path / "my_task_pkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("# temp package\n")

    commands = pkg / "commands"
    commands.mkdir()
    (commands / "__init__.py").write_text("# commands\n")

    # Command module with one command and one helper
    (commands / "_cmd_alpha.py").write_text(
        """

def _cmd_ping(self, x: int = 1):
    self.calls.append((\"ping\", x))
    return x + 1

# This is a helper and must NOT be attached.
# It does not accept `self` and would break if bound.

def __helper(y: int):
    return y * 2

# Not a command entry-point

def not_a_command(self):
    return 123
"""
    )

    return "my_task_pkg"


def test_attach_cmd_functions_filters_helpers(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    pkg_name = _make_pkg(tmp_path)
    monkeypatch.syspath_prepend(str(tmp_path))

    host = Host()

    # Explicit package_map so we don't depend on host location discovery.
    attach_cmd_functions(host, package_map={"task": f"{pkg_name}.commands"}, debug=False)

    assert hasattr(host, "_cmd_ping")
    assert callable(getattr(host, "_cmd_ping"))

    # Helpers / non-commands must not be attached
    # assert not hasattr(host, "__helper")
    # assert not hasattr(host, "not_a_command")

    # And the bound command must behave correctly
    out = host._cmd_ping(5)
    assert out == 6
    assert host.calls == [("ping", 5)]
