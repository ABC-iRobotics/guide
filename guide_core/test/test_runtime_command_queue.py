import logging

import pytest

from guide_core.core.runtime import Command, IsaacSimRuntime


class TestableRuntime(IsaacSimRuntime):
    """Subclass that bypasses Isaac initialization for unit testing."""

    def __init__(self):
        # do not call super().__init__
        self._logger = logging.getLogger("TestableRuntime")
        self._cmd_q = __import__("queue").Queue()

    def _cmd_echo(self, value):
        return value


def test_call_and_process_commands_success():
    rt = TestableRuntime()

    # Enqueue a call and then process it (normally done by run_loop)
    reply_q = __import__("queue").Queue(maxsize=1)
    rt._cmd_q.put(Command(name="echo", args=("hi",), kwargs={}, reply_q=reply_q))

    rt._process_commands(max_per_cycle=10)
    assert reply_q.get(timeout=0.1) == "hi"


def test_call_unknown_command_returns_exception_object():
    rt = TestableRuntime()

    reply_q = __import__("queue").Queue(maxsize=1)
    rt._cmd_q.put(Command(name="does_not_exist", args=(), kwargs={}, reply_q=reply_q))

    rt._process_commands(max_per_cycle=10)
    res = reply_q.get(timeout=0.1)
    assert isinstance(res, RuntimeError)


def test_runtime_call_timeout():
    rt = TestableRuntime()

    # Do not process queue -> should timeout
    with pytest.raises(TimeoutError):
        rt.call("echo", 0.01, "x")
