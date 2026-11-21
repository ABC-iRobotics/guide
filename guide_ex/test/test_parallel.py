"""ParallelExecuter: overlap, merge, failures, robot claims, levels, and joined prompts (fake robots)."""

import logging
import time
from types import SimpleNamespace

import pytest

from guide_ex.core.base_node import BaseNode
from guide_ex.core.composite_node import CompositeNode
from guide_ex.core.parallel import ParallelExecuter
from guide_ex.core.states import DemoStatus, ExecutionResult, Layer
from guide_ex.recipe import RecipeError, load
from guide_ex.utility.recording import SetPrompt
from guide_ex.utility.wait import WaitForSeconds


class Robot:
    """An arm: takes actions, answers the recorder's services into the shared `log`."""

    def __init__(self, name, log=None):
        self.name, self.sent, self.log = name, [], log if log is not None else []
        self.node = SimpleNamespace(create_client=lambda srv_type, name, callback_group: name)
        self._reentrant_callback_group = None

    def send_action(self, action):
        self.sent.append(action)

    def callService(self, client, request, message=None, timeout_sec=60.0):
        self.log.append((request.task, request.subtask))
        return SimpleNamespace(success=True, message="")

    def __repr__(self):
        return self.name


class Move(BaseNode):
    level = Layer.STEP

    def __init__(self, alias, **kwargs):
        super().__init__("Move", alias, **kwargs)

    def run(self, robot, seconds=0.0, out=None, fail=False):
        time.sleep(seconds)
        robot.send_action(self.name)
        if fail:
            return ExecutionResult(DemoStatus.FAILURE, error_message="blocked")
        return ExecutionResult(DemoStatus.PERFECT, outputs=out or {})


def arm(name, robot_key, *children, level=Layer.SEQUENCE):
    return CompositeNode(name, level, children=list(children), dynamic_map={"robot": robot_key})


def announce(alias, task="", subtask=""):
    return SetPrompt(alias=alias, dynamic_map={"robot": "robot"},
                     static_args={"sim_namespace": "/Sim_0", "scene_id": 0, "task": task, "subtask": subtask})


@pytest.fixture
def ctx():
    log = []
    return {"psm1": Robot("psm1", log), "psm2": Robot("psm2", log), "ecm": Robot("ecm", log), "log": log}


def both(branches, prefix=None):
    return ParallelExecuter("Both", branches, prefix=prefix,
                            dynamic_map={k: k for k in ("psm1", "psm2", "ecm")})


def test_branches_overlap_in_time(ctx):
    node = both([Move("A", dynamic_map={"robot": "psm1"}, static_args={"seconds": 0.4}),
                 Move("B", dynamic_map={"robot": "psm2"}, static_args={"seconds": 0.4})])

    start = time.monotonic()
    result = node.execute(ctx)

    assert result.status == DemoStatus.PERFECT
    assert time.monotonic() - start < 0.7  # sequential would take 0.8
    assert ctx["psm1"].sent == ["A"] and ctx["psm2"].sent == ["B"]


def test_only_changed_keys_merge_and_the_first_branch_wins_a_clash(ctx, caplog):
    keep = ctx["ecm"]
    node = both([Move("A", dynamic_map={"robot": "psm1"}, static_args={"seconds": 0.2, "out": {"x": 1, "k": "a"}}),
                 Move("B", dynamic_map={"robot": "psm2"}, static_args={"out": {"y": 2, "k": "b", "ecm": keep}})])

    with caplog.at_level(logging.WARNING):
        result = node.execute(ctx)

    assert result.outputs == {"x": 1, "y": 2, "k": "a"}  # B finished first, A still wins; ecm unchanged
    assert "both set 'k'" in caplog.text


def test_a_failed_branch_is_listed_and_the_others_run_to_their_end(ctx):
    node = both([Move("A", dynamic_map={"robot": "psm1"}, static_args={"fail": True}),
                 Move("B", dynamic_map={"robot": "psm2"}, static_args={"seconds": 0.2}),
                 Move("C", dynamic_map={"robot": "missing"})])

    result = node.execute(ctx)

    assert result.status == DemoStatus.FAILURE
    assert "0 (A): blocked" in result.error_message and "2 (C): KeyError" in result.error_message
    assert ctx["psm2"].sent == ["B"]


def test_a_robot_claimed_by_one_branch_is_refused_to_another_before_it_runs(ctx, caplog):
    node = both([arm("Left", "psm1", Move("A", dynamic_map={"robot": "robot"})),
                 arm("Late", "psm1", WaitForSeconds(alias="Wait", static_args={"seconds": 0.2}),
                     Move("B", dynamic_map={"robot": "robot"}))])

    result = node.execute(ctx)

    assert result.status == DemoStatus.FAILURE
    assert "1 (Late)" in result.error_message and "[B] robot psm1 belongs to parallel branch 0" in caplog.text
    assert ctx["psm1"].sent == ["A"]  # B never ran


def test_outside_an_executer_nothing_is_claimed(ctx):
    Move("A", dynamic_map={"robot": "psm1"}).execute(ctx)
    assert Move("B", dynamic_map={"robot": "psm1"}).execute(ctx).status == DemoStatus.PERFECT


def test_prompts_join_in_prefix_order_with_unprefixed_last(ctx):
    node = both([arm("Left", "psm1", announce("L", subtask="Move to the exchange point.")),
                 arm("Cam", "ecm", announce("C", subtask="Hold the view.")),
                 arm("Right", "psm2", Move("Wait", dynamic_map={"robot": "robot"}, static_args={"seconds": 0.1}),
                     announce("R", subtask="Turn the block level."))],
                prefix={"psm2": "Right", "psm1": "Left"})

    assert node.execute(ctx).status == DemoStatus.PERFECT
    assert ctx["log"][-1] == ("", "Right: Turn the block level. Left: Move to the exchange point. Hold the view.")


def test_a_nested_executer_writes_its_join_into_the_parent_slot(ctx):
    prefix = {"psm2": "Right", "psm1": "Left"}
    inner = both([arm("Left", "psm1", announce("L", task="Hand the peg over.", subtask="Reach.")),
                  arm("Right", "psm2", announce("R", subtask="Grasp."))], prefix)
    outer = both([arm("Cam", "ecm", Move("Wait", dynamic_map={"robot": "robot"}, static_args={"seconds": 0.1}),
                      announce("C", subtask="Hold the view.")), inner], prefix)

    assert outer.execute(ctx).status == DemoStatus.PERFECT
    # the inner branch holds both arms: unprefixed at the outer level, so declaration order
    assert ctx["log"][-1] == ("Left: Hand the peg over.", "Hold the view. Right: Grasp. Left: Reach.")


def test_outside_an_executer_set_prompt_sends_as_given(ctx):
    announce("A", subtask="Pick.").execute({**ctx, "robot": ctx["psm1"]})
    assert ctx["log"] == [("", "Pick.")]


def wait(alias):
    return {"ref": "WaitForSeconds", "alias": alias, "static_args": {"seconds": 0.0}}


def comp(name, level, *children):
    return {"node_type": "CompositeNode", "name": name, "level": level, "children": list(children)}


def par(*branches):
    return {"node_type": "ParallelExecuter", "name": "Both", "prefix": {"psm2": "Right"}, "branches": list(branches)}


def test_the_loader_builds_an_executer_ranked_at_its_parent_level():
    tree = load(comp("Task", "TASK", par(comp("A", "SUBTASK", wait("a")), comp("B", "SUBTASK", wait("b")))))

    node = tree.children[0]
    assert isinstance(node, ParallelExecuter) and node.prefix == {"psm2": "Right"}
    assert [b.name for b in node.branches] == ["A", "B"] and isinstance(node.branches[0].children[0], WaitForSeconds)


def test_a_branch_not_below_the_executer_parent_is_refused():
    with pytest.raises(RecipeError, match="SUBTASK"):
        load(comp("Seq", "SEQUENCE", par(comp("A", "SUBTASK", wait("a")))))
    with pytest.raises(RecipeError, match="SEQUENCE"):  # nested: still ranked at the composite's level
        load(comp("Seq", "SEQUENCE", par(par(comp("A", "SEQUENCE", wait("a"))))))


def test_a_root_executer_is_unconstrained():
    assert isinstance(load(par(comp("A", "TASK", wait("a")), comp("B", "PROCEDURE", wait("b")))), ParallelExecuter)
