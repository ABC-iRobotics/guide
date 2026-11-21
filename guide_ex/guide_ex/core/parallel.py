"""ParallelExecuter: a UTILITY node that runs its branches at the same time, one thread each."""

import contextvars
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional

from guide_ex.core.base_node import BRANCH, BaseNode
from guide_ex.core.states import DemoStatus, ExecutionResult, Layer


class _Run:
    """One run of a ParallelExecuter: who claimed which robot, and each branch's prompts."""

    def __init__(self, n: int, context: dict, prefix: Dict[str, str], parent: Optional[tuple]):
        self.claims, self.sending = threading.Lock(), threading.Lock()  # sending holds no claim up
        self.owner = {}  # id(robot) -> branch index
        self.prompts = [{"task": "", "subtask": ""} for _ in range(n)]
        self.ready = [False] * n  # prompted or ended: no join goes out before every branch is, so no half prompt
        self.send = None
        self.context, self.prefix, self.parent = context, prefix, parent

    def claim(self, branch: int, robot) -> Optional[int]:
        """None once `robot` is `branch`'s, else the branch that holds it."""
        with self.claims:
            owner = self.owner.setdefault(id(robot), branch)
        return None if owner == branch else owner

    def _joined(self, level: str) -> str:
        """The branches' prompts of `level` as one: "<side>: <text>" for a branch that holds exactly one
        robot of the prefix map, in the map's order, then the others in declaration order."""
        keys = list(self.prefix)
        owned = {}  # branch -> the prefix keys whose robot it holds
        for key in keys:
            if key in self.context and id(self.context[key]) in self.owner:
                owned.setdefault(self.owner[id(self.context[key])], []).append(key)
        parts = []
        for branch, mine in enumerate(self.prompts):
            if mine[level]:
                side = owned.get(branch, [])
                if len(side) == 1:
                    parts.append((keys.index(side[0]), f"{self.prefix[side[0]]}: {mine[level]}"))
                else:
                    parts.append((len(keys), mine[level]))
        return " ".join(text for _, text in sorted(parts, key=lambda p: p[0]))  # stable: declaration order

    def prompt(self, branch: int, task: str, subtask: str, send) -> ExecutionResult:
        """Store `branch`'s prompts (empty leaves one as it is) and pass the joined prompts up: into the
        enclosing executer's slot, or at the top to send(task, subtask)."""
        with self.sending:  # one at a time, so the scene never gets an older join after a newer one
            mine = self.prompts[branch]
            mine["task"], mine["subtask"] = task or mine["task"], subtask or mine["subtask"]
            self.send = send
            return self._flush(branch)

    def done(self, branch: int) -> None:
        """`branch` ended: a join held back for it goes out now."""
        with self.sending:
            if not self.ready[branch]:
                self._flush(branch)

    def _flush(self, branch: int) -> ExecutionResult:
        """Mark `branch` ready; once all are, pass the joined prompts up (the caller holds `sending`).
        ponytail: a branch that never prompts holds its siblings' prompts until it ends."""
        self.ready[branch] = True
        if self.send is None or not all(self.ready):
            return ExecutionResult(DemoStatus.PERFECT)
        task, subtask = self._joined("task"), self._joined("subtask")
        if self.parent is None:
            return self.send(task, subtask)
        run, outer = self.parent
        return run.prompt(outer, task, subtask, self.send)


def prompt(task: str, subtask: str, send) -> ExecutionResult:
    """SetPrompt's way out: send(task, subtask) outside a ParallelExecuter, else into its innermost
    branch's slot, joined with the sibling branches' prompts."""
    frames = BRANCH.get()
    if not frames:
        return send(task, subtask)
    run, branch = frames[-1]
    return run.prompt(branch, task, subtask, send)


class ParallelExecuter(BaseNode):
    """Runs its branches at the same time, each in its own thread on its own shallow copy of the context
    (robots shared), and waits for all. PERFECT only if every branch succeeds, else FAILURE naming the
    failed ones. Merge: the keys a branch added or changed (a new object) go back to the parent; on a
    clash the first branch in declaration order wins, with a warning. A robot belongs to the first branch
    whose leaf takes it; another branch's leaf taking it fails before it runs. `prefix` maps context
    keys of robots to the image side that starts their branch's prompt (e.g. {psm2: Right, psm1: Left})."""

    level = Layer.UTILITY
    claims_robots = False  # its branches' leaves claim

    def __init__(self, name: str, branches: List[BaseNode], prefix: Optional[Dict[str, str]] = None, **kwargs):
        super().__init__(name, **kwargs)
        self.branches = branches
        self.prefix = prefix or {}

    def run(self, **context) -> ExecutionResult:
        self.logger.info(f"\n[STARTING] PARALLEL: {self.name} ({len(self.branches)} branches)")
        frames = BRANCH.get()
        run = _Run(len(self.branches), context, self.prefix, frames[-1] if frames else None)

        def branch_run(i: int, branch: BaseNode) -> ExecutionResult:
            BRANCH.set((*frames, (run, i)))
            try:
                return branch.execute(dict(context))
            except Exception as e:  # one branch's error must not hide the others' results
                self.logger.exception(f"[{self.name}] branch {i} ({branch.name}) raised")
                return ExecutionResult(DemoStatus.FAILURE, error_message=repr(e))
            finally:
                run.done(i)

        # copy_context per branch: each thread starts from this one's contextvars, so nested executers see their parents
        with ThreadPoolExecutor(len(self.branches), thread_name_prefix=self.name) as pool:
            futures = [pool.submit(contextvars.copy_context().run, branch_run, i, b)
                       for i, b in enumerate(self.branches)]
        results = [f.result() for f in futures]

        merged, source = {}, {}
        for i, result in enumerate(results):
            for key, value in result.outputs.items():
                if key in context and context[key] is value:
                    continue  # unchanged
                if key in merged:
                    if merged[key] is not value:
                        self.logger.warning(f"[{self.name}] branches {source[key]} and {i} both set '{key}': "
                                            f"branch {source[key]}'s value kept")
                    continue
                merged[key], source[key] = value, i

        failed = [f"{i} ({b.name}): {r.error_message}" for i, (b, r) in enumerate(zip(self.branches, results))
                  if r.status == DemoStatus.FAILURE]
        if failed:
            message = f"[{self.name}] parallel branches failed: " + "; ".join(failed)
            self.logger.error(message)
            return ExecutionResult(DemoStatus.FAILURE, error_message=message, outputs=merged)
        return ExecutionResult(DemoStatus.PERFECT, outputs=merged)
