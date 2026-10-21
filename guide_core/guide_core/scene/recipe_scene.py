"""A Scene whose behaviour is all data, so a compiled task's scene.py is one line:

    from guide_core.scene.recipe_scene import RecipeScene as Scene

config/task.yaml gives the language, the zone target, the goal (handed to the solver in the Randomize
reply) and, optionally, the free draw's separation radii. config/success.yaml lists success predicates
(on, on_region, in_region, joint_above, joint_below; guide_core.scene.predicates), which must all hold.

The goal items name the skills' inputs: {on: [obj, support]}, {in: [obj, region]}, {on_region | on_zone (a table zone, pushed into): {obj, region}} and {open | close | turn_on | turn_off: {joint, body, value, direction[, inclusive]}}; the solver reads them
from the Randomize reply (guide_ex.recipe.solve.context). A task may list a joint item its success does not test
(the drawer to pull before putting something in it).
"""
import itertools
import json
from pathlib import Path

import numpy as np
import yaml

from guide_core.scene.predicates import PREDICATE_PATHS
from guide_core.scene.scene_orchestrator import SceneOrchestrator

TASK_KEYS = {"language", "target", "goal", "separation"}
GOAL_LISTS = {"on", "in"}  # [a, b]
GOAL_DICTS = {"on_region": {"obj", "region"}, "on_zone": {"obj", "region"}, "open": {"joint", "body", "value", "direction"}}  # required keys
GOAL_DICTS |= {k: GOAL_DICTS["open"] for k in ("close", "turn_on", "turn_off")}
MAX_LAYOUT_DRAWS = 200  # as cube_stack: a free draw is redrawn until the separation holds


def load_task(path) -> dict:
    """config/task.yaml, checked: {language: {task, subtasks}, target, goal, separation: {prim: radius}}."""
    doc = yaml.safe_load(Path(path).read_text()) or {}
    unknown = set(doc) - TASK_KEYS
    if unknown:
        raise ValueError(f"{path}: unknown keys {sorted(unknown)} (it takes {sorted(TASK_KEYS)})")
    language = doc.get("language") or {}
    if not isinstance(language.get("task"), str) or not language["task"]:
        raise ValueError(f"{path}: language.task must be the task's sentence")
    goal = [{"on" if k is True else k: v for k, v in item.items()} for item in doc.get("goal") or []]
    for item in goal:
        ((kind, value),) = item.items()
        if kind in GOAL_LISTS and isinstance(value, list) and len(value) == 2:
            continue
        if kind in GOAL_DICTS and isinstance(value, dict) and GOAL_DICTS[kind] <= set(value):
            continue
        raise ValueError(f"{path}: goal item {item}: one of {sorted(GOAL_LISTS)} as [a, b] or {sorted(GOAL_DICTS)} as a mapping with "
                         f"the keys {sorted(GOAL_DICTS.get(kind, ()))} (YAML reads a bare on as true: write 'on')")
    return {
        "language": {"task": language["task"], "subtasks": list(language.get("subtasks") or [])},
        "target": doc.get("target"),
        "goal": goal,
        "separation": {str(prim): float(radius) for prim, radius in (doc.get("separation") or {}).items()},
    }


def separated_by_radii(positions, radii) -> bool:
    """LIBERO's placement test in the table plane: every pair further apart than its two radii."""
    return all(np.hypot(*(np.asarray(p, float)[:2] - np.asarray(q, float)[:2])) > r + s
               for (p, r), (q, s) in itertools.combinations(zip(positions, radii), 2))


class RecipeScene(SceneOrchestrator):
    """task.yaml + success predicates; arrangements, robot profiles and the scene recipe come from
    SceneOrchestrator."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.task_config = load_task(Path(str(self._path)) / "config" / "task.yaml")
        commands = [instruction.get("cmd") for instruction in self.success_instructions]
        if not commands or set(commands) - set(PREDICATE_PATHS):
            raise ValueError(f"config/success.yaml must list success predicates only ({sorted(PREDICATE_PATHS)}); "
                             f"got {commands} (YAML reads a bare on as true: write 'on')")
        self.task = self.task_config["language"]["task"]
        if self.task_config["separation"] and "randomize" not in self.replicator_files:
            raise ValueError("separation needs a Replicator randomize.yaml")

    def randomize(self, *, seed=None, inject=None, zone=None, arrangement=None):
        """A free draw is redrawn until task.yaml's separation holds (cube_stack's loop); a seed, an
        injection or an arrangement is a fixed start and is taken as it is."""
        radii = self.task_config["separation"]
        for _ in range(MAX_LAYOUT_DRAWS):
            context = super().randomize(seed=seed, inject=inject, zone=zone, arrangement=arrangement)
            fixed = seed is not None or inject is not None or bool(arrangement)
            if not radii or fixed or separated_by_radii(self._positions(list(radii)), list(radii.values())):
                return context
        self._logger.warning(f"[RecipeScene] no layout with {radii} apart in {MAX_LAYOUT_DRAWS} draws; using the last")
        return context

    def _positions(self, prims) -> list:
        """Where ``prims`` (scene-relative) are now: Replicator has applied the draw."""
        from isaacsim.core.prims import XFormPrim

        return [np.asarray(XFormPrim(prim_paths_expr=f"/Scene_{self._scene_id}{prim}",
                                     reset_xform_properties=False).get_world_poses()[0])[0] for prim in prims]

    def zone_target(self):
        target = self.task_config["target"]
        return f"/Scene_{self._scene_id}{target}" if target else None

    def randomize_preprocess(self, randomizer):
        self.task = self.task_config["language"]["task"]
        return randomizer

    def randomize_postprocess(self, result):
        return json.dumps({"task": self.task, "target": self.task_config["target"],
                           "goal": self.task_config["goal"], "subtasks": self.task_config["language"]["subtasks"]})

    def is_success_preprocess(self, instructions):
        return instructions

    def is_success_postprocess(self, result):
        failed = [r["reason"] for r in result if not r["success"]]
        for reason in failed:
            self._logger.debug(f"[RecipeScene] scene {self._scene_id} not done: {reason}")
        return not failed

    def reset_preprocess(self, instructions):
        return instructions

    def reset_postprocess(self, result):
        return True  # the Reset service answers a bool; the commands raise on failure

    def check_warmup(self):
        return True
