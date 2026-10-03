import itertools
import json

import numpy as np

from guide_core.scene.scene_orchestrator import SceneOrchestrator
from guide_core.types.randomization import Categorical
from guide_ex.utility.stacking import is_on_top

CUBE = 0.05  # edge of the BlocksWorld cubes, m
# Spawn distance between cube centres. The open gripper spans 8 cm plus two ~1.5 cm
# fingers, so a neighbour closer than ~9 cm is hit on the way down; 12 cm leaves margin.
MIN_SEPARATION = 0.12
STACKED = {"xy_tolerance": 0.02, "z_tolerance": 0.01}
# Layouts drawn before giving up on separation; ~20% of draws pass (all 50 failing: 1e-5).
MAX_LAYOUT_DRAWS = 50


def subtask_prompts(order):
    """One prompt per stacking step; `order` is the tower's colours, bottom first."""
    return [f"Put the {top} cube on the {base} cube." for base, top in zip(order, order[1:])]


def separated(positions, distance):
    """True if no two positions are closer than `distance` in the table plane."""
    return all(
        np.hypot(*(np.asarray(a)[:2] - np.asarray(b)[:2])) >= distance
        for a, b in itertools.combinations(positions, 2)
    )


class Scene(SceneOrchestrator):
    """Stack the cubes: every cube of the scene into one tower, in a random order.

    The order is a seeded draw over all 24 permutations, so any cube can be the base
    and every pair occurs in both directions; it lands in the dataset sidecar's drawn
    values and in the subtask prompts.
    """

    colors = ["red", "yellow", "green", "blue"]

    def randomize(self, *, seed=None, inject=None, zone=None):
        # Draw until no two cubes are too close to grasp. Each draw advances the episode
        # counter, i.e. takes the next seed, so the sequence is still reproducible from
        # the master seed. A fixed seed or injected values give the same layout every
        # time: take it as it is.
        for _ in range(MAX_LAYOUT_DRAWS):
            context = super().randomize(seed=seed, inject=inject, zone=zone)
            if (
                seed is not None
                or inject is not None
                or separated(self._cube_positions(), MIN_SEPARATION)
            ):
                break
        else:
            self._logger.warning(
                f"No layout with {MIN_SEPARATION} m between cubes; using the last."
            )
        return context

    def _cube_positions(self):
        cubes = next(
            i
            for i in self.randomize_instructions
            if i.get("_prim_pattern", "").endswith("/blocks/*")
        )
        return [pose.position.to_numpy() for pose in cubes["kwargs"]["pose"]]

    def randomize_preprocess(self, randomizer):
        self.order = list(
            randomizer.draw("order", Categorical(tuple(itertools.permutations(self.colors))))
        )
        self.task = "Stack the cubes."
        return randomizer

    def randomize_postprocess(self, result):
        return json.dumps(
            {
                "task": self.task,
                "order": [f"/blocks/{c}_block" for c in self.order],
                "subtasks": subtask_prompts(self.order),
            }
        )

    def is_success_preprocess(self, instructions):
        template = instructions[0]
        return [
            {**template, "kwargs": {"prim_path": f"/Scene_{self._scene_id}/blocks/{c}_block"}}
            for c in self.order
        ]

    def is_success_postprocess(self, result):
        return all(
            is_on_top(upper, lower, CUBE, **STACKED) for lower, upper in zip(result, result[1:])
        )

    def reset_preprocess(self, instructions):
        return instructions

    def reset_postprocess(self, result):
        return True  # the Reset service answers a bool; the commands raise on failure

    def check_warmup(self):
        return True

    def reset_lightweight(self):
        pass
