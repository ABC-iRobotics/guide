from guide_core.types.randomization import Categorical

from guide_core.scene.scene_orchestrator import SceneOrchestrator


class Scene(SceneOrchestrator):
    colors = ["red", "yellow", "green", "blue"]
    sides = ["left", "right"]

    def reset_preprocess(self, instructions):
        return super().reset_preprocess(instructions)

    def reset_postprocess(self, result):
        return super().reset_postprocess(result)

    def randomize_preprocess(self, randomizer):
        # Seeded, captured discrete draws through the single Randomizer. Runs
        # BEFORE the pose draws, so self.c (the target colour) is known when
        # zone_target() is queried to place that block in the requested zone.
        self.c = randomizer.draw("color", Categorical(tuple(self.colors)))
        self.s = randomizer.draw("side", Categorical(tuple(self.sides)))

        self.task: str = f"Put the {self.c} block in the {self.s} bin."
        print(f"Task: {self.task}")

        return randomizer

    def zone_target(self):
        # The color-selected block is the one placed in the requested zone; the
        # other three blocks (from the same /blocks/* grid instruction) stay free.
        c = getattr(self, "c", None)
        return f"/Scene_{self._scene_id}/blocks/{c}_block" if c else None

    def randomize_postprocess(self, result):
        return f"{{ \"goal\": \"/bin_{0 if self.s == 'left' else 1}\", \"target\": \"/blocks/{self.c}_block\", \"task\": \"{self.task}\" }}"

    def is_success_preprocess(self, instructions):
        kwargs = instructions[0].get("kwargs", {})
        instructions[0].update(
            {
                "kwargs": {
                    "prim_path": "/".join(
                        kwargs.get("prim_path", "").split("/")[:-1] + [f"{self.c}_block"]
                    ),
                    "scope": "/".join(
                        kwargs.get("scope", "").split("/")[:-1]
                        + [f'bin_{0 if self.s == "left" else 1}']
                    ),
                    "tolerance": 0.05,
                }
            }
        )
        return instructions

    def is_success_postprocess(self, result: list):
        return all(bool(r) for r in result)

    def check_warmup(self):
        # No task-specific check: warmup is complete once SceneManager's warm-up frames
        # have passed (SceneOrchestrator.check_warmup is abstract).
        return True
