import random

from guide_core.scene.scene_orchestrator import SceneOrchestrator


class Scene(SceneOrchestrator):
    colors = ["red", "yellow", "green", "blue"]
    sides = ["left", "right"]

    def reset_preprocess(self, instructions):
        return super().reset_preprocess(instructions)

    def reset_postprocess(self, result):
        return super().reset_postprocess(result)

    def randomize_preprocess(self, instructions):
        self.c = random.choice(self.colors)
        self.s = random.choice(self.sides)

        self.task: str = f"Put the {self.c} block in the {self.s} bin."
        print(f"Task: {self.task}")

        return instructions

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
        # Default implementation: warmup is always complete after the required frames
        return True

    def reset_lightweight(self):
        # To be implemented with actual scene reset logic
        pass
