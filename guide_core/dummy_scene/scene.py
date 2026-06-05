from guide_core.scene.scene_orchestrator import SceneOrchestrator


class Scene(SceneOrchestrator):
    def check_warmup(self):
        pass

    def record_step(self, step):
        pass

    def is_success_preprocess(self, instructions):
        return []

    def is_success_postprocess(self, result):
        return True

    def randomize_preprocess(self, instructions):
        return []

    def randomize_postprocess(self, result):
        return True

    def reset_preprocess(self, instructions):
        return []

    def reset_postprocess(self, result):
        return True

    def reset_lightweight(self):
        pass
