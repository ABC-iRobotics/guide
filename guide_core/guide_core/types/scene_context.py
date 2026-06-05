class SceneContext:
    def __init__(self, scene_id: str):
        self.scene_id = scene_id
        self.episode_id = None
        self.seed = None
        self.step_count = 0
        self.active = False
