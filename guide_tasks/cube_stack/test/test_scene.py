from cube_stack.scene import Scene, separated, subtask_prompts

from guide_core.scene.scene_orchestrator import SceneOrchestrator
from guide_core.types.geometry import Point, Pose

CRAMPED = [(0.0, 0.0, 0.025), (0.05, 0.0, 0.025), (0.3, 0.0, 0.025), (0.0, 0.3, 0.025)]
APART = [(0.0, 0.0, 0.025), (0.15, 0.0, 0.025), (0.3, 0.0, 0.025), (0.0, 0.3, 0.025)]


def test_one_prompt_per_step_bottom_up():
    assert subtask_prompts(["blue", "red", "green"]) == [
        "Put the red cube on the blue cube.",
        "Put the green cube on the red cube.",
    ]


def test_separation_is_measured_in_the_table_plane():
    assert separated(APART, 0.12)
    assert not separated(CRAMPED, 0.12)
    assert separated([(0, 0, 0), (0, 0, 1)], 0.0)


def scene_drawing(monkeypatch, layouts):
    layouts = iter(layouts)

    def randomize(self, *, seed=None, inject=None, zone=None):
        self.draws += 1
        self.layout = next(layouts)
        return self.draws

    monkeypatch.setattr(SceneOrchestrator, "randomize", randomize)
    monkeypatch.setattr(Scene, "_cube_positions", lambda self: self.layout)
    scene = Scene.__new__(Scene)
    scene.draws = 0
    return scene


def test_a_cramped_layout_is_redrawn(monkeypatch):
    scene = scene_drawing(monkeypatch, [CRAMPED, CRAMPED, APART])

    assert scene.randomize() == 3


def test_a_seeded_layout_is_taken_as_drawn(monkeypatch):
    scene = scene_drawing(monkeypatch, [CRAMPED, APART])

    assert scene.randomize(seed=7) == 1


def test_the_starting_cube_is_the_zone_target():
    scene = Scene.__new__(Scene)
    scene._scene_id, scene.order = 2, ["green", "red", "blue", "yellow"]

    assert scene.zone_target() == "/Scene_2/blocks/green_block"


def test_success_is_one_tower_in_the_drawn_order():
    scene = Scene.__new__(Scene)
    tower = [Pose(position=Point([0.1, 0.2, 0.025 + 0.05 * i])) for i in range(4)]
    slid = tower[:3] + [Pose(position=Point([0.14, 0.2, 0.175]))]

    assert scene.is_success_postprocess(tower)
    assert not scene.is_success_postprocess(slid)
    assert not scene.is_success_postprocess(tower[::-1])
