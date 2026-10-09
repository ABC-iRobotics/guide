"""The Reset service answers a bool: SceneManager.reset_postprocess must always give one.

A list in the response's bool field makes rclpy abort the whole simulator.
"""
from guide_core.scene.scene_manager import SceneManager


class Scene:
    def __init__(self, hook):
        self.hook = hook

    def reset_postprocess(self, result):
        return self.hook(result)


def manager(hook):
    m = SceneManager()
    m._scenes = [Scene(hook)]
    return m


def not_implemented(result):
    raise NotImplementedError


def test_a_scene_without_the_hook_reports_success_not_the_raw_results():
    assert manager(not_implemented).reset_postprocess(0, [None, None]) is True


def test_a_scene_answer_becomes_a_bool():
    assert manager(lambda result: result).reset_postprocess(0, [None]) is True
    assert manager(lambda result: False).reset_postprocess(0, [None]) is False
