# flake8: noqa: E402
"""The recorder samples on the physics clock, and only renders the frames it samples.

``current_time_step_index`` counts PHYSICS steps, not rendered frames, so a record
interval measured against ``step_freq`` is wrong by the substep count -- which is how
``f_sim = 120`` came to be hard-coded next to a ``step_freq: 60`` config. Deriving it
from ``get_physics_dt()`` keeps the capture rate fixed at ``record_frequency`` whatever
``physics_freq`` is set to.

The second thing under test is the render-product gate: a product has to be switched on
one rendered frame BEFORE the capture reads it, because the ordering inside an
``app.update()`` is pre-step callbacks -> physics -> render.
"""

import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

# SceneManager.step() imports omni.replicator.core so it is off the per-tick path;
# importing it for real bootstraps the Kit kernel. isaacsim itself imports fine, so
# leave it alone -- stubbing it leaks a MagicMock SimulationApp into every later test.
REPLICATOR = ("omni", "omni.replicator", "omni.replicator.core")


@pytest.fixture
def SceneManager():
    saved = {name: sys.modules.get(name) for name in REPLICATOR}
    for name in REPLICATOR:
        sys.modules[name] = MagicMock()

    from guide_core.scene.scene_manager import SceneManager as _SceneManager

    yield _SceneManager

    for name, module in saved.items():
        if module is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = module


class FakeRecorder:
    def __init__(self):
        self.frames = []
        self.idle_polls = 0

    def put_record_data(self, data):
        self.frames.append(data)

    def is_idle(self):
        self.idle_polls += 1
        return False

    def set_start_recording(self):
        pass


class FakeScene:
    """Just enough scene for the RECORDING branch of step_task."""

    def __init__(self, record_frequency=10):
        from guide_core.types.scene_state import SceneState

        self.state = SceneState.RECORDING
        self.record_frequency = record_frequency
        self.recorder = FakeRecorder()
        self.render_enabled = True
        self.render_log = []
        self._config = {}

    def set_render_products_enabled(self, enabled):
        self.render_enabled = enabled
        self.render_log.append(enabled)

    def record_step(self, current_step):
        assert self.render_enabled, (
            f"step {current_step} captured with the render products switched off -- "
            "the image would be whatever was last rendered, not this frame"
        )
        return {"timestamp": current_step}


def drive(SceneManager, scene, physics_hz, render_hz, ticks):
    """Run step_task over `ticks` physics steps and hand back the scene."""
    manager = SceneManager(logger=MagicMock())
    manager._scenes = [scene]
    manager._locks = {0: MagicMock()}

    world = SimpleNamespace(
        get_physics_dt=lambda: 1.0 / physics_hz,
        get_rendering_dt=lambda: 1.0 / render_hz,
        current_time_step_index=0,
    )
    step_task = manager.step(SimpleNamespace(_world=world))

    for tick in range(ticks):
        world.current_time_step_index = tick
        step_task(1.0 / physics_hz)
    return scene


@pytest.mark.parametrize("physics_hz", [60, 120, 240])
def test_capture_rate_is_record_frequency_whatever_the_physics_rate(SceneManager, physics_hz):
    # One second of simulated time at each physics rate must yield the same 10 frames.
    scene = drive(SceneManager, FakeScene(record_frequency=10), physics_hz, 60, physics_hz)

    assert len(scene.recorder.frames) == 10


def test_the_old_hard_coded_120_would_have_halved_the_rate(SceneManager):
    # The regression this guards: f_sim pinned at 120 while physics runs at 60 gives
    # interval = 12 on a 60 Hz clock, i.e. 5 Hz instead of the configured 10 Hz.
    scene = drive(SceneManager, FakeScene(record_frequency=10), 60, 60, 60)

    assert len(scene.recorder.frames) == 10, "10 Hz configured"
    assert len(scene.recorder.frames) != 5, "5 Hz is what f_sim=120 on a 60 Hz clock gives"


def test_render_products_are_on_for_exactly_one_frame_per_capture(SceneManager):
    scene = drive(SceneManager, FakeScene(record_frequency=10), 120, 60, 120)

    # FakeScene.record_step asserts the products were live when it ran, so reaching
    # here means every capture saw a freshly rendered frame.
    assert len(scene.recorder.frames) == 10
    # Strictly alternating: one enable per capture, one disable per capture, never two
    # of a kind in a row. It opens with a disable because the capture on tick 0 lands
    # the instant RECORDING begins -- start_recording already switched the products on
    # for warmup, which is the state FakeScene starts in.
    assert scene.render_log == [False, True] * 10
    assert scene.render_enabled is True, (
        "the last enable has no matching capture inside the window; it is the one the "
        "next capture will read"
    )


def test_the_gate_is_skipped_when_there_is_no_idle_frame_to_skip(SceneManager):
    # record_frequency == the render rate: every rendered frame is captured, so
    # toggling would switch the products off over the only frame that matters.
    scene = drive(SceneManager, FakeScene(record_frequency=60), 120, 60, 120)

    assert len(scene.recorder.frames) == 60
    assert scene.render_log == [], "nothing to gate at the full render rate"


def test_finalizing_polls_the_recorder_at_the_record_rate_not_every_tick(SceneManager):
    from guide_core.types.scene_state import SceneState

    scene = FakeScene(record_frequency=10)
    scene.state = SceneState.FINALIZING
    drive(SceneManager, scene, 120, 60, 120)

    # is_idle() is a blocking round trip to the recorder process; 10 polls a second,
    # not 120.
    assert scene.recorder.idle_polls == 10
