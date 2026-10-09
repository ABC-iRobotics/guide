# flake8: noqa: E402
"""Two clocks, and only one of them can select a single rendered frame.

The recorder samples on the PHYSICS clock: ``current_time_step_index`` counts physics
steps, not rendered frames, so a record interval measured against ``step_freq`` is wrong
by the substep count -- which is how ``f_sim = 120`` came to be hard-coded next to a
``step_freq: 60`` config.

Gating the render products, though, has to happen on the RENDER clock. Expressing "one
rendered frame" as a run of ``substeps`` consecutive physics indices only works while
the two clocks stay in phase, and they do not: a reset, a randomization, or any extra
``simulation_app.update()`` slips the phase, and a window that then straddles two
updates renders -- and publishes -- twice per interval. That is what ``gate_render``
exists for, and what these tests pin down.
"""

import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

# SceneManager.step() imports omni.replicator.core, so it is off the per-tick path and
# has to be stubbed. Stubbing `omni` then breaks the real `isaacsim`, whose bootstrap
# needs omni.kit -- and it fails by raising SystemExit, which runtime.py's
# `except Exception` around `from isaacsim import SimulationApp` does not catch. So
# isaacsim is stubbed too, and everything is restored on teardown: a MagicMock
# SimulationApp left in sys.modules would follow every later test in the session.
# (Without the restore this file passed only when something alphabetically earlier had
# already stubbed isaacsim, and failed when run on its own.)
STUBBED = ("omni", "omni.replicator", "omni.replicator.core", "isaacsim")


@pytest.fixture
def SceneManager():
    saved = {name: sys.modules.get(name) for name in STUBBED}
    for name in STUBBED:
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
    """Just enough scene for step_task and gate_render."""

    def __init__(self, record_frequency=10):
        from guide_core.types.scene_state import SceneState

        self.state = SceneState.RECORDING
        self.record_frequency = record_frequency
        self.recorder = FakeRecorder()
        self.prompts = {"task": "", "subtask": ""}
        self._config = {}
        # The real setter early-returns when nothing changed (gate_render calls it once
        # per rendered frame), so mirror that and the log becomes a list of transitions.
        self.render_enabled = None
        self.render_log = []

    def set_render_products_enabled(self, enabled):
        if self.render_enabled == enabled:
            return
        self.render_enabled = enabled
        self.render_log.append(enabled)

    def record_step(self, current_step):
        return {"timestamp": current_step}


def drive(SceneManager, scene, physics_hz, render_hz, ticks, first_tick=0):
    """Run step_task over `ticks` physics steps. `first_tick` offsets the phase."""
    manager = SceneManager(logger=MagicMock())
    manager._scenes = [scene]
    manager._locks = {0: MagicMock()}

    world = SimpleNamespace(
        get_physics_dt=lambda: 1.0 / physics_hz,
        get_rendering_dt=lambda: 1.0 / render_hz,
        current_time_step_index=first_tick,
    )
    step_task = manager.step(SimpleNamespace(_world=world))

    for tick in range(first_tick, first_tick + ticks):
        world.current_time_step_index = tick
        step_task(1.0 / physics_hz)
    return scene


def render(SceneManager, scene, step_hz, frames, first_frame=0):
    """Run gate_render over `frames` rendered frames and hand back the scene."""
    manager = SceneManager(logger=MagicMock())
    manager._scenes = [scene]
    manager._locks = {0: MagicMock()}

    for frame in range(first_frame, first_frame + frames):
        manager.gate_render(frame, step_hz)
    return scene


# ~~~~~~~~~~~~~~~~~~~~~~~ the physics clock: what gets captured ~~~~~~~~~~~~~~~~~~~~~~~


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


def test_the_physics_callback_no_longer_touches_the_render_switches(SceneManager):
    # It used to open the window itself, on physics-step parity. That is the bug this
    # whole split exists to fix -- one owner, and it is gate_render.
    scene = drive(SceneManager, FakeScene(record_frequency=10), 120, 60, 120)

    assert scene.render_log == []


def test_finalizing_polls_the_recorder_at_the_record_rate_not_every_tick(SceneManager):
    from guide_core.types.scene_state import SceneState

    scene = FakeScene(record_frequency=10)
    scene.state = SceneState.FINALIZING
    drive(SceneManager, scene, 120, 60, 120)

    # is_idle() is a blocking round trip to the recorder process; 10 polls a second,
    # not 120.
    assert scene.recorder.idle_polls == 10


# ~~~~~~~~~~~~~~~~~~~~~~~~ the render clock: what gets rendered ~~~~~~~~~~~~~~~~~~~~~~~


def test_one_render_per_interval_at_the_configured_rate(SceneManager):
    # 60 rendered frames is one second at step_freq 60; ten of them render.
    scene = render(SceneManager, FakeScene(record_frequency=10), 60, 60)

    assert scene.render_log.count(True) == 10


def test_the_window_is_one_frame_wide_and_strictly_alternates(SceneManager):
    scene = render(SceneManager, FakeScene(record_frequency=10), 60, 18)

    # (frame + 1) % 6 == 0 -> frames 5, 11, 17. Off before each, on for exactly one.
    assert scene.render_log == [False, True, False, True, False, True]


def test_the_window_closes_before_the_capture_reads_it(SceneManager):
    # The frame has to be rendered BEFORE the physics tick that captures it: the
    # recorder reads whatever the annotator last received. Frame 5 renders, the capture
    # at physics step 12 (== frame 6) reads it.
    scene = render(SceneManager, FakeScene(record_frequency=10), 60, 6)

    assert scene.render_log == [False, True], "on for frame 5, the one before the capture"


def test_the_rate_does_not_depend_on_the_physics_phase(SceneManager):
    # The reported bug: a window expressed in physics-step indices was one rendered
    # frame only while the two clocks stayed in phase. Off phase it straddled two
    # updates and published at 20 Hz instead of 10. gate_render never consults the
    # physics counter, so an offset changes nothing.
    aligned = render(SceneManager, FakeScene(record_frequency=10), 60, 60, first_frame=0)
    offset = render(SceneManager, FakeScene(record_frequency=10), 60, 60, first_frame=1)

    assert aligned.render_log.count(True) == 10
    assert offset.render_log.count(True) == 10


def test_every_frame_renders_when_the_rate_matches_the_render_rate(SceneManager):
    scene = render(SceneManager, FakeScene(record_frequency=60), 60, 60)

    assert scene.render_log == [True], "on once, and never switched off again"


def test_stopping_holds_the_render_products_open(SceneManager):
    # gate_render only runs while the loop is RUNNING, so whatever it last wrote sticks
    # for the whole reset/randomize/home chain -- five frames in six that is "off", and
    # a camera topic that goes quiet across a scene change looks to a policy evaluation
    # exactly like a simulator that died. The loop overrides while stopped.
    manager = SceneManager(logger=MagicMock())
    scene = FakeScene(record_frequency=10)
    manager._scenes = [scene]
    manager._locks = {0: MagicMock()}

    manager.gate_render(0, 60)  # frame 0 of 6: closed
    assert scene.render_enabled is False

    manager.gate_render(0, 60, enabled=True)
    assert scene.render_enabled is True
