"""A requested seed has to survive the trip from the service to the RNG.

The whole point of ``Randomize.use_seed`` is that two evaluations see identical
layouts and can be compared as paired samples. If the seed is dropped anywhere along
``Randomize.srv -> guide_ros -> GUIDESimulator -> _cmd_randomize_scene -> scene``,
nothing fails and nothing logs -- the scenes are simply different, the pairing is
silently gone, and the sweep that relied on it reports a difference between two
checkpoints that is really a difference between two sets of scenes.

The other half of the same trap is ``use_seed=False`` arriving as seed 0 rather than
as "no seed": every episode would then draw the SAME layout, and a policy would be
scored twenty times on one scene.
"""

import sys
from unittest.mock import MagicMock

import pytest

for name in ("isaacsim", "isaacsim.core", "isaacsim.core.api", "omni", "omni.usd"):
    sys.modules.setdefault(name, MagicMock())

from guide_core.core.guide_simulator import GUIDESimulator  # noqa: E402
from guide_core.types.randomization import SeedTree  # noqa: E402


class FakeRuntime:
    def __init__(self):
        self.calls = []

    def call(self, name, timeout=None, *args, **kwargs):
        self.calls.append((name, kwargs))
        return True


@pytest.fixture
def simulator():
    sim = GUIDESimulator(sim_id=0)
    sim._runtime = FakeRuntime()
    return sim


def test_a_requested_seed_reaches_the_runtime(simulator):
    simulator.randomize_scene(0, seed=42)

    assert simulator._runtime.calls[0][1]["seed"] == 42


def test_no_seed_is_a_fresh_draw_not_seed_zero(simulator):
    # None keeps the scene's own (scene_id, episode_index) stream. A 0 here would pin
    # every episode of every run to one layout.
    simulator.randomize_scene(0)

    assert simulator._runtime.calls[0][1]["seed"] is None


def test_the_zone_still_goes_through_alongside_it(simulator):
    simulator.randomize_scene(3, use_zone=True, zone=8, seed=1)

    _, kwargs = simulator._runtime.calls[0]
    assert (kwargs["scene_id"], kwargs["use_zone"], kwargs["zone"], kwargs["seed"]) == (
        3,
        True,
        8,
        1,
    )


def test_the_same_seed_draws_the_same_scene():
    tree = SeedTree.create(master=7)

    first, _ = tree.generator(11)
    second, _ = tree.generator(11)

    assert first.random(5).tolist() == second.random(5).tolist()


def test_different_seeds_draw_different_scenes():
    tree = SeedTree.create(master=7)

    assert tree.generator(11)[0].random() != tree.generator(12)[0].random()


def test_pairing_does_not_survive_a_new_master_seed():
    # Worth knowing before trusting a comparison across simulator restarts: the master
    # is drawn from OS entropy at scene registration unless it was injected, so equal
    # seeds only reproduce a layout WITHIN one session.
    assert SeedTree.create(master=1).generator(5)[0].random() != (
        SeedTree.create(master=2).generator(5)[0].random()
    )
