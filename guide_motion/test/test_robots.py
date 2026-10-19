"""The committed descriptions (Tasks 8-9) load, and every barrier holds at their home pose."""

import json

import numpy as np
import pytest

from guide_motion import description
from guide_motion.control import Controller
from guide_motion.model import Model


@pytest.mark.parametrize("robot", ["fr3", "panda"])
def test_committed_descriptions_are_safe_at_home(robot):
    model = Model.load(robot)
    meta = json.loads((model.robot_dir / "description.json").read_text())
    assert meta["pipeline_version"] == description.PIPELINE_VERSION and meta["checks"]["home_self_collision"] == "free"
    assert meta["checks"]["random_self_collision"] <= 0.08 and max(meta["checks"]["mass_matrix_rel"], meta["checks"]["gravity_rel"]) <= 1e-3
    assert model.tool_frame == f"{robot}_hand_tcp" and model.base_link == f"{robot}_link0"
    assert Controller(model, "velocity").barrier_values(model.home).min() > 0
    assert np.all(model.max_effort > 0) and np.all(model.max_velocity > 0)


def test_the_panda_override_references_isaacs_asset():
    text = (description.ROBOTS_DIR / "panda" / "panda.usda").read_text()
    assert "FrankaPanda/franka.usd" in text and "physics:diagonalInertia" in text


def test_the_fr3_masses_are_corrected():
    """Ruling 10, as PhysX ran the committed fr3.usda: 1 g frames, franka_description's hand and fingers."""
    meta = json.loads((description.ROBOTS_DIR / "fr3" / "description.json").read_text())
    physx = {name: kg for name, (kg, _) in meta["mass_check_kg"].items()}
    assert physx["fr3_link8"] == pytest.approx(0.001) and physx["fr3_hand_tcp"] == pytest.approx(0.001)
    assert physx["fr3_hand"] == pytest.approx(0.6544) and physx["fr3_leftfinger"] == physx["fr3_rightfinger"] == pytest.approx(0.0291)
    assert meta["mass_override_kg"]["fr3_link8"] == pytest.approx({"asset": 1.0, "override": 0.001})
    assert "fr3_link0" not in meta["mass_override_kg"]
    assert "FrankaFR3/fr3.usd" in (description.ROBOTS_DIR / "fr3" / "fr3.usda").read_text()
