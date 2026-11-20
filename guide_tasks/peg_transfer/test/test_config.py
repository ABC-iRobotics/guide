"""config/init.yaml against the scene USD and guide_motion's robot profiles."""
from pathlib import Path

import pytest
import yaml

PKG = Path(__file__).resolve().parents[1]
INIT = yaml.safe_load((PKG / "config/init.yaml").read_text())
USD = PKG / "assets/peg_transfer.usd"


def test_every_joint_key_names_its_robot():
    """psm1 and psm2 share joint names: a bare joint would be ambiguous (the recorder raises)."""
    for section in ("observations", "action"):
        for item in INIT["dataset"][section]:
            ((key, target),) = item.items()
            if "." in target:
                robot = target.split(".")[0]
                assert robot in INIT["robots"] and key.startswith(f"{robot}."), (section, key, target)


def test_paths_exist_in_the_scene_usd():
    """Scene-relative paths: the USD's defaultPrim /World is referenced at /Scene_<id>."""
    pytest.importorskip("pxr")
    if not USD.exists():
        pytest.skip(f"{USD} absent")
    from pxr import Usd

    stage = Usd.Stage.Open(str(USD))
    paths = [r["path"] for r in INIT["robots"].values()] + [c["path"] for c in INIT["cameras"].values()]
    paths += list(INIT["dataset"]["tracked_objects"].values())
    assert [p for p in paths if not stage.GetPrimAtPath(f"/World{p}").IsValid()] == []


def test_the_three_robots_resolve_with_6_6_4_joints():
    from guide_motion import description

    if not all((description.ROBOTS_DIR / r / f"{r}.xrdf").is_file() for r in ("psm", "ecm")):
        pytest.skip("guide_motion robots psm/ecm not there yet (plan T2)")
    from guide_core.scene import robot_profile

    robots = {name: robot_profile.resolve(entry) for name, entry in INIT["robots"].items()}
    assert {n: len(r["joint_names"]) for n, r in robots.items()} == {"psm1": 6, "psm2": 6, "ecm": 4}
    assert all(r["controller"] == "oscbf" and r["oscbf"]["robot"] == r["profile"] for r in robots.values())
