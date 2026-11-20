"""Several cartesian robots in one dataset (dataset.cartesian_robots): prefixed pose keys, per-robot order and
DART labels, prismatic joint units, and a joint name two robots share."""

import logging
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

from guide_core.scene.extra_features import POSE, cartesian_robots, key_info, listed, pose_keys, pose_names
from guide_core.scene.scene_recorder import SceneRecorder

sys.path.insert(0, str(Path(__file__).parent))  # this folder's test modules, for the Isaac stubs
from test_command_fixes import isaac_import  # noqa: E402, F401  (fixture)


TWO = ["psm1", "psm2"]


def test_pose_keys_are_bare_for_one_robot_and_prefixed_for_several():
    assert pose_keys(None) == POSE and pose_keys("psm1") == [f"psm1.{k}" for k in POSE]
    assert pose_names(["franka"]) == {"franka": POSE}
    assert pose_names(TWO)["psm2"][0] == "psm2.x"
    assert cartesian_robots({"cartesian_velocity_robot": "franka"}) == ["franka"]
    assert cartesian_robots({"cartesian_robots": TWO}) == TWO and cartesian_robots({}) == []
    assert listed({"observations": [{"psm1.vz": "vz"}]}, ["vz"]) == ["vz"]


def test_key_info_of_prefixed_and_prismatic_keys():
    assert key_info("psm1.x") == ("m", "base frame") and key_info("psm2.rz", action=True)[0] == "rad"
    assert key_info("psm1.insertion.pos") == ("rad", "")
    assert [key_info(f"psm1.insertion.{a}", prismatic={f"psm1.insertion.{a}"})[0] for a in ("pos", "vel", "eff")] == ["m", "m/s", "N"]


def _view(dofs, value=0.5):
    return SimpleNamespace(dof_names=dofs, handles_initialized=True, get_joint_positions=lambda joint_indices: np.array([value] * len(joint_indices)),
                           get_applied_action=lambda: None)


def test_record_step_writes_each_robot_in_its_own_base_frame_in_robot_order(isaac_import):  # noqa: F811
    orchestrator = isaac_import("guide_core.scene.scene_orchestrator").SceneOrchestrator
    ident = np.array([[1.0, 0.0, 0.0, 0.0]])
    pose = lambda p: SimpleNamespace(get_world_poses=lambda: (np.array([p]), ident))  # noqa: E731
    host = lambda x: SimpleNamespace(clean_target=[x, 0, 0], clean_rotation=np.eye(3))  # noqa: E731
    obs_masks = {r: {"pos": {"indices": [0], "keys": [f"{r}.j.pos"]}} for r in TWO}
    scene = SimpleNamespace(
        _logger=MagicMock(), _config={"dataset": {"cartesian_robots": TWO}},
        ee_views={"psm1": pose([1.1, 0, 0]), "psm2": pose([2.2, 0, 0])},
        base_views={"psm1": pose([1.0, 0, 0]), "psm2": pose([2.0, 0, 0])},
        robots_views={r: _view(["j"]) for r in TWO}, obs_masks=obs_masks, act_masks={},
        motion_hosts={"psm1": host(0.3), "psm2": host(0.4)}, prompts={"task": "", "subtask": ""}, task="",
    )
    scene._recorded_annotators = dict

    frame = orchestrator.record_step(scene, 0)

    assert list(frame["observation"]) == ["psm1.j.pos", *pose_keys("psm1"), "psm2.j.pos", *pose_keys("psm2")]
    assert list(frame["action"]) == [*pose_keys("psm1"), *pose_keys("psm2")]
    assert frame["observation"]["psm1.x"] == pytest.approx(0.1) and frame["observation"]["psm2.x"] == pytest.approx(0.2)
    assert frame["dart"]["psm2"]["clean"][0] == 0.4 and frame["dart"]["psm1"]["measured"][0] == pytest.approx(0.1)


def test_a_joint_two_robots_have_must_name_its_robot(isaac_import, monkeypatch):  # noqa: F811
    module = isaac_import("guide_core.scene.scene_orchestrator")
    views = {"psm1": _view(["psm_yaw_joint"]), "psm2": _view(["psm_yaw_joint"])}
    monkeypatch.setattr(sys.modules["isaacsim.core.prims"], "SingleArticulation",
                        lambda prim_path, name: views[name.split("_view")[0]])
    obs = [{"psm1.yaw.pos": "psm1.psm_yaw_joint.pos"}, {"psm2.yaw.pos": "psm2.psm_yaw_joint.pos"}, {"yaw.pos": "psm.psm_yaw_joint.pos"}]
    scene = SimpleNamespace(_logger=MagicMock(), _scene_id=0, _named_xform=lambda *a: None,
                            _config={"robots": {r: {"path": f"/{r}"} for r in TWO}, "dataset": {"observations": obs}})

    with pytest.raises(ValueError, match=r"yaw.pos: joint psm_yaw_joint is in \['psm1', 'psm2'\]; name the robot"):
        module.SceneOrchestrator.setup_dataset(scene)


def test_the_recorder_writes_two_robots_and_their_dart_labels(tmp_path):
    lerobot_dataset = pytest.importorskip("lerobot.datasets.lerobot_dataset")
    recorder = SceneRecorder("pkg", "two", {"dataset": {"fps": 10, "cartesian_robots": TWO}})
    recorder._logger = logging.getLogger("test_multi_robot_features")
    recorder.LeRobotDataset = lerobot_dataset.LeRobotDataset
    recorder.set_output_path(str(tmp_path))
    recorder.set_prismatic(["psm1.ins.pos"])
    zero = {k: 0.0 for r in TWO for k in ["%s.ins.pos" % r, *pose_keys(r)]}
    start = None
    for i in range(3):  # psm1's clean target runs 1 cm ahead in x, psm2's 2 cm in y
        dart = {"psm1": {"clean": [0.01 * (i + 1), 0, 0, 0, 0, 0], "measured": [0.01 * i, 0, 0, 0, 0, 0]},
                "psm2": {"clean": [0, 0.02 * (i + 1), 0, 0, 0, 0], "measured": [0, 0.02 * i, 0, 0, 0, 0]}}
        start = recorder._process_frame({"timestamp": i, "observation": dict(zero), "action": dict(zero), "task": "t", "dart": dart}, start)
    recorder._finalize_episode()
    root = recorder.dataset.root
    names = recorder.dataset.features["action"]["names"]
    recorder._finalize_dataset()

    ds = lerobot_dataset.LeRobotDataset("pkg", root=root)
    assert ds.features["observation.state"]["names"] == list(zero) and "x" not in names
    assert ds.features["observation.state"]["info"]["unit"]["psm1.ins.pos"] == "m"
    assert ds.features["observation.state"]["info"]["unit"]["psm2.ins.pos"] == "rad"
    actions = [dict(zip(names, (round(float(v), 6) for v in ds[i]["action"]))) for i in range(len(ds))]
    # c[t+1] - s[t] per robot; the last frame takes its own target
    assert [(a["psm1.x"], a["psm2.y"]) for a in actions] == [(0.02, 0.04), (0.02, 0.04), (0.01, 0.02)]
    assert {(a["psm1.y"], a["psm2.x"]) for a in actions} == {(0.0, 0.0)}
