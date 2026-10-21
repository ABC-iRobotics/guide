"""Named keys in observation.state / action (.vel, .eff, vx..wz, fx..tz), the recorder, and the wrench pipeline against Pinocchio."""

import logging

import numpy as np
import pinocchio as pin
import pytest

from guide_core.scene.extra_features import TWIST, WRENCH, check_layout, key_info, listed, tcp_wrench, with_info
from guide_core.scene.scene_recorder import SceneRecorder

OBS = [{"joint1.pos": "franka.fr3_joint1.pos"}, {"joint1.vel": "franka.fr3_joint1.vel"}, {"x": "x"}, {"vz": "vz"}, {"fx": "fx"}]
ACT = [{"joint1.pos": "joint_command.fr3_joint1.pos"}, {"gripper.eff": "franka.fr3_finger_joint1.eff"}, {"wz": "wz"}]


def test_listed_picks_the_keys_a_section_names_in_canonical_order():
    cfg = {"observations": OBS, "action": ACT}
    assert listed(cfg, TWIST) == ["vz", "wz"]  # wz from the action list: twist is shared
    assert listed(cfg, WRENCH) == ["fx"]
    assert listed({"observations": OBS[:3]}, TWIST + WRENCH) == []


def test_key_info_gives_units_and_frames():
    assert key_info("joint1.vel") == ("rad/s", "") and key_info("gripper.eff") == ("N", "") and key_info("gripper.pos")[0] == "m"
    assert key_info("joint3.eff")[0] == "N m" and key_info("rx") == ("rad", "base frame; rotation vector")
    assert key_info("x", action=True)[1].startswith("per-frame") and key_info("rx", action=True)[1].startswith("per-frame")
    assert key_info("vx") == ("m/s", key_info("wx")[1]) and key_info("wx")[0] == "rad/s" and "TCP" in key_info("wx")[1]
    assert key_info("fz")[0] == "N" and key_info("tz")[0] == "N m" and "environment applies" in key_info("tx")[1]
    info = with_info({"dtype": "float32", "shape": (3,), "names": ["x", "joint1.vel", "tz"]})["info"]
    assert info["unit"] == {"x": "m", "joint1.vel": "rad/s", "tz": "N m"} and set(info["frame"]) == {"x", "tz"}


def test_wrench_recovers_a_known_force_at_the_tcp():
    """Static FR3: Pinocchio's RNEA with a force at the TCP gives the force through the last joint (gravity
    and the hand's weight included); the pipeline must hand the TCP force back, in the TCP axes."""
    from guide_motion.model import Model

    model = Model.load("fr3")
    pm, data, tool = model.pin_model, model.pin_data, model.pin_model.getFrameId(model.tool_frame)
    rng = np.random.default_rng(1)
    for q in rng.uniform(model.lower, model.upper, size=(5, 7)):
        w = rng.uniform(-10, 10, 6)  # the environment's wrench at the TCP, world (= pin) axes
        pin.framesForwardKinematics(pm, data, q)
        T_tcp = data.oMf[tool].copy()
        on_joint = pm.frames[tool].placement.act(pin.Force(T_tcp.rotation.T @ w[:3], T_tcp.rotation.T @ w[3:]))
        fext = [pin.Force.Zero() for _ in range(pm.njoints)]
        fext[7] = on_joint
        pin.rnea(pm, data, q, np.zeros(7), np.zeros(7), fext)
        f7, oMi = data.f[7], data.oMi[7]
        body = pm.inertias[7]  # link 7, the hand and the locked fingers
        got = tcp_wrench(f7.linear, f7.angular, oMi.rotation, oMi.translation, T_tcp.translation, T_tcp.rotation,
                         body.mass, oMi.act(body.lever))
        np.testing.assert_allclose(got, np.r_[T_tcp.rotation.T @ w[:3], T_tcp.rotation.T @ w[3:]], atol=1e-9)


def test_free_space_reads_zero_after_the_tare():
    from guide_motion.model import Model

    model = Model.load("fr3")
    pm, data = model.pin_model, model.pin_model.createData()
    q = model.home
    pin.rnea(pm, data, q, np.zeros(7), np.zeros(7))
    pin.framesForwardKinematics(pm, data, q)
    oMi, body = data.oMi[7], pm.inertias[7]
    w = tcp_wrench(data.f[7].linear, data.f[7].angular, oMi.rotation, oMi.translation, data.oMf[pm.getFrameId(model.tool_frame)].translation,
                   data.oMf[pm.getFrameId(model.tool_frame)].rotation, body.mass, oMi.act(body.lever))
    np.testing.assert_allclose(w, 0, atol=1e-9)


def test_the_recorder_writes_the_named_keys_into_state_and_action(tmp_path):
    lerobot_dataset = pytest.importorskip("lerobot.datasets.lerobot_dataset")
    recorder = SceneRecorder("pkg", "extras", {"dataset": {"fps": 10}})
    recorder._logger = logging.getLogger("test_extra_features")
    recorder.LeRobotDataset = lerobot_dataset.LeRobotDataset
    recorder.set_output_path(str(tmp_path))
    pose = dict.fromkeys(["x", "y", "z"], 0.0)
    start = None
    for i in range(3):
        obs = {**pose, "rx": 0.0, "ry": 0.0, "rz": 0.0, "joint1.pos": 0.0, "joint1.vel": float(i), "vz": 2.0 * i, "fx": -float(i)}
        act = {**pose, "rx": 0.0, "ry": 0.0, "rz": 0.0, "joint1.pos": 0.0, "joint1.vel": float(i), "vz": 2.0 * i}
        start = recorder._process_frame({"timestamp": i, "observation": obs, "action": act, "task": "t"}, start)
    recorder._finalize_episode()
    root = recorder.dataset.root
    recorder._finalize_dataset()
    ds = lerobot_dataset.LeRobotDataset("pkg", root=root)
    names = ds.features["observation.state"]["names"]
    assert names == ["x", "y", "z", "rx", "ry", "rz", "joint1.pos", "joint1.vel", "vz", "fx"]
    assert ds.features["action"]["names"] == ["x", "y", "z", "rx", "ry", "rz", "joint1.pos", "joint1.vel", "vz"]
    assert set(ds.features).isdisjoint({"observation.velocity", "observation.tcp_twist", "observation.tcp_wrench"})
    assert ds.features["observation.state"]["info"]["unit"]["vz"] == "m/s"  # survives info.json
    assert ds.features["action"]["info"]["unit"]["rx"] == "rad"
    np.testing.assert_allclose(ds[2]["observation.state"].numpy()[-3:], [2, 4, -2])
    np.testing.assert_allclose(ds[2]["action"].numpy()[-2:], [2, 4])


def test_the_old_opt_in_flags_are_refused():
    with pytest.raises(ValueError, match="removed"):
        SceneRecorder("pkg", "extras", {"dataset": {"velocity": True}})


def test_zero_joint_force_records_zero_not_the_tare():
    """Right after a Reset PhysX's joint force can be exactly 0: no spurious +m g."""
    args = (np.eye(3), np.zeros(3), np.zeros(3), np.eye(3), 0.65, np.zeros(3))
    assert not tcp_wrench(np.zeros(3), np.zeros(3), *args).any()
    assert tcp_wrench([0, 0, 1.0], np.zeros(3), *args).any()


def test_old_layout_wx_as_rotation_is_refused():
    with pytest.raises(ValueError, match="rx ry rz"):
        check_layout({"observations": [{"wx": "wx"}, {"wz": "wz"}]})
    check_layout({"observations": [{"vx": "vx"}, {"wx": "wx"}]})  # a twist
    check_layout({"observations": [{"rx": "rx"}]})
