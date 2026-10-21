"""dataset.velocity / dataset.tcp_wrench: features, the recorder, and the wrench pipeline against Pinocchio."""

import logging

import numpy as np
import pinocchio as pin
import pytest
from scipy.spatial.transform import Rotation

from guide_core.scene.extra_features import extra_features, tcp_wrench
from guide_core.scene.scene_recorder import SceneRecorder

OBS = [{"joint1.pos": "franka.fr3_joint1.pos"}, {"joint2.pos": "franka.fr3_joint2.pos"}, {"gripper.pos": "franka.fr3_finger_joint1.pos"},
       {"x": "x"}, {"wz": "wz"}]


def test_off_by_default_leaves_the_features_unchanged():
    assert extra_features({"observations": OBS}) == {}
    assert extra_features({"observations": OBS, "velocity": False, "tcp_wrench": False}) == {}


def test_feature_shapes_and_names():
    f = extra_features({"observations": OBS, "velocity": True, "tcp_wrench": True})
    assert list(f) == ["observation.velocity", "observation.tcp_twist", "observation.tcp_wrench"]
    assert f["observation.velocity"]["names"] == ["joint1.vel", "joint2.vel", "gripper.vel"]  # observation.state's joints
    assert f["observation.velocity"]["shape"] == (3,)
    assert f["observation.tcp_twist"]["names"] == ["vx_m_s", "vy_m_s", "vz_m_s", "wx_rad_s", "wy_rad_s", "wz_rad_s"]
    assert f["observation.tcp_wrench"]["names"] == ["fx_N", "fy_N", "fz_N", "tx_Nm", "ty_Nm", "tz_Nm"]
    assert all(v["dtype"] == "float32" and v["info"]["frame"] for k, v in f.items() if k != "observation.velocity")
    assert "environment applies" in f["observation.tcp_wrench"]["info"]["sign"]
    assert list(extra_features({"observations": OBS, "tcp_wrench": True})) == ["observation.tcp_wrench"]
    with pytest.raises(ValueError):
        extra_features({"velocity": True})


@pytest.mark.parametrize("base", [np.eye(3), Rotation.from_euler("xyz", [0.3, -0.5, 1.2]).as_matrix()])
def test_wrench_recovers_a_known_force_at_the_tcp(base):
    """Static FR3: Pinocchio's RNEA with a force at the TCP gives the force through the last joint (gravity
    and the hand's weight included); the pipeline must hand the TCP force back."""
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
        got = tcp_wrench(f7.linear, f7.angular, oMi.rotation, oMi.translation, T_tcp.translation, base,
                         body.mass, oMi.act(body.lever))
        np.testing.assert_allclose(got, np.r_[base.T @ w[:3], base.T @ w[3:]], atol=1e-9)


def test_free_space_reads_zero_after_the_tare():
    from guide_motion.model import Model

    model = Model.load("fr3")
    pm, data = model.pin_model, model.pin_model.createData()
    q = model.home
    pin.rnea(pm, data, q, np.zeros(7), np.zeros(7))
    pin.framesForwardKinematics(pm, data, q)
    oMi, body = data.oMi[7], pm.inertias[7]
    w = tcp_wrench(data.f[7].linear, data.f[7].angular, oMi.rotation, oMi.translation, data.oMf[pm.getFrameId(model.tool_frame)].translation,
                   np.eye(3), body.mass, oMi.act(body.lever))
    np.testing.assert_allclose(w, 0, atol=1e-9)


def test_the_recorder_writes_the_extra_features(tmp_path):
    lerobot_dataset = pytest.importorskip("lerobot.datasets.lerobot_dataset")
    cfg = {"fps": 10, "velocity": True, "tcp_wrench": True, "observations": OBS[:1]}
    recorder = SceneRecorder("pkg", "extras", {"dataset": cfg})
    recorder._logger = logging.getLogger("test_extra_features")
    recorder.LeRobotDataset = lerobot_dataset.LeRobotDataset
    recorder.set_output_path(str(tmp_path))
    pose = {k: 0.0 for k in ("x", "y", "z", "wx", "wy", "wz")}
    start = None
    for i in range(3):
        extras = {"observation.velocity": np.array([i], np.float32), "observation.tcp_twist": np.full(6, i, np.float32),
                  "observation.tcp_wrench": np.arange(6, dtype=np.float32) + i}
        start = recorder._process_frame({"timestamp": i, "observation": {"joint1.pos": 0.0, **pose}, "action": {"joint1.pos": 0.0, **pose},
                                         "task": "t", "extras": extras}, start)
    recorder._finalize_episode()
    root = recorder.dataset.root
    recorder._finalize_dataset()
    ds = lerobot_dataset.LeRobotDataset("pkg", root=root)
    assert ds.features["observation.state"]["names"][0] == "joint1.pos" and "observation.velocity" not in ds.features["observation.state"]["names"]
    assert ds.features["observation.tcp_wrench"]["shape"] == (6,)
    assert ds.features["observation.tcp_wrench"]["names"][0] == "fx_N"
    assert "environment applies" in ds.features["observation.tcp_wrench"]["info"]["sign"]  # survives info.json
    np.testing.assert_allclose(ds[2]["observation.tcp_wrench"].numpy(), np.arange(6) + 2)
    np.testing.assert_allclose(ds[1]["observation.velocity"].numpy(), [1])
