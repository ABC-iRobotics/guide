"""Tests for the offline replay viewer."""
import json

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")
matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")
rr = pytest.importorskip("block_bin.replay_rollout")


def an_episode(root, name, shade):
    """A single-trace directory with one flat-colour base frame, as a study writes it."""
    d = root / name
    (d / "frames" / "base").mkdir(parents=True)
    (d / "meta.json").write_text(json.dumps(
        {"episode": 0, "zone": 1, "seed": 0, "cameras": ["base"], "trial": name}))
    (d / "steps.jsonl").write_text(json.dumps(
        {"step": 0, "state": [0.0] * 8, "eef_position": [0, 0, 0.5], "plan": [],
         "q_measured": [0.0] * 7, "grip_command": 0.04, "grip_measured": 0.04}) + "\n")
    cv2.imwrite(str(d / "frames" / "base" / "0000.jpg"), np.full((480, 640, 3), shade, np.uint8))
    return d


def test_stepping_to_the_next_episode_shows_its_frames_not_a_blank(tmp_path):
    """Camera rows were keyed by trial name, which for study episodes is the episode
    name itself -- so the row built for ep_0000 found nothing in ep_0001 and blanked."""
    first = an_episode(tmp_path, "ep_0000", 40)
    second = an_episode(tmp_path, "ep_0001", 200)

    viewer = rr.Replay([first, second], "ep_0000")
    viewer.select(1)
    viewer.draw()

    _slot, _camera, _axes, artist = viewer.camera_axes[0]
    assert artist.get_array().mean() > 150, "second episode's frame should be showing"


def test_the_first_episode_shows_its_own_frame(tmp_path):
    first = an_episode(tmp_path, "ep_0000", 40)
    an_episode(tmp_path, "ep_0001", 200)

    viewer = rr.Replay([first], "ep_0000")
    viewer.draw()

    _slot, _camera, _axes, artist = viewer.camera_axes[0]
    assert artist.get_array().mean() < 100
