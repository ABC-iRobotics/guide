"""A discarded attempt's camera frames never reach the next saved episode's video.

lerobot 0.6's clear_episode_buffer deletes the buffered frames of image features only;
cameras are video features here, so a retry (same episode index) overwrote just its own
frames and a longer failed attempt's tail was encoded after them: the clip ran past the
episode (22-319 frames, about one episode per dataset with failed attempts).
"""

import logging

import numpy as np
import pytest

from guide_core.scene.scene_recorder import SceneRecorder

lerobot_dataset = pytest.importorskip("lerobot.datasets.lerobot_dataset")

POSE = {k: 0.0 for k in ("x", "y", "z", "wx", "wy", "wz")}


def record(recorder, frames):
    start = None
    for i in range(frames):
        start = recorder._process_frame(
            {
                "timestamp": i,
                "observation": {"joint1.pos": float(i), "top": np.full((64, 64, 3), 40 * i, np.uint8), **POSE},
                "action": {"joint1.pos": float(i), **POSE},
                "task": "Stack the cubes.",
            },
            start,
        )


def test_a_discarded_longer_attempt_leaves_no_frames_in_the_next_video(tmp_path):
    recorder = SceneRecorder("pkg", "discard", {"dataset": {"fps": 10}})
    recorder._logger = logging.getLogger("test_discarded_frames")
    recorder.LeRobotDataset = lerobot_dataset.LeRobotDataset
    recorder.set_output_path(str(tmp_path))

    record(recorder, 5)
    recorder._discard_episode()
    record(recorder, 3)
    recorder._finalize_episode()
    root = recorder.dataset.root
    recorder._finalize_dataset()

    ds = lerobot_dataset.LeRobotDataset("pkg", root=root)
    episode = ds.meta.episodes[0]
    key = "observation.images.top"
    frames = round((episode[f"videos/{key}/to_timestamp"] - episode[f"videos/{key}/from_timestamp"]) * 10)
    assert (episode["length"], frames) == (3, 3)
