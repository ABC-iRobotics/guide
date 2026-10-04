"""Subtask prompts end up in the dataset as LeRobot subtask annotations.

Drives the real writer methods into a real (image-free) LeRobot dataset: frames carry a
subtask, episodes are saved or discarded, and after finalize every frame of a saved
episode resolves to the prompt that was active when it was recorded.
"""

import json
import logging

import pytest

from guide_core.scene.scene_recorder import SceneRecorder

lerobot_dataset = pytest.importorskip("lerobot.datasets.lerobot_dataset")
language_render = pytest.importorskip("lerobot.datasets.language_render")

POSE = {k: 0.0 for k in ("x", "y", "z", "wx", "wy", "wz")}


def record(recorder, subtasks):
    start = None
    for i, subtask in enumerate(subtasks):
        start = recorder._process_frame(
            {
                "timestamp": i,
                "observation": {"joint1.pos": float(i), **POSE},
                "action": {"joint1.pos": float(i), **POSE},
                "task": "Stack the cubes.",
                "subtask": subtask,
            },
            start,
        )


def test_saved_episodes_resolve_every_frame_to_its_subtask(tmp_path):
    recorder = SceneRecorder("pkg", "subtasks", {"dataset": {"fps": 10}})
    recorder._logger = logging.getLogger("test_subtasks")
    recorder.LeRobotDataset = lerobot_dataset.LeRobotDataset
    recorder.set_output_path(str(tmp_path))

    record(recorder, ["red on blue"] * 2 + ["green on red"] * 3)
    recorder._finalize_episode()
    record(recorder, ["dropped"] * 2)
    recorder._discard_episode()
    record(recorder, ["", "", "yellow on green", "yellow on green"])
    recorder._finalize_episode()
    root = recorder.dataset.root
    recorder._finalize_dataset()

    info = json.loads((root / "meta" / "info.json").read_text())
    assert "language_persistent" in info["features"]

    ds = lerobot_dataset.LeRobotDataset("pkg", root=root)
    rows = ds.hf_dataset.select_columns(["episode_index", "timestamp", "language_persistent"])
    active = {}
    for row in rows:
        hit = language_render.active_at(
            float(row["timestamp"]), persistent=row["language_persistent"], style="subtask"
        )
        active.setdefault(int(row["episode_index"]), []).append(hit and hit["content"])

    assert active == {
        0: ["red on blue"] * 2 + ["green on red"] * 3,
        1: [None, None, "yellow on green", "yellow on green"],
    }


def test_every_dataset_of_one_recorder_gets_its_run_info(tmp_path):
    recorder = SceneRecorder("pkg", "runs", {"dataset": {"fps": 10}})
    recorder._logger = logging.getLogger("test_subtasks")
    recorder.LeRobotDataset = lerobot_dataset.LeRobotDataset
    recorder.set_output_path(str(tmp_path))

    roots = []
    for _ in range(2):  # e.g. a trial run, then the real one, in one simulator session
        record(recorder, ["a", "a"])
        recorder._finalize_episode()
        roots.append(recorder.dataset.root)
        recorder._finalize_dataset()

    assert roots[0] != roots[1]
    assert all((root / "meta" / "guide_info.json").exists() for root in roots)
