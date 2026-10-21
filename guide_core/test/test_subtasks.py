"""Every GUIDE-EX layer's prompt ends up in the dataset, the LeRobot way.

Drives the real writer methods into a real (image-free) LeRobot dataset: frames carry what
SceneOrchestrator.record_step hands over -- the GUIDE-EX task as the frame's task (the
procedure outside every task), the procedure and the subtask as language prompts --,
episodes are saved or discarded, and after finalize every frame of a saved episode
resolves at each layer to the prompt active when it was recorded.
"""

import json
import logging

import pytest

from guide_core.scene.scene_recorder import SceneRecorder, register_guide_styles

lerobot_dataset = pytest.importorskip("lerobot.datasets.lerobot_dataset")
language_render = pytest.importorskip("lerobot.datasets.language_render")

POSE = {k: 0.0 for k in ("x", "y", "z", "rx", "ry", "rz")}
PROCEDURE = "Stack the cubes."


def record(recorder, prompts):
    """`prompts`: one (task, subtask) per frame, handed over as record_step does."""
    start = None
    for i, (task, subtask) in enumerate(prompts):
        start = recorder._process_frame(
            {
                "timestamp": i,
                "observation": {"joint1.pos": float(i), **POSE},
                "action": {"joint1.pos": float(i), **POSE},
                "task": task or PROCEDURE,
                "prompts": {"procedure": PROCEDURE, "subtask": subtask},
            },
            start,
        )


def active(root, style):
    """{episode: [content of the `style` row active at each frame, None if none]}."""
    register_guide_styles()
    ds = lerobot_dataset.LeRobotDataset("pkg", root=root)
    rows = ds.hf_dataset.select_columns(["episode_index", "timestamp", "language_persistent"])
    out = {}
    for row in rows:
        hit = language_render.active_at(float(row["timestamp"]), persistent=row["language_persistent"], style=style)
        out.setdefault(int(row["episode_index"]), []).append(hit and hit["content"])
    return out


def test_saved_episodes_resolve_every_frame_to_its_procedure_task_and_subtask(tmp_path):
    recorder = SceneRecorder("pkg", "subtasks", {"dataset": {"fps": 10}})
    recorder._logger = logging.getLogger("test_subtasks")
    recorder.LeRobotDataset = lerobot_dataset.LeRobotDataset
    recorder.set_output_path(str(tmp_path))

    put = "Put the red cube on the blue cube."
    record(recorder, [(put, "Pick up the red cube.")] * 2 + [(put, "Place it on the blue cube.")] * 3)
    recorder._finalize_episode()
    record(recorder, [("dropped", "dropped")] * 2)
    recorder._discard_episode()
    # A late first prompt, then the procedure closing outside every task (an empty
    # subtask is not recorded: the subtask keeps its last prompt until the next).
    record(recorder, [("", "")] * 2 + [(put, "Pick up the red cube.")] * 2 + [("", "")]
           + [("", "Return home.")] * 2)
    recorder._finalize_episode()
    root = recorder.dataset.root
    recorder._finalize_dataset()

    info = json.loads((root / "meta" / "info.json").read_text())
    assert "language_persistent" in info["features"]
    ds = lerobot_dataset.LeRobotDataset("pkg", root=root)
    tasks = {}
    for i in range(len(ds)):
        tasks.setdefault(int(ds[i]["episode_index"]), []).append(ds[i]["task"])
    assert tasks == {0: [put] * 5, 1: [PROCEDURE] * 2 + [put] * 2 + [PROCEDURE] * 3}
    assert active(root, "procedure") == {0: [PROCEDURE] * 5, 1: [PROCEDURE] * 7}
    assert active(root, "subtask") == {
        0: ["Pick up the red cube."] * 2 + ["Place it on the blue cube."] * 3,
        1: [None, None] + ["Pick up the red cube."] * 3 + ["Return home."] * 2,
    }


def test_the_camera_transition_label_holds_for_its_own_frames_only(tmp_path):
    transition = {"enabled": True, "duration": 2.0, "from": "base", "label": "Startup"}
    recorder = SceneRecorder("pkg", "startup", {"cameras": {"base": {}}, "dataset": {"fps": 10, "camera_transition": transition}})
    recorder._logger = logging.getLogger("test_subtasks")
    recorder.LeRobotDataset = lerobot_dataset.LeRobotDataset
    recorder.set_output_path(str(tmp_path))

    put = "Put the red block in the left bin."
    record(recorder, [(put, "Startup")] * 20 + [(put, "")] * 5)  # block_bin: no subtask of its own
    recorder._finalize_episode()
    record(recorder, [(put, "Startup")] * 20 + [(put, "Pick up the red cube.")] * 3)  # one announced before
    recorder._finalize_episode()
    root = recorder.dataset.root
    recorder._finalize_dataset()

    assert active(root, "subtask") == {
        0: ["Startup"] * 20 + [""] * 5,
        1: ["Startup"] * 20 + ["Pick up the red cube."] * 3,
    }


def test_every_dataset_of_one_recorder_gets_its_run_info(tmp_path):
    recorder = SceneRecorder("pkg", "runs", {"dataset": {"fps": 10}})
    recorder._logger = logging.getLogger("test_subtasks")
    recorder.LeRobotDataset = lerobot_dataset.LeRobotDataset
    recorder.set_output_path(str(tmp_path))

    roots = []
    for _ in range(2):  # e.g. a trial run, then the real one, in one simulator session
        record(recorder, [("a", "b"), ("a", "b")])
        recorder._finalize_episode()
        roots.append(recorder.dataset.root)
        recorder._finalize_dataset()

    assert roots[0] != roots[1]
    assert all((root / "meta" / "guide_info.json").exists() for root in roots)


def test_an_empty_subtask_row_round_trips_through_write_language(tmp_path):
    # After "Startup" block_bin announces no subtask, so the recorder closes the label with a
    # row of content "". LeRobot keeps it as written and reads it back as the active subtask.
    from guide_core.scene.scene_recorder import write_language

    recorder = SceneRecorder("pkg", "empty", {"dataset": {"fps": 10}})
    recorder._logger = logging.getLogger("test_subtasks")
    recorder.LeRobotDataset = lerobot_dataset.LeRobotDataset
    recorder.set_output_path(str(tmp_path))
    start = None
    for i in range(5):  # no prompt at all: the recorder writes no language of its own
        start = recorder._process_frame({"timestamp": i, "observation": {"joint1.pos": 0.0, **POSE},
                                         "action": {"joint1.pos": 0.0, **POSE}, "task": "t",
                                         "prompts": {"procedure": "", "subtask": ""}}, start)
    recorder._finalize_episode()
    root = recorder.dataset.root
    recorder._finalize_dataset()

    assert write_language(root, {0: {"subtask": [(0, "Startup"), (3, "")]}})

    register_guide_styles()
    rows = lerobot_dataset.LeRobotDataset("pkg", root=root)[4]["language_persistent"]
    assert [(r["style"], r["content"], r["timestamp"]) for r in rows] == [
        ("subtask", "Startup", 0.0), ("subtask", "", pytest.approx(0.3))]
    assert active(root, "subtask") == {0: ["Startup"] * 3 + [""] * 2}
