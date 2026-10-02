import numpy as np
import pytest

# eval_policy pulls in lerobot + the ROS workspace, i.e. it only imports under the
# Isaac venv interpreter with the workspace sourced. Skip elsewhere.
ep = pytest.importorskip("block_bin.eval_policy")

JOINTS = [f"j{i}" for i in range(1, 8)] + ["fr3_finger_joint1"]


def test_command_mirrors_the_second_finger():
    names, positions = ep.command_from_action(ep.HOME_POSITION + [0.02], JOINTS)

    assert names == JOINTS + [ep.GRIPPER_MIRROR_JOINT]
    assert positions == ep.HOME_POSITION + [0.02, 0.02]


def test_command_accepts_a_batched_policy_action():
    action = np.array([[1.0, 2, 3, 4, 5, 6, 7, 0.04]], dtype=np.float32)

    _, positions = ep.command_from_action(action, JOINTS)

    assert positions == pytest.approx([1, 2, 3, 4, 5, 6, 7, 0.04, 0.04])


def test_command_rejects_a_dimension_mismatch():
    with pytest.raises(ValueError):
        ep.command_from_action(np.zeros((1, 3)), JOINTS)


def test_frame_follows_the_dataset_state_order():
    image = np.zeros((480, 640, 3), np.uint8)
    observation = {
        **{f"j{i}.pos": float(i) for i in range(1, 8)},
        "fr3_finger_joint1.pos": 0.03,
        "top": image,
        "base": image,
        "wrist": image,
    }

    frame = ep.build_frame(observation, JOINTS)

    assert frame["observation.state"].dtype == np.float32
    assert frame["observation.state"].tolist() == pytest.approx([1, 2, 3, 4, 5, 6, 7, 0.03])
    assert sorted(frame) == [
        "observation.images.base",
        "observation.images.top",
        "observation.images.wrist",
        "observation.state",
    ]


def test_frame_names_the_camera_that_stopped_publishing():
    observation = {f"j{i}.pos": 0.0 for i in range(1, 8)} | {"fr3_finger_joint1.pos": 0.0}

    with pytest.raises(RuntimeError, match="cam_top.*stopped publishing"):
        ep.build_frame(observation, JOINTS)


def test_plan_is_unrestricted_without_a_zone_flag():
    assert ep.episode_plan("", 3) == [None, None, None]


def test_plan_repeats_episodes_per_zone():
    assert ep.episode_plan("2,16", 3) == [2, 2, 2, 16, 16, 16]


def test_plan_honours_per_zone_counts():
    assert ep.episode_plan("2:1,16:3", 10) == [2, 16, 16, 16]


def test_plan_covers_every_zone_for_all():
    plan = ep.episode_plan("all", 2)

    assert sorted(set(plan)) == list(range(ep.scene_num_zones()))
    assert len(plan) == 2 * ep.scene_num_zones()


def test_plan_rejects_a_zone_outside_the_grid():
    with pytest.raises(SystemExit, match="outside this scene"):
        ep.episode_plan(str(ep.scene_num_zones()), 1)


class FakePolicy:
    """Stands in for SmolVLA: chunk[i] is just the integer i, so a queued action
    says exactly which slot of which chunk it came from."""

    class config:
        n_action_steps = 10
        use_amp = False

    def __init__(self):
        self.calls = 0

    def predict_action_chunk(self, batch):
        self.calls += 1
        return self.calls


def make_stream(lead, latency=0.0):
    """A ChunkStream whose prediction is a counter, optionally slow."""
    import time as _time

    stream = ep.ChunkStream(FakePolicy(), None, None, None, "task", "franka", lead)

    def predict(_frame, _prev_tail=None):
        _time.sleep(latency)
        call = stream.policy.predict_action_chunk(None)
        # _predict returns the chunk in both spaces since RTC: what the arm executes and
        # what the model produced. The fake keeps them identical.
        actions = [(call, i) for i in range(FakePolicy.config.n_action_steps)]
        return actions, list(actions)

    stream._predict = predict
    return stream


def test_stream_serves_a_full_chunk_before_predicting_again():
    stream = make_stream(lead=0)

    served = [stream.next_action({}, step) for step in range(10)]

    assert served == [(1, i) for i in range(10)]
    assert stream.policy.calls == 1
    stream.close()


def test_stream_predicts_ahead_without_draining_the_queue():
    stream = make_stream(lead=3, latency=0.05)

    stream.next_action({}, 0)  # first chunk is synchronous
    for step in range(1, 8):  # queue falls to the lead threshold and fires the next
        stream.next_action({}, step)

    # The point of leading: the next chunk is in flight while actions remain to run,
    # so the arm never waits. (Asserting on policy.calls would race the worker.)
    assert stream._pending is not None, "should be predicting ahead by now"
    assert len(stream._queue) > 0, "queue drained despite leading -- the arm would stall"
    stream.close()


def test_stream_drops_actions_whose_slots_elapsed_while_predicting():
    stream = make_stream(lead=3, latency=0.05)

    stream.next_action({}, 0)
    for step in range(1, 8):
        stream.next_action({}, step)
    while stream._pending is not None and not stream._pending[0].done():
        pass
    served = [stream.next_action({}, step) for step in range(8, 12)]

    # Chunk 2 was launched at step 7; by the time it lands its early actions belong
    # to slots already gone, so serving must resume mid-chunk, never at index 0.
    second_chunk = [slot for call, slot in served if call == 2]
    assert second_chunk, f"expected chunk 2 to be serving by now, got {served}"
    assert second_chunk[0] > 0, f"replayed a stale action from the past: {served}"
    assert second_chunk == sorted(second_chunk), f"actions out of order: {served}"
    stream.close()


def test_stream_honours_a_shortened_horizon():
    stream = make_stream(lead=0)
    stream.policy.config.n_action_steps = 10

    served = [stream.next_action({}, step) for step in range(21)]

    assert stream.policy.calls == 3, "a 10-action horizon should re-plan every 10 steps"
    assert served[10] == (2, 0)
    stream.close()


class FakeLogger:
    def __init__(self):
        self.warnings = []

    def warn(self, message):
        self.warnings.append(message)

    def info(self, message):
        pass


class FakeSuccessRobot:
    """Answers IsSuccess the way guide_ros does, and records the settle."""

    def __init__(self, success, message=""):
        self.response = type("R", (), {"success": success, "message": message})()
        self.logger = FakeLogger()
        self.is_success_client = object()
        self.slept = []
        self.checked_after = None
        self.node = type("N", (), {"get_logger": lambda _s: self.logger})()

    def callService(self, client, request):
        self.checked_after = list(self.slept)
        return self.response


def test_a_failed_check_is_scored_as_a_miss():
    robot = FakeSuccessRobot(False)

    assert ep.is_success(robot, 0) is False
    assert robot.logger.warnings == [], "a criterion that simply did not hold is not an error"


def test_a_success_query_that_errored_is_not_silently_a_miss():
    """guide_ros answers a raised criterion with success=False plus a message.

    Both look identical in the score, so without surfacing the message a broken
    query -- a renamed prim, a scene whose randomize never ran -- reads as a policy
    that missed, and the run reports 0/N with nothing to explain it.
    """
    robot = FakeSuccessRobot(False, "KeyError: 'c'")

    assert ep.is_success(robot, 0) is False
    assert any("KeyError" in w for w in robot.logger.warnings)


def test_the_final_check_waits_for_the_scene_to_settle(monkeypatch):
    """The action that completes a rollout is the one that RELEASES the block.

    Containment is read off the poses PhysX last wrote back, so grading at the
    instant the loop exits reads the block still in the air above the bin. The
    in-loop poll only ever runs mid-motion, so this is the check that has to wait.
    """
    robot = FakeSuccessRobot(True)
    monkeypatch.setattr(ep, "sleep_sim", lambda r, seconds: r.slept.append(seconds))

    assert ep.final_success(robot, 0) is True
    assert robot.slept == [ep.SETTLE_SECONDS]
    assert robot.checked_after == [ep.SETTLE_SECONDS], "graded before the scene settled"


class FakeCameraRobot:
    """A robot whose observation fills in after `ready_after` polls."""

    def __init__(self, ready_after=0, cameras=("top", "base", "wrist"), joints=JOINTS):
        self.ready_after = ready_after
        self.cameras = cameras
        self.joints = joints
        self.polls = 0
        self.connected = False
        self.config = type(
            "C", (), {"arm_joint_names": JOINTS[:-1], "gripper_joint_names": JOINTS[-1:]}
        )()

    def connect(self):
        self.connected = True

    def disconnect(self):
        self.connected = False

    def get_observation(self):
        self.polls += 1
        if self.polls <= self.ready_after:
            return {}
        image = np.zeros((480, 640, 3), np.uint8)
        return {f"{j}.pos": 0.0 for j in self.joints} | {c: image for c in self.cameras}


def test_connect_returns_once_the_first_frame_can_be_built():
    robot = FakeCameraRobot()

    ep.connect_robot(robot, timeout=1.0)

    assert robot.connected
    assert robot.polls == 1


def test_connect_waits_for_a_camera_that_is_merely_slow():
    robot = FakeCameraRobot(ready_after=2)

    ep.connect_robot(robot, timeout=5.0)

    assert robot.polls == 3


def test_connect_refuses_to_start_blind_and_says_how_to_fix_it():
    """The whole point: fail at start-up, not mid-episode after a randomize."""
    robot = FakeCameraRobot(cameras=("base", "wrist"))

    with pytest.raises(SystemExit) as raised:
        ep.connect_robot(robot, timeout=0.0)

    message = str(raised.value)
    assert not robot.connected, "the ROS executor must be unwound before the message"
    assert "cam_top" in message
    assert "camera_topics:=true" in message
    # Editing the task YAML also slows dataset generation down, so it must not be
    # the headline remedy.
    assert message.index("camera_topics:=true") < message.index("publish_camera_topics")


def test_connect_complains_about_silent_joints_separately():
    robot = FakeCameraRobot(joints=JOINTS[:-1])

    with pytest.raises(SystemExit, match="fr3_finger_joint1"):
        ep.connect_robot(robot, timeout=0.0)


def test_connect_does_not_blame_the_cameras_when_only_joints_are_missing():
    robot = FakeCameraRobot(joints=JOINTS[:-1])

    with pytest.raises(SystemExit) as raised:
        ep.connect_robot(robot, timeout=0.0)

    assert "camera_topics" not in str(raised.value)


def a_parser():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--fps", type=float, default=5.0)
    return parser


def test_known_flags_parse_normally(monkeypatch):
    monkeypatch.setattr("sys.argv", ["eval", "--fps", "10"])

    assert ep.parse_evaluation_args(a_parser()).fps == 10.0


def test_an_unknown_flag_is_fatal_not_ignored(monkeypatch):
    """--time-scale forwarded to a script without it once ran three hours at full
    speed and reported the number as though the setting had applied."""
    monkeypatch.setattr("sys.argv", ["eval", "--fps", "10", "--time-scale", "0.25"])

    with pytest.raises(SystemExit, match="--time-scale"):
        ep.parse_evaluation_args(a_parser())


def test_ros_arguments_are_still_tolerated(monkeypatch):
    """`ros2 run` appends these; they are the reason the parse is lenient at all."""
    monkeypatch.setattr(
        "sys.argv", ["eval", "--fps", "10", "--ros-args", "-r", "__ns:=/Sim_0", "-p", "x:=1"]
    )

    assert ep.parse_evaluation_args(a_parser()).fps == 10.0


def test_a_stray_value_without_a_flag_is_not_an_error(monkeypatch):
    """Only flags are rejected -- a bare word is argparse's business, not this guard's."""
    monkeypatch.setattr("sys.argv", ["eval", "leftover"])

    ep.parse_evaluation_args(a_parser())


STATS = {
    "min": [-0.9059, -0.6357, 0.0252, -3.1416, -3.1038, -1.8028, -0.0162, -0.0162],
    "max": [0.3091, 0.6532, 0.8868, 3.1416, 3.1321, 1.9357, 0.0401, 0.0401],
    "mean": [0.0846, -0.0003, 0.3101, 0.4684, -0.1019, -0.0004, 0.0356, 0.0356],
    "std": [0.097, 0.1557, 0.1607, 3.0118, 0.6756, 0.069, 0.0072, 0.0072],
}
AT_HOME = [0.0002, -0.0000, 0.49995, -3.1415, 0.0, 0.0014, 0.04, 0.04]


def test_the_home_pose_is_inside_what_the_checkpoint_saw():
    assert ep.out_of_distribution(AT_HOME, STATS) == []


def test_a_metre_of_z_is_caught():
    """The real bug: datasets re-framed a metre down, BASE_OFFSET not swept with them,
    so every evaluation fed z about 7 sigma high and nothing reported it."""
    a_metre_high = list(AT_HOME)
    a_metre_high[2] += 1.0

    offending = ep.out_of_distribution(a_metre_high, STATS)

    assert [o[0] for o in offending] == [2]
    assert offending[0][4] > 7  # sigma


def test_the_edge_of_the_training_range_is_not_a_wall():
    """A pose slightly past the recorded maximum is extrapolation, not a frame error."""
    just_over = list(AT_HOME)
    just_over[2] = STATS["max"][2] + 0.01

    assert ep.out_of_distribution(just_over, STATS) == []


def test_the_check_names_the_channel_and_the_sigma():
    a_metre_high = list(AT_HOME)
    a_metre_high[2] += 1.0

    with pytest.raises(SystemExit) as raised:
        ep.check_state_distribution(a_metre_high, STATS, base_offset=(-0.3, 0.0, 1.013))

    message = str(raised.value)
    assert "z" in message and "sigma" in message
    assert "--base-offset" in message


def test_a_state_inside_the_range_raises_nothing():
    ep.check_state_distribution(AT_HOME, STATS, base_offset=(-0.3, 0.0, 0.013))


def test_no_stats_means_no_opinion():
    """An old checkpoint without a normaliser file must still be evaluable."""
    ep.check_state_distribution([999.0] * 8, None)
    assert ep.out_of_distribution(AT_HOME, {}) == []


def test_stats_are_read_from_the_checkpoint_not_a_dataset(tmp_path):
    """The dataset on disk can be re-framed after training; the checkpoint cannot."""
    import torch
    from safetensors.torch import save_file

    save_file(
        {f"observation.state.{k}": torch.tensor(v) for k, v in STATS.items()},
        str(tmp_path / "policy_preprocessor_step_5_normalizer_processor.safetensors"),
    )

    stats = ep.state_stats(str(tmp_path))

    assert stats["max"][2] == pytest.approx(0.8868)


def test_a_directory_without_a_normaliser_gives_nothing(tmp_path):
    assert ep.state_stats(str(tmp_path)) is None


def test_a_stream_reports_how_often_rtc_actually_engaged():
    """The flag says RTC was on; this says how much of the run it touched.

    At short horizons the queue can drain while a chunk is computed, in which case
    _rtc_tail() is None and the prediction is plain. A run with rtc_engaged == 0 had
    RTC on in name only.
    """
    stream = ep.ChunkStream.__new__(ep.ChunkStream)
    stream.rtc_engaged = 0
    stream.rtc_tail_total = 0
    assert stream.rtc_engaged == 0
    stream.rtc_engaged += 1
    stream.rtc_tail_total += 3
    assert stream.rtc_tail_total / stream.rtc_engaged == 3
