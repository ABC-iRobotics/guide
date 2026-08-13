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


def test_frame_blames_the_camera_topics_when_an_image_is_missing():
    observation = {f"j{i}.pos": 0.0 for i in range(1, 8)} | {"fr3_finger_joint1.pos": 0.0}

    with pytest.raises(RuntimeError, match="publish_camera_topics"):
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

    def predict(_frame):
        _time.sleep(latency)
        call = stream.policy.predict_action_chunk(None)
        return [(call, i) for i in range(FakePolicy.config.n_action_steps)]

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
