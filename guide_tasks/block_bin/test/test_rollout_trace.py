import json

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

# rollout_trace needs cv2; replay_rollout needs matplotlib. Both live in the Isaac
# venv, so skip anywhere else rather than fail.
rt = pytest.importorskip("block_bin.rollout_trace")
rr = pytest.importorskip("block_bin.replay_rollout")


def test_arrays_are_written_as_short_lists():
    """float64 noise makes a trace unreadable and compresses badly; nothing here is
    measured to fifteen digits."""
    plain = rt._plain({"q": np.array([1.234567891, 2.0])})

    assert plain == {"q": [1.23457, 2.0]}


def test_nested_structures_survive():
    assert rt._plain({"a": [np.float64(1.5), {"b": np.int64(3)}]}) == {"a": [1.5, {"b": 3}]}


def write_trace(directory, steps=3, cameras=("wrist",), frame_interval=1, pose_interval=1):
    trace = rt.RolloutTrace(
        directory, cameras=cameras, frame_interval=frame_interval, pose_interval=pose_interval
    )
    trace.poses = lambda: {"blue_block": [0.1, 0.2, 1.0]}
    trace.open({"seed": 7, "trial": "named", "cameras": list(cameras)})
    image = np.zeros((48, 64, 3), np.uint8)
    for step in range(steps):
        trace.step(
            step,
            {"eef_position": np.array([float(step), 0.0, 1.0]), "residual": 1e-5},
            {"wrist": image},
        )
    trace.close({"reached": "left"})
    return trace


def test_a_trace_round_trips(tmp_path):
    write_trace(tmp_path / "named")

    loaded = rt.load_trace(tmp_path / "named")

    assert loaded["meta"]["seed"] == 7
    assert loaded["meta"]["summary"]["reached"] == "left"
    assert [s["step"] for s in loaded["steps"]] == [0, 1, 2]
    assert loaded["steps"][2]["eef_position"] == [2.0, 0.0, 1.0]


def test_steps_are_readable_before_the_rollout_ends(tmp_path):
    """A killed rollout must leave the prefix behind -- that is usually the part worth
    looking at, and it is why steps.jsonl is flushed per step."""
    trace = rt.RolloutTrace(tmp_path / "named", cameras=())
    trace.open({"seed": 1})
    trace.step(0, {"residual": 0.1})
    trace.step(1, {"residual": 0.2})

    partial = rt.load_trace(tmp_path / "named")

    assert len(partial["steps"]) == 2
    assert "summary" not in partial["meta"]


def test_poses_are_polled_on_their_interval(tmp_path):
    write_trace(tmp_path / "named", steps=6, pose_interval=3)

    steps = rt.load_trace(tmp_path / "named")["steps"]

    assert [s["step"] for s in steps if "poses" in s] == [0, 3]


def test_frames_are_written_and_found(tmp_path):
    write_trace(tmp_path / "named", steps=2)

    trace = rt.load_trace(tmp_path / "named")

    assert rt.frame_path(trace, "wrist", 1).name == "0001.jpg"


def test_a_subsampled_frame_holds_until_the_next_one(tmp_path):
    """Blanking the panel between samples would make the viewer flicker; a video
    player holds the last frame and so does this."""
    write_trace(tmp_path / "named", steps=6, frame_interval=3)

    trace = rt.load_trace(tmp_path / "named")

    assert rt.frame_path(trace, "wrist", 3).name == "0003.jpg"
    assert rt.frame_path(trace, "wrist", 5).name == "0003.jpg"


def test_no_frame_before_the_first_one(tmp_path):
    trace = rt.RolloutTrace(tmp_path / "named", cameras=("wrist",))
    trace.open({"seed": 1})
    trace.close({})

    assert rt.frame_path(rt.load_trace(tmp_path / "named"), "wrist", 5) is None


def test_a_directory_without_meta_is_not_a_trace(tmp_path):
    (tmp_path / "empty").mkdir()

    with pytest.raises(SystemExit, match="not a rollout trace"):
        rt.load_trace(tmp_path / "empty")


def test_a_pair_loads_both_trials_left_first(tmp_path):
    """Left before right whatever the filesystem hands back, so the report and the
    viewer's legend read the same way every time."""
    write_trace(tmp_path / "seed_0000" / "right")
    write_trace(tmp_path / "seed_0000" / "left")

    pair = rt.load_pair(tmp_path / "seed_0000")

    assert list(pair) == ["left", "right"]


def test_an_older_named_flipped_pair_still_opens(tmp_path):
    """The labelling changed once; traces recorded before that are still worth reading."""
    write_trace(tmp_path / "seed_0000" / "named")
    write_trace(tmp_path / "seed_0000" / "flipped")

    assert list(rt.load_pair(tmp_path / "seed_0000")) == ["named", "flipped"]


def test_a_single_trial_directory_still_loads(tmp_path):
    write_trace(tmp_path / "named")

    assert sorted(rt.load_pair(tmp_path / "named")) == ["named"]


# ------------------------------------------------------------------ the viewer's maths


def step_with(plan, position=(0.0, 0.0, 1.0)):
    return {"eef_position": list(position), "plan": plan}


def test_an_empty_plan_is_just_the_current_position():
    path = rr.planned_path(step_with([]), {})

    assert path.shape == (1, 3)
    assert path[0].tolist() == [0.0, 0.0, 1.0]


def test_the_plan_integrates_the_queued_deltas():
    """Position deltas add, exactly as apply_delta composes them."""
    plan = [[0.1, 0, 0, 0, 0, 0, 0.04]] * 3

    path = rr.planned_path(step_with(plan), {"action_scale": 1.0})

    assert path[:, 0].tolist() == pytest.approx([0.0, 0.1, 0.2, 0.3])


def test_the_plan_is_drawn_through_the_action_scale():
    path = rr.planned_path(step_with([[0.1, 0, 0, 0, 0, 0, 0]]), {"action_scale": 0.5})

    assert path[1, 0] == pytest.approx(0.05)


def test_the_plan_is_clamped_the_way_the_executor_clamps_it():
    """Drawing the unclamped plan would show a trajectory the arm was never going to
    take, which is worse than showing none."""
    path = rr.planned_path(
        step_with([[1.0, 0, 0, 0, 0, 0, 0]]), {"action_scale": 1.0, "max_step": 0.02}
    )

    assert path[1, 0] == pytest.approx(0.02)


def test_the_rotation_ceiling_shortens_the_translation_too():
    """clamp_delta scales both halves by one factor, so a step clamped on rotation
    travels less far as well."""
    path = rr.planned_path(
        step_with([[0.1, 0, 0, 1.0, 0, 0, 0]]),
        {"action_scale": 1.0, "max_rotation_step": 0.5},
    )

    assert path[1, 0] == pytest.approx(0.05)


def test_a_short_action_stops_the_plan_rather_than_crashing():
    path = rr.planned_path(step_with([[0.1, 0, 0]]), {})

    assert path.shape == (1, 3)


def test_poses_hold_between_polls():
    steps = [{"poses": {"blue_block": [0, 0, 1]}}, {}, {}]

    assert rr.latest_poses(steps, 2, {})["blue_block"] == [0, 0, 1]


def test_poses_fall_back_to_the_start_of_the_scene():
    meta = {"objects_at_start": {"left_bin": [0.25, -0.4, 1.0]}}

    assert rr.latest_poses([{}, {}], 1, meta) == meta["objects_at_start"]


def test_a_missing_field_becomes_nan_not_an_exception():
    """Traces written by an older run are still worth opening."""
    values = rr.column([{"residual": 1.0}, {}], "residual")

    assert values[0] == 1.0
    assert np.isnan(values[1])


def test_a_component_is_picked_out_of_a_vector():
    steps = [{"q_measured": [1, 2, 3]}, {"q_measured": [4, 5, 6]}]

    assert rr.column(steps, "q_measured", 1).tolist() == [2.0, 5.0]


def test_a_short_vector_does_not_index_out_of_range():
    assert np.isnan(rr.column([{"q_measured": [1]}], "q_measured", 3)[0])


def test_the_viewer_opens_a_recorded_pair_and_renders(tmp_path):
    """End to end through the real figure: load a pair, draw a step, write a PNG."""
    import matplotlib

    matplotlib.use("Agg")
    write_trace(tmp_path / "seed_0000" / "left", steps=4)
    write_trace(tmp_path / "seed_0000" / "right", steps=4)

    replay = rr.Replay(rt.find_pairs(tmp_path), "left")
    replay.goto(2)
    replay.figure.savefig(tmp_path / "frame.png")

    assert (tmp_path / "frame.png").stat().st_size > 0
    assert replay.index == 2


def test_a_trace_with_no_steps_says_so(tmp_path):
    trace = rt.RolloutTrace(tmp_path / "named", cameras=())
    trace.open({"seed": 1})
    trace.close({})

    with pytest.raises(SystemExit, match="nothing to replay"):
        rr.Replay([tmp_path / "named"], "named")


def test_meta_json_stays_human_readable(tmp_path):
    write_trace(tmp_path / "named")

    text = (tmp_path / "named" / "meta.json").read_text()

    assert "\n" in text and json.loads(text)["trial"] == "named"


# ----------------------------------------------------------------- a run of seeds


def test_the_seed_is_read_off_the_directory_name(tmp_path):
    assert rt.seed_number(tmp_path / "seed_0007") == 7
    assert rt.seed_number(tmp_path / "named") == -1


def test_a_run_directory_yields_its_seeds_in_order(tmp_path):
    for seed in (2, 0, 1):
        (tmp_path / f"seed_{seed:04d}").mkdir()

    assert [rt.seed_number(p) for p in rt.find_pairs(tmp_path)] == [0, 1, 2]


def test_seeds_sort_numerically_not_alphabetically(tmp_path):
    """An unpadded run would otherwise put seed_10 between seed_1 and seed_2."""
    for name in ("seed_1", "seed_10", "seed_2"):
        (tmp_path / name).mkdir()

    assert [rt.seed_number(p) for p in rt.find_pairs(tmp_path)] == [1, 2, 10]


def test_a_single_pair_directory_is_still_one_pair(tmp_path):
    (tmp_path / "named").mkdir()
    (tmp_path / "flipped").mkdir()

    assert rt.find_pairs(tmp_path) == [tmp_path]


def test_a_missing_directory_says_so(tmp_path):
    with pytest.raises(SystemExit, match="not a directory"):
        rt.find_pairs(tmp_path / "nope")


def a_run(tmp_path, seeds=(0, 1, 2), steps=4):
    for seed in seeds:
        for trial in ("left", "right"):
            write_trace(tmp_path / f"seed_{seed:04d}" / trial, steps=steps)
    return rt.find_pairs(tmp_path)


def test_the_viewer_opens_a_whole_run(tmp_path):
    import matplotlib

    matplotlib.use("Agg")
    replay = rr.Replay(a_run(tmp_path), "left")

    assert len(replay.pair_paths) == 3
    assert replay.position == 0


def test_stepping_to_the_next_seed_keeps_the_step_and_the_trial(tmp_path):
    """Comparing the same moment across seeds is the reason to have them side by side."""
    import matplotlib

    matplotlib.use("Agg")
    replay = rr.Replay(a_run(tmp_path), "right")
    replay.goto(2)

    replay.select(replay.position + 1)

    assert replay.position == 1
    assert replay.index == 2
    assert replay.trial == "right"


def test_seed_navigation_wraps(tmp_path):
    import matplotlib

    matplotlib.use("Agg")
    replay = rr.Replay(a_run(tmp_path), "left")

    replay.select(-1)

    assert replay.position == 2


def test_a_named_seed_opens_directly(tmp_path):
    import matplotlib

    matplotlib.use("Agg")
    replay = rr.Replay(a_run(tmp_path), "left", seed=2)

    assert replay.position == 2


def test_asking_for_a_seed_that_was_not_recorded_lists_the_ones_that_were(tmp_path):
    import matplotlib

    matplotlib.use("Agg")
    with pytest.raises(SystemExit, match=r"\[0, 1, 2\]"):
        rr.Replay(a_run(tmp_path), "left", seed=9)


def test_a_shorter_seed_moves_the_slider_bound_with_it(tmp_path):
    """One aborted rollout in a run would otherwise leave the slider running off the
    end of the next seed's data."""
    import matplotlib

    matplotlib.use("Agg")
    for trial in ("left", "right"):
        write_trace(tmp_path / "seed_0000" / trial, steps=9)
        write_trace(tmp_path / "seed_0001" / trial, steps=3)
    replay = rr.Replay(rt.find_pairs(tmp_path), "left")
    replay.goto(8)

    replay.select(1)

    assert replay.slider.valmax == 2
    assert replay.index == 2


def test_the_viewer_takes_its_keys_back_from_matplotlib():
    """left/right pan and g toggles the grid by default, so both would fire."""
    import matplotlib.pyplot as plt

    rr.free_keys()

    assert "left" not in plt.rcParams["keymap.back"]
    assert "right" not in plt.rcParams["keymap.forward"]
    assert "g" not in plt.rcParams["keymap.grid"]
    assert "home" not in plt.rcParams["keymap.home"]
    # Keys the viewer does not use are left alone.
    assert "p" in plt.rcParams["keymap.pan"]


def test_both_trials_get_their_own_camera_row(tmp_path):
    """The pair exists to be compared; showing one trial's inputs at a time would hide
    exactly the comparison the tool is for."""
    import matplotlib

    matplotlib.use("Agg")
    replay = rr.Replay(a_run(tmp_path, seeds=(0,)), "left")

    rows = {name for name, _camera, _axes, _artist in replay.camera_axes}

    assert rows == {"left", "right"}
    assert len(replay.camera_axes) == 2 * len(replay.cameras)


def test_a_single_trial_trace_gets_one_row(tmp_path):
    import matplotlib

    matplotlib.use("Agg")
    write_trace(tmp_path / "solo", steps=3)

    replay = rr.Replay([tmp_path / "solo"], "solo")

    assert {name for name, *_ in replay.camera_axes} == {"solo"}


def test_the_info_panel_names_the_side_the_scene_drew(tmp_path):
    """A renamed metadata field silently printed 'None' here, which reads as a scene
    that drew nothing rather than a viewer looking up the wrong key."""
    import matplotlib

    matplotlib.use("Agg")
    for side in ("left", "right"):
        trace = rt.RolloutTrace(tmp_path / "seed_0000" / side, cameras=())
        trace.open({"seed": 0, "trial": side, "asked_for": side, "scene_side": "right"})
        trace.step(0, {"residual": 1e-5})
        trace.close({})

    replay = rr.Replay(rt.find_pairs(tmp_path), "left")
    text = replay.info_text.get_text()

    assert "scene drew : right" in text
    assert "asked for: left" in text


def test_a_pose_that_never_moved_has_no_tilt():
    steps = [{"eef_rotvec": [np.pi, 0, 0]}] * 3

    assert rr.tilt_from_start(steps).tolist() == pytest.approx([0.0, 0.0, 0.0])


def test_tilt_measures_the_true_relative_rotation():
    """90 degrees about z away from the start pose reads as 90, not as the difference
    of the two rotation vectors."""
    start = [np.pi, 0, 0]
    turned = (
        Rotation.from_rotvec([0, 0, np.pi / 2]) * Rotation.from_rotvec(start)
    ).as_rotvec()

    tilt = rr.tilt_from_start([{"eef_rotvec": start}, {"eef_rotvec": list(turned)}])

    assert tilt[1] == pytest.approx(90.0, abs=1e-6)


def test_the_antipodal_sign_flip_is_not_reported_as_tumbling():
    """The down-pose sits at |rotvec| = pi where the sign is arbitrary. Subtracting
    rotation vectors would call this a 360 degree flip; it is the same pose."""
    tilt = rr.tilt_from_start(
        [{"eef_rotvec": [np.pi, 0, 0]}, {"eef_rotvec": [-np.pi, 0, 0]}]
    )

    assert tilt[1] == pytest.approx(0.0, abs=1e-6)


def test_a_trace_without_orientation_has_no_tilt_trace():
    assert rr.tilt_from_start([{"residual": 1.0}]) is None


def a_replay(tmp_path, seeds=(0, 1)):
    import matplotlib

    matplotlib.use("Agg")
    return rr.Replay(a_run(tmp_path, seeds=seeds), "left")


def test_the_camera_survives_stepping(tmp_path):
    """clear() throws the view away, so every step used to snap the 3D panel back to
    default -- which made it impossible to rotate while stepping, i.e. useless."""
    replay = a_replay(tmp_path)
    replay.space.view_init(elev=11.0, azim=-33.0, roll=0.0)

    replay.goto(2)

    assert replay.space.elev == pytest.approx(11.0)
    assert replay.space.azim == pytest.approx(-33.0)


def test_the_camera_survives_a_seed_change(tmp_path):
    replay = a_replay(tmp_path)
    replay.space.view_init(elev=11.0, azim=-33.0, roll=0.0)
    replay.draw()

    replay.select(1)

    assert replay.space.elev == pytest.approx(11.0)


def test_v_resets_the_camera(tmp_path):
    replay = a_replay(tmp_path)
    replay.space.view_init(elev=80.0, azim=5.0, roll=0.0)
    replay.draw()

    replay.on_key(type("E", (), {"key": "v"})())

    assert replay.space.elev == pytest.approx(rr.DEFAULT_VIEW[0])


def test_the_three_axes_share_one_scale(tmp_path):
    """Matplotlib autoscales each axis on its own, which stretches a 0.4 m move in z
    to look like a 0.9 m move in y -- exactly the comparison this panel is for."""
    replay = a_replay(tmp_path)

    spans = [
        replay.space.get_xlim()[1] - replay.space.get_xlim()[0],
        replay.space.get_ylim()[1] - replay.space.get_ylim()[0],
        replay.space.get_zlim()[1] - replay.space.get_zlim()[0],
    ]

    assert spans[0] == pytest.approx(spans[1]) == pytest.approx(spans[2])


def test_the_bounds_do_not_move_while_stepping(tmp_path):
    replay = a_replay(tmp_path)
    before = replay.space.get_xlim()

    replay.goto(3)

    assert replay.space.get_xlim() == pytest.approx(before)


def test_a_new_seed_gets_its_own_bounds(tmp_path):
    replay = a_replay(tmp_path)
    replay.select(1)

    assert replay.bounds is not None


def test_a_trace_without_positions_still_opens(tmp_path):
    """An all-NaN point cloud gave NaN axis limits, which matplotlib refuses."""
    import matplotlib

    matplotlib.use("Agg")
    for side in ("left", "right"):
        trace = rt.RolloutTrace(tmp_path / "seed_0000" / side, cameras=())
        trace.open({"seed": 0, "trial": side})
        trace.step(0, {"residual": 1e-5})
        trace.close({})

    replay = rr.Replay(rt.find_pairs(tmp_path), "left")

    assert replay.bounds is None  # nothing to frame, and it said so instead of raising


def test_the_viewer_draws_objects_in_the_tool_frame():
    """Without this the blocks float a metre above the trajectory reaching for them."""
    steps = [{"poses": {"red_block": [0.25, -0.4, 1.09]}}]

    placed = rr.latest_poses(steps, 0, {"scene_origin": [0.0, 0.0, 1.0]})

    assert placed["red_block"] == pytest.approx([0.25, -0.4, 0.09])


def test_an_older_trace_without_an_origin_is_drawn_as_recorded():
    steps = [{"poses": {"red_block": [0.25, -0.4, 1.09]}}]

    assert rr.latest_poses(steps, 0, {})["red_block"] == [0.25, -0.4, 1.09]
