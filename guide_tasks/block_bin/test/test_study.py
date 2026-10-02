import json
import math

import pandas as pd
import pytest

st = pytest.importorskip("block_bin.study")
rt = pytest.importorskip("block_bin.rollout_trace")


def a_trace(tmp_path, name, heights, episode=0, zone=3, summary=None):
    trace = rt.RolloutTrace(tmp_path / "traces" / "180000" / name, cameras=(), pose_interval=1)
    trace.poses = None
    trace.open({"episode": episode, "zone": zone})
    for i, h in enumerate(heights):
        row = {"eef_position": [0, 0, 0.5], "sim_seconds": 0.2 * i}
        trace._steps.write(json.dumps({"step": i, **row, "poses": {"red_block": [0, 0, h]}}) + "\n")
    trace.close(summary or {})
    return trace


def test_lift_is_the_rise_above_where_the_block_started(tmp_path):
    a_trace(tmp_path, "ep_0000", [1.02, 1.10, 1.18])

    trace = rt.load_trace(tmp_path / "traces" / "180000" / "ep_0000")

    assert st.lift_of(trace) == pytest.approx(0.16)


def test_a_block_that_only_jittered_has_no_lift(tmp_path):
    """The push that fooled IsSuccess peaked at 0.0155 m -- table jitter."""
    a_trace(tmp_path, "ep_0000", [1.041, 1.048, 1.0565])

    assert st.lift_of(rt.load_trace(tmp_path / "traces" / "180000" / "ep_0000")) < st.LIFT_MIN


def test_a_trace_without_poses_has_no_opinion(tmp_path):
    """NaN, not 0.0 -- an unpolled episode must not be scored as a failed lift."""
    trace = rt.RolloutTrace(tmp_path / "traces" / "180000" / "ep_0000", cameras=())
    trace.open({"episode": 0, "zone": 3})
    trace.step(0, {"eef_position": [0, 0, 0.5]})
    trace.close({})

    assert math.isnan(st.lift_of(rt.load_trace(tmp_path / "traces" / "180000" / "ep_0000")))


def test_episodes_are_found_in_order(tmp_path):
    for i in range(3):
        a_trace(tmp_path, f"ep_{i:04d}", [1.0, 1.1], episode=i)

    found = st.episode_traces(tmp_path)

    assert [p.name for _c, p in found] == ["ep_0000", "ep_0001", "ep_0002"]
    assert {c for c, _p in found} == {"180000"}


def test_zones_are_split_by_what_the_checkpoint_trained_on():
    frame = pd.DataFrame({"zone": [3, 12, 7, 17]})

    split = st.label_zones(frame, {1, 2, 3, 6, 7, 8})["split"].tolist()

    assert split == ["in-distribution", "held-out", "in-distribution", "held-out"]


def scored_frame(rows):
    """Episodes as score() sees them. `success` is the scene's IsSuccess verdict;
    `task_success` is the ladder's own -- the right cube in the right bin -- and it is
    what `placed` is built from."""
    rows = [{"task_success": r.get("success", False), **r} for r in rows]
    return pd.DataFrame(rows)


def test_a_claimed_success_without_a_lift_is_not_a_placement():
    """The whole reason this script exists rather than sweep_checkpoints alone."""
    frame = scored_frame([
        {"checkpoint": "180000", "split": "in-distribution", "success": True, "lift": 0.18},
        {"checkpoint": "180000", "split": "in-distribution", "success": True, "lift": 0.01},
    ])

    row = st.score(frame).iloc[0]

    assert row["claimed"] == 2
    assert row["placed"] == 1


def test_the_two_splits_are_scored_separately():
    """Pooling gives a number that is neither, and that moves with the mix."""
    frame = scored_frame(
        [{"checkpoint": "a", "split": "in-distribution", "success": True, "lift": 0.2}] * 6
        + [{"checkpoint": "a", "split": "held-out", "success": False, "lift": 0.0}] * 4
    )

    scored = st.score(frame).set_index("split")

    assert scored.loc["in-distribution", "rate"] == 1.0
    assert scored.loc["held-out", "rate"] == 0.0


def test_overlapping_intervals_are_reported_as_unrankable():
    """n=10 cannot separate 40% from 30%, and saying so is the point."""
    frame = scored_frame(
        [{"checkpoint": "180000", "split": "in-distribution", "success": i < 4, "lift": 0.2}
         for i in range(10)]
        + [{"checkpoint": "150000", "split": "in-distribution", "success": i < 3, "lift": 0.2}
           for i in range(10)]
    )

    notes = " ".join(st.ranking(st.score(frame)))

    assert "NOT separable" in notes
    assert "180000" in notes


def test_a_real_ordering_is_called_one():
    frame = scored_frame(
        [{"checkpoint": "180000", "split": "in-distribution", "success": True, "lift": 0.2}
         for _ in range(30)]
        + [{"checkpoint": "030000", "split": "in-distribution", "success": False, "lift": 0.0}
           for _ in range(30)]
    )

    notes = " ".join(st.ranking(st.score(frame)))

    assert "do not overlap" in notes


def test_a_disagreement_with_the_scenes_verdict_is_reported():
    """One episode IsSuccess called a win never left the table."""
    frame = scored_frame([
        {"checkpoint": "180000", "split": "in-distribution", "success": True, "lift": 0.01},
        {"checkpoint": "180000", "split": "in-distribution", "success": True, "lift": 0.2},
    ])

    assert "differ on 1" in " ".join(st.ranking(st.score(frame)))


def test_the_ranking_shows_where_the_behaviour_stops():
    frame = scored_frame([
        {"checkpoint": "180000", "split": "in-distribution", "success": False, "lift": 0.0,
         "closed": True, "lifted": False},
    ])

    notes = " ".join(st.ranking(st.score(frame)))

    assert "closed on a cube 1/1" in notes
    assert "lifted one 0/1" in notes


def test_nothing_recorded_ranks_nothing():
    assert "No in-distribution episodes" in st.ranking(pd.DataFrame(
        [{"checkpoint": "a", "split": "held-out", "episodes": 1, "placed": 0, "rate": 0.0,
          "ci_low": 0.0, "ci_high": 1.0, "pushes": 0}]))[0]


def steps_with(grips, cube_z=None, name="red_block"):
    """Steps carrying a gripper trace and optionally one cube's height."""
    rows = []
    for i, (c, m) in enumerate(grips):
        row = {"step": i, "grip_command": c, "grip_measured": m, "sim_seconds": 0.2 * i}
        if cube_z is not None:
            row["poses"] = {name: [0.0, 0.0, cube_z[i]]}
        rows.append(row)
    return rows


def test_fingers_closing_through_to_zero_grasped_nothing():
    """An empty close runs to ~0; measured across the campaign, misses hit -0.02..0.01."""
    assert not st.grasped_something(steps_with([(0.01, 0.03), (0.01, 0.01), (0.01, -0.02)]))


def test_fingers_stalling_on_a_cube_count_as_a_grasp():
    """A cube's half-width stops them near 0.025."""
    assert st.grasped_something(steps_with([(0.01, 0.03), (0.01, 0.026), (0.01, 0.025)]))


def test_one_good_grasp_is_not_erased_by_a_later_empty_close():
    """Scored per contiguous close, not over the episode -- otherwise a policy that
    grasps then re-opens and closes on air is scored by its worst attempt."""
    grips = [(0.01, 0.025), (0.04, 0.04), (0.01, 0.001)]

    assert st.grasped_something(steps_with(grips))


def test_never_commanding_a_close_is_not_a_grasp():
    assert not st.grasped_something(steps_with([(0.04, 0.04)] * 3))


def in_memory_trace(steps, summary):
    """A trace dict as load_trace would return it, without touching disk."""
    return {"directory": None, "meta": {"summary": summary}, "steps": steps}


def test_a_cube_that_rose_counts_as_lifted():
    t = in_memory_trace(steps_with([(0.01, 0.025)] * 3, cube_z=[1.02, 1.10, 1.20]), {})

    assert st.lifts(t)["red_block"] == pytest.approx(0.18)
    assert st.rungs(t)["lifted"]


def test_lifting_the_wrong_cube_registers_but_is_not_the_target():
    """Grasping a distractor is a different failure from never closing at all."""
    steps = steps_with([(0.01, 0.025)] * 3, cube_z=[1.02, 1.10, 1.20], name="blue_block")
    t = in_memory_trace(steps, {"target": "/blocks/red_block", "goal": "/bin_0",
                        "bin_contents": {"blue_block": "left", "red_block": None}})

    r = st.rungs(t)

    assert r["lifted"] and not r["lifted_target"]
    assert r["binned"] and not r["binned_target"]
    assert not r["task_success"]


def test_the_right_cube_in_the_right_bin_is_the_only_success():
    steps = steps_with([(0.01, 0.025)] * 3, cube_z=[1.02, 1.10, 1.20])
    t = in_memory_trace(steps, {"target": "/blocks/red_block", "goal": "/bin_0",
                        "bin_contents": {"red_block": "left"}})

    assert st.rungs(t)["task_success"]


def test_the_right_cube_in_the_wrong_bin_is_not_a_success():
    steps = steps_with([(0.01, 0.025)] * 3, cube_z=[1.02, 1.10, 1.20])
    t = in_memory_trace(steps, {"target": "/blocks/red_block", "goal": "/bin_1",
                        "bin_contents": {"red_block": "left"}})

    r = st.rungs(t)

    assert r["binned_target"] and not r["task_success"]


def test_a_cube_shoved_into_a_bin_without_a_lift_is_not_binned():
    """The push that satisfied IsSuccess peaked at 0.0155 m."""
    steps = steps_with([(0.01, 0.001)] * 3, cube_z=[1.041, 1.048, 1.0565])
    t = in_memory_trace(steps, {"target": "/blocks/red_block", "goal": "/bin_0",
                        "bin_contents": {"red_block": "left"}})

    r = st.rungs(t)

    assert not r["lifted"] and not r["binned"] and not r["task_success"]


def test_the_lift_column_follows_the_target_not_whichever_cube_sorts_first():
    """Once all four cubes are polled, taking the first dict entry silently measured
    a distractor -- and a real success then read lift 0.0 and scored as a failure."""
    steps = [
        {"poses": {"blue_block": [0, 0, 1.02], "red_block": [0, 0, 1.02]}},
        {"poses": {"blue_block": [0, 0, 1.02], "red_block": [0, 0, 1.20]}},
    ]
    t = in_memory_trace(steps, {"target": "/blocks/red_block"})

    assert st.lift_of(t) == pytest.approx(0.18)


def test_a_graded_success_is_not_re_gated_on_the_lift_column():
    frame = scored_frame([
        {"checkpoint": "a", "split": "in-distribution", "success": False,
         "task_success": True, "graded": True, "lift": 0.0},
    ])

    assert int(st.score(frame).iloc[0]["placed"]) == 1


def wrong_cube_trace(contents, goal="/bin_0"):
    steps = [
        {"poses": {"red_block": [0, 0, 1.02], "blue_block": [0, 0, 1.02]}},
        {"poses": {"red_block": [0, 0, 1.02], "blue_block": [0, 0, 1.20]}},
    ]
    return in_memory_trace(
        steps, {"target": "/blocks/red_block", "goal": goal, "bin_contents": contents}
    )


def test_the_whole_task_done_to_the_wrong_cube_is_its_own_outcome():
    """Everything except identifying the cube worked -- a different defect from a
    fumbled grasp, and invisible if the ladder stops at 'lifted the wrong cube'."""
    r = st.rungs(wrong_cube_trace({"blue_block": "left", "red_block": None}))

    assert r["wrong_lifted"] and r["wrong_binned"] and r["wrong_in_goal_bin"]
    assert not r["task_success"]


def test_the_wrong_cube_in_the_wrong_bin_is_not_that_case():
    r = st.rungs(wrong_cube_trace({"blue_block": "right", "red_block": None}))

    assert r["wrong_binned"] and not r["wrong_in_goal_bin"]


def test_lifting_a_distractor_without_binning_it_is_not_that_case():
    r = st.rungs(wrong_cube_trace({"blue_block": None, "red_block": None}))

    assert r["wrong_lifted"] and not r["wrong_binned"] and not r["wrong_in_goal_bin"]


def _write_trace(directory, episode, zone=1, seed=0):
    import json
    d = directory / f"ep_{episode:04d}"
    d.mkdir(parents=True)
    (d / "meta.json").write_text(json.dumps({"episode": episode, "zone": zone, "seed": seed}))
    (d / "steps.jsonl").write_text("")
    return d


def test_episode_traces_reads_a_study_layout(tmp_path):
    _write_trace(tmp_path / "traces" / "180000", 0)
    _write_trace(tmp_path / "traces" / "180000", 1)

    found = st.episode_traces(tmp_path)

    assert [(c, p.name) for c, p in found] == [("180000", "ep_0000"), ("180000", "ep_0001")]


def test_episode_traces_reads_a_flat_debug_layout(tmp_path):
    """A direct eval_policy_pink run has no checkpoint level; the folder names it."""
    run = tmp_path / "debug_run"
    _write_trace(run, 0)
    _write_trace(run, 1)
    (run / "episodes.jsonl").write_text("")            # sits beside them, not a trace

    found = st.episode_traces(run)

    assert [(c, p.name) for c, p in found] == [("debug_run", "ep_0000"), ("debug_run", "ep_0001")]


def test_a_study_layout_wins_over_stray_flat_episodes(tmp_path):
    _write_trace(tmp_path / "traces" / "180000", 0)
    _write_trace(tmp_path, 7)                          # should be ignored

    assert [p.name for _c, p in st.episode_traces(tmp_path)] == ["ep_0000"]
