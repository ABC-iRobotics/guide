import numpy as np
import pytest

cr = pytest.importorskip("block_bin.compare_runs")
rt = pytest.importorskip("block_bin.rollout_trace")


def trace(steps, target="/blocks/red_block", objects=None, summary=None, horizon=10):
    return {
        "directory": None,
        "meta": {
            "target": target,
            "objects_at_start": objects
            or {"red_block": [0.0, 0.0, 1.0], "left_bin": [0.25, -0.4], "right_bin": [0.25, 0.5]},
            "n_action_steps": horizon,
            "summary": summary or {},
        },
        "steps": steps,
    }


def walk(positions, grips=None, rotvecs=None, period=0.2):
    """A rollout that moves through `positions`, optionally closing the gripper."""
    grips = grips if grips is not None else [0.04] * len(positions)
    rotvecs = rotvecs if rotvecs is not None else [[np.pi, 0, 0]] * len(positions)
    return [
        {
            "step": i,
            "eef_position": list(p),
            "grip_command": g,
            "eef_rotvec": list(r),
            "sim_seconds": round(i * period, 4),
        }
        for i, (p, g, r) in enumerate(zip(positions, grips, rotvecs))
    ]


def test_the_block_position_holds_between_polls():
    steps = walk([[0, 0, 1]] * 4)
    steps[0]["poses"] = {"red_block": [1.0, 0.0, 1.0]}

    track = cr.target_track(trace(steps))

    assert track[3].tolist() == [1.0, 0.0, 1.0]


def test_a_block_that_moves_is_followed_not_assumed():
    """Using the start position would score a knocked-aside block as still in place."""
    steps = walk([[0, 0, 1]] * 4)
    steps[2]["poses"] = {"red_block": [0.5, 0.0, 1.0]}

    track = cr.target_track(trace(steps))

    assert track[1].tolist() == [0.0, 0.0, 1.0]
    assert track[3].tolist() == [0.5, 0.0, 1.0]


def test_no_target_means_no_track():
    assert cr.target_track(trace(walk([[0, 0, 1]]), target="")) is None


def test_the_closest_approach_is_found():
    steps = walk([[0.3, 0, 1], [0.1, 0, 1], [0.02, 0, 1], [0.4, 0, 1]])

    metrics = cr.rollout_metrics(trace(steps))

    assert metrics["closest"] == pytest.approx(0.02)
    assert metrics["closest_step"] == 2


def test_lateness_is_the_gap_between_arriving_and_closing():
    """The whole diagnosis: aim is `closest`, timing is this gap."""
    steps = walk(
        [[0.3, 0, 1], [0.02, 0, 1], [0.2, 0, 1], [0.4, 0, 1]],
        grips=[0.04, 0.04, 0.04, 0.01],
    )

    metrics = cr.rollout_metrics(trace(steps))

    assert metrics["closest_step"] == 1
    assert metrics["grasp"] == 3
    assert metrics["late"] == 2
    assert metrics["at_close"] == pytest.approx(0.4)


def test_a_rollout_that_never_closes_has_no_lateness():
    metrics = cr.rollout_metrics(trace(walk([[0.1, 0, 1]] * 3)))

    assert metrics["grasp"] is None
    assert metrics["late"] is None


def test_the_gripper_state_at_the_closest_approach_is_kept():
    """Wide open at the moment it was on the block is the signature of a late close."""
    steps = walk([[0.3, 0, 1], [0.02, 0, 1]], grips=[0.04, 0.0395])

    assert cr.rollout_metrics(trace(steps))["grip_at_closest"] == pytest.approx(0.0395)


def test_tilt_is_reported_at_both_moments():
    """Only measuring tilt at the close blames orientation for a miss that already
    happened."""
    turned = [np.pi, 0, 0]
    steps = walk(
        [[0.02, 0, 1], [0.3, 0, 1]],
        grips=[0.04, 0.01],
        rotvecs=[turned, [2.2, 0, 2.2]],
    )

    metrics = cr.rollout_metrics(trace(steps))

    assert metrics["tilt_at_closest"] == pytest.approx(0.0)
    assert metrics["tilt_at_close"] > 45


def test_the_control_period_is_measured_not_assumed():
    assert cr.rollout_metrics(trace(walk([[0, 0, 1]] * 5, period=0.5)))["period"] == pytest.approx(
        0.5
    )


def test_an_empty_rollout_reduces_to_nothing():
    assert cr.rollout_metrics(trace([])) == {}


def leaning(towards):
    """A rollout whose tool ends up on one bin."""
    bins = {"left": [0.25, -0.4, 1.0], "right": [0.25, 0.5, 1.0]}
    return trace(walk([[0.25, 0.0, 1.0], bins[towards]]))


def test_following_both_instructions_is_a_positive_shift():
    shift = cr.pair_shift({"left": leaning("left"), "right": leaning("right")})

    assert shift > 0.05


def test_moving_the_same_way_regardless_scores_zero():
    """A resting bias must not read as half-grounded."""
    shift = cr.pair_shift({"left": leaning("right"), "right": leaning("right")})

    assert shift == pytest.approx(0.0)


def test_a_missing_trial_is_not_a_shift():
    assert np.isnan(cr.pair_shift({"left": leaning("left")}))


def test_the_table_renders_a_row_per_run():
    summary = {
        "run": "horizon_5", "horizon": 5, "rollouts": 8, "pairs": 4, "followed": 4,
        "shift": 0.4, "period": 0.2, "closest": 0.05, "at_close": 0.1, "late": 3.0,
        "tilt_at_closest": 8.0, "tilt_at_close": 20.0, "grip_at_closest": 0.04,
        "lift": 0.12, "placed": 1, "claimed": 1, "wrong_bin": 0,
    }

    rendered = cr.table([summary])

    assert "horizon_5" in rendered
    assert "4/4" in rendered


def test_a_run_directory_is_summarised_end_to_end(tmp_path):
    """Through the real trace files, so the reader and the metrics agree."""
    for side, path in (("left", [[0.3, 0, 1], [0.02, 0, 1], [0.3, 0, 1]]),
                       ("right", [[0.3, 0, 1], [0.3, 0.5, 1], [0.3, 0.5, 1]])):
        writer = rt.RolloutTrace(tmp_path / "seed_0000" / side, cameras=())
        writer.open({
            "target": "/blocks/red_block", "n_action_steps": 5,
            "objects_at_start": {
                "red_block": [0.0, 0.0, 1.0], "left_bin": [0.25, -0.4, 1.0],
                "right_bin": [0.25, 0.5, 1.0],
            },
        })
        for i, position in enumerate(path):
            writer.step(i, {"eef_position": position, "grip_command": 0.04 if i < 2 else 0.01,
                            "eef_rotvec": [np.pi, 0, 0], "sim_seconds": 0.2 * i})
        writer.close({"reached": "neither"})

    summary = cr.summarise(tmp_path)

    assert summary["rollouts"] == 2
    assert summary["pairs"] == 1
    assert summary["horizon"] == 5
    assert summary["placed"] == 0


def test_the_columns_line_up():
    """Two runs whose horizons differ in width once ran together as '14/4'."""
    base = {
        "rollouts": 8, "pairs": 4, "followed": 4, "shift": 0.4, "period": 0.2,
        "closest": 0.05, "at_close": 0.1, "late": 3.0, "tilt_at_closest": 8.0,
        "tilt_at_close": 20.0, "grip_at_closest": 0.04, "lift": 0.12, "placed": 1,
        "claimed": 1, "wrong_bin": 0,
    }
    rendered = cr.table([
        base | {"run": "horizon_1", "horizon": 1},
        base | {"run": "horizon_10", "horizon": 10},
    ]).splitlines()

    header, first, second = rendered[0], rendered[2], rendered[3]
    assert len(first) == len(header) == len(second)
    assert first.index("4/4") == second.index("4/4")


def rising(heights, distance=0.02):
    """A rollout whose target block moves up by `heights`."""
    steps = walk([[distance, 0, 1.0]] * len(heights), grips=[0.01] * len(heights))
    for step, height in zip(steps, heights):
        step["poses"] = {"red_block": [0.0, 0.0, 1.0 + height]}
    return steps


def test_a_block_that_never_left_the_table_was_pushed_not_placed():
    """Observed for real: bin shoved 0.152 m, block dragged 0.459 m, peak lift 0.0155 m,
    fingers closing through to -0.022 m on air -- and the scene called it a success."""
    metrics = cr.rollout_metrics(trace(rising([0.0, 0.008, 0.0155])))

    assert metrics["lift"] == pytest.approx(0.0155)
    assert metrics["lift"] < cr.LIFT_MIN


def test_a_real_pick_clears_the_threshold():
    assert cr.rollout_metrics(trace(rising([0.0, 0.09, 0.15])))["lift"] > cr.LIFT_MIN


def test_a_pushed_block_in_a_bin_is_not_counted_as_placed(tmp_path):
    for side in ("left", "right"):
        writer = rt.RolloutTrace(tmp_path / "seed_0000" / side, cameras=())
        writer.open({
            "target": "/blocks/red_block", "n_action_steps": 4,
            "objects_at_start": {
                "red_block": [0.0, 0.0, 1.0], "left_bin": [0.25, -0.4, 1.0],
                "right_bin": [0.25, 0.5, 1.0],
            },
        })
        for i in range(3):
            writer.step(i, {
                "eef_position": [0.02, 0.0, 1.0], "grip_command": 0.01,
                "eef_rotvec": [np.pi, 0, 0], "sim_seconds": 0.2 * i,
                "poses": {"red_block": [0.0, 0.0, 1.0 + 0.005 * i]},
            })
        # The scene says it landed in a bin; the block never rose.
        writer.close({"reached": "left"})

    summary = cr.summarise(tmp_path)

    assert summary["claimed"] == 2
    assert summary["placed"] == 0


def test_the_table_says_so_when_the_criterion_was_fooled():
    summary = {
        "run": "horizon_4", "horizon": 4, "rollouts": 8, "pairs": 4, "followed": 4,
        "shift": 0.8, "period": 0.2, "closest": 0.08, "at_close": 0.1, "late": 6.0,
        "tilt_at_closest": 9.0, "tilt_at_close": 15.0, "grip_at_closest": 0.04,
        "lift": 0.01, "placed": 0, "claimed": 1, "wrong_bin": 0,
    }

    rendered = cr.table([summary])

    assert "WARNING" in rendered
    assert "1 claimed, 0 lifted" in rendered


def placed_trace(reached, asked, lift=0.15):
    steps = walk([[0.02, 0, 1.0]] * 3, grips=[0.01] * 3)
    for i, step in enumerate(steps):
        step["poses"] = {"red_block": [0.0, 0.0, 1.0 + lift * i / 2]}
    t = trace(steps, summary={"reached": reached})
    t["meta"]["asked_for"] = asked
    return t


def test_the_bin_it_was_told_is_the_one_that_counts():
    assert cr._placed(cr.rollout_metrics(placed_trace("left", "left")))


def test_the_other_bin_is_not_a_placement():
    """A policy that always goes left would otherwise score perfectly."""
    assert not cr._placed(cr.rollout_metrics(placed_trace("right", "left")))


def test_the_right_bin_without_a_lift_is_still_a_push():
    assert not cr._placed(cr.rollout_metrics(placed_trace("left", "left", lift=0.01)))


def test_wrong_bin_placements_are_counted_separately(tmp_path):
    for side, reached in (("left", "right"), ("right", "right")):
        writer = rt.RolloutTrace(tmp_path / "seed_0000" / side, cameras=())
        writer.open({
            "target": "/blocks/red_block", "asked_for": side, "n_action_steps": 10,
            "objects_at_start": {"red_block": [0.0, 0.0, 1.0], "left_bin": [0.25, -0.4, 1.0],
                                 "right_bin": [0.25, 0.5, 1.0]},
        })
        for i in range(3):
            writer.step(i, {"eef_position": [0.02, 0, 1.0], "grip_command": 0.01,
                            "eef_rotvec": [np.pi, 0, 0], "sim_seconds": 0.2 * i,
                            "poses": {"red_block": [0.0, 0.0, 1.0 + 0.1 * i]}})
        writer.close({"reached": reached})

    summary = cr.summarise(tmp_path)

    assert summary["placed"] == 1
    assert summary["wrong_bin"] == 1


def test_world_object_poses_are_converted_into_the_tool_frame():
    """The policy's state is scene-relative; PoseRequest answers in world. Getting this
    wrong put a metre into every tool-to-block distance."""
    meta = {"scene_origin": [0.0, 0.0, 1.0]}

    assert cr.to_state_frame(meta, [0.25, -0.4, 1.09]).tolist() == pytest.approx([0.25, -0.4, 0.09])


def test_a_trace_without_a_scene_origin_is_left_alone():
    """Older traces recorded the tool in world frame too, so they need no correction."""
    assert cr.to_state_frame({}, [0.25, -0.4, 1.09]).tolist() == pytest.approx([0.25, -0.4, 1.09])


def test_the_conversion_reaches_the_distance_metrics():
    steps = walk([[0.0, 0.0, 0.05]] * 2, grips=[0.04, 0.01])
    steps[0]["poses"] = {"red_block": [0.0, 0.0, 1.025]}   # world
    t = trace(steps)
    t["meta"]["scene_origin"] = [0.0, 0.0, 1.0]

    metrics = cr.rollout_metrics(t)

    # Tool at scene z=0.05, block at scene z=0.025 -> 2.5 cm apart, not 1.0 m.
    assert metrics["closest"] == pytest.approx(0.025, abs=1e-6)
