"""Tests for the combined validation tool."""

import json

import pandas as pd
import pytest

from block_bin import validate as v


def a_trace(steps, meta=None, summary=None):
    """A trace dict shaped the way load_trace returns one."""
    base = {"episode": 0, "zone": 1, "seed": 0,
            "summary": {"task": "Put the red block in the left bin.",
                        "goal": "/bin_0", "target": "/Scene_0/blocks/red_block"}}
    if summary is not None:
        base["summary"].update(summary)
    if meta:
        base.update(meta)
    return {"directory": "", "meta": base, "steps": steps}


def poses(**cubes):
    """Bins at the recorded scene offsets; cubes wherever the test puts them."""
    out = {"left_bin": [0.25, -0.4, 1.09], "right_bin": [0.25, 0.4, 1.09]}
    out.update({k: list(vv) for k, vv in cubes.items()})
    return out


def test_a_cube_inside_the_bin_counts_as_delivered():
    trace = a_trace([{"step": 0, "poses": poses(red_block=[0.25, -0.4, 1.046])}])

    assert v.contents_geometric(trace)["red_block"] == "left"


def test_a_cube_resting_against_the_outer_wall_does_not():
    """The sim's own check is a collision test, which fires for this.

    Measured offsets are bimodal: inside is |dy| <= 0.09, against the wall >= 0.165.
    """
    trace = a_trace([{"step": 0, "poses": poses(red_block=[0.246, -0.221, 1.041])}])

    assert v.contents_geometric(trace)["red_block"] is None


def test_a_cube_left_on_the_table_is_in_no_bin():
    trace = a_trace([{"step": 0, "poses": poses(red_block=[0.0, 0.0, 1.025])}])

    assert v.contents_geometric(trace)["red_block"] is None


def test_containment_uses_the_last_polled_step():
    """Cubes move; only where they end up counts."""
    trace = a_trace([
        {"step": 0, "poses": poses(red_block=[0.0, 0.0, 1.025])},
        {"step": 9, "poses": poses(red_block=[0.25, -0.4, 1.046])},
    ])

    assert v.contents_geometric(trace)["red_block"] == "left"


def lift_and_place(cube, where, target="/Scene_0/blocks/red_block"):
    """Steps that grasp, raise `cube` clear of the table, and set it in `where`."""
    start = poses(**{cube: [0.0, 0.0, 1.025]})
    high = poses(**{cube: [0.0, 0.0, 1.2]})
    end = poses(**{cube: list(where)})
    return [
        {"step": 0, "grip_command": 0.04, "poses": start},
        {"step": 1, "grip_command": 0.01, "grip_measured": 0.02, "poses": start},
        {"step": 2, "grip_command": 0.01, "grip_measured": 0.02, "poses": high},
        {"step": 3, "grip_command": 0.04, "poses": end},
    ]


def test_the_named_cube_in_the_named_bin_is_a_success():
    trace = a_trace(lift_and_place("red_block", [0.25, -0.4, 1.046]))

    result = v.outcome(trace, "180000")

    assert result["success"] is True
    assert result["wrong_cube"] is False
    assert result["took"] is None
    assert result["asked"] == "red"
    assert result["checkpoint"] == "180000"


def test_the_wrong_cube_in_the_named_bin_is_not_a_success():
    trace = a_trace(lift_and_place("yellow_block", [0.25, -0.4, 1.046]))

    result = v.outcome(trace, "180000")

    assert result["success"] is False
    assert result["wrong_cube"] is True
    assert result["took"] == "yellow"


def test_the_named_cube_shoved_against_the_bin_is_not_a_success():
    """The failure the collision test cannot see, and the reason this module exists."""
    trace = a_trace(lift_and_place("red_block", [0.246, -0.221, 1.041]))

    assert v.outcome(trace, "180000")["success"] is False


def test_the_right_cube_in_the_wrong_bin_is_not_a_success():
    trace = a_trace(lift_and_place("red_block", [0.25, 0.4, 1.046]))

    result = v.outcome(trace, "180000")

    assert result["success"] is False
    assert result["wrong_cube"] is False


def test_a_cube_never_lifted_is_not_delivered_however_it_ended_up():
    """Nudged across the table into a bin is not a pick and place."""
    flat = poses(red_block=[0.25, -0.4, 1.025])
    trace = a_trace([{"step": 0, "grip_command": 0.01, "grip_measured": 0.02,
                      "poses": poses(red_block=[0.0, 0.0, 1.025])},
                     {"step": 1, "grip_command": 0.01, "grip_measured": 0.02,
                      "poses": flat}])

    assert v.outcome(trace, "1")["success"] is False


def test_first_lift_step_finds_when_the_cube_leaves_the_table():
    trace = a_trace([{"step": 0, "poses": poses(red_block=[0, 0, 1.025])},
                     {"step": 1, "poses": poses(red_block=[0, 0, 1.03])},
                     {"step": 2, "poses": poses(red_block=[0, 0, 1.2])}])

    assert v.first_lift_step(trace, "red_block") == 2


def test_first_lift_step_is_none_when_the_cube_never_rises():
    trace = a_trace([{"step": 0, "poses": poses(red_block=[0, 0, 1.025])}])

    assert v.first_lift_step(trace, "red_block") is None


@pytest.mark.parametrize("world_x, world_y, expected", [
    (0.05, 0.20, 0),      # near row, far column
    (0.05, -0.20, 4),
    (0.35, 0.20, 15),
    (0.05, 0.0, 2),
])
def test_zone_of_matches_the_grid(world_x, world_y, expected):
    assert v.zone_of(world_x, world_y) == expected


def test_zone_of_rejects_points_off_the_grid():
    assert v.zone_of(0.9, 0.0) is None
    assert v.zone_of(0.05, 0.9) is None


def frame_of(rows):
    frame = pd.DataFrame(rows)
    frame["split"] = "in-distribution"
    return frame


def test_the_funnel_counts_each_rung_and_bounds_the_rate():
    frame = frame_of([
        {"checkpoint": "a", "closed": True, "lifted_any": True, "binned_any": True,
         "success": True, "wrong_cube": False, "success_collision": True},
        {"checkpoint": "a", "closed": True, "lifted_any": True, "binned_any": True,
         "success": False, "wrong_cube": True, "success_collision": False},
        {"checkpoint": "a", "closed": True, "lifted_any": False, "binned_any": False,
         "success": False, "wrong_cube": False, "success_collision": False},
    ])

    row = v.funnel(frame).iloc[0]

    assert (row["episodes"], row["closed"], row["lifted_any"]) == (3, 3, 2)
    assert row["success"] == 1
    assert row["ci_low"] < row["rate"] < row["ci_high"]


def test_paired_compares_only_scenes_both_checkpoints_ran():
    frame = frame_of([
        {"checkpoint": "a", "zone": 1, "seed": 0, "success": True},
        {"checkpoint": "a", "zone": 1, "seed": 1, "success": False},
        {"checkpoint": "b", "zone": 1, "seed": 0, "success": False},
        {"checkpoint": "b", "zone": 1, "seed": 1, "success": False},
        {"checkpoint": "b", "zone": 1, "seed": 2, "success": True},   # a never ran this
    ])

    row = v.paired(frame).iloc[0]

    assert row["scenes"] == 2
    assert (row["only_a"], row["only_b"]) == (1, 0)


def test_identity_reports_what_was_taken_instead():
    frame = frame_of([
        {"checkpoint": "a", "zone": 12, "seed": 0, "success": False, "wrong_cube": True,
         "asked": "green", "took": "red", "took_from_zone": 2},
        {"checkpoint": "a", "zone": 12, "seed": 1, "success": False, "wrong_cube": True,
         "asked": "green", "took": "red", "took_from_zone": 2},
        {"checkpoint": "a", "zone": 1, "seed": 2, "success": True, "wrong_cube": False,
         "asked": "red", "took": None, "took_from_zone": None},
    ])

    result = v.identity(frame)

    assert result["episodes"] == 2
    assert result["flow"].iloc[0]["count"] == 2
    assert result["confusion"].iloc[0]["count"] == 2


def test_identity_survives_a_run_with_no_wrong_cube_episodes():
    frame = frame_of([{"checkpoint": "a", "zone": 1, "seed": 0, "success": True,
                       "wrong_cube": False, "asked": "red", "took": None,
                       "took_from_zone": None}])

    assert v.identity(frame)["episodes"] == 0


def test_forensics_names_the_stage_each_failure_reached():
    frame = frame_of([
        {"checkpoint": "a", "zone": 1, "seed": 0, "success": False, "wrong_cube": True,
         "binned_any": True, "lifted_any": True, "closed": True, "asked": "red",
         "took": "blue", "grasp_step": 5, "decisive_step": 9, "trace": ""},
        {"checkpoint": "a", "zone": 1, "seed": 1, "success": False, "wrong_cube": False,
         "binned_any": False, "lifted_any": False, "closed": False, "asked": "red",
         "took": None, "grasp_step": None, "decisive_step": None, "trace": ""},
    ])

    kinds = set(v.forensics(frame)["kind"])

    assert kinds == {"wrong cube", "never grasped"}


def test_a_finished_run_writes_tables_a_summary_and_a_page(tmp_path):
    frame = frame_of([
        {"checkpoint": "180000", "zone": 1, "seed": 0, "closed": True, "lifted_any": True,
         "binned_any": True, "success": True, "wrong_cube": False, "asked": "red",
         "took": None, "took_from_zone": None, "success_collision": True,
         "grasp_step": 4, "decisive_step": 7, "trace": ""},
        {"checkpoint": "180000", "zone": 1, "seed": 1, "closed": True, "lifted_any": True,
         "binned_any": True, "success": False, "wrong_cube": True, "asked": "green",
         "took": "red", "took_from_zone": 2, "success_collision": False,
         "grasp_step": 4, "decisive_step": 7, "trace": ""},
    ])
    meta = {"model": "m", "trained": [1], "held_out": [0]}
    rungs, pairs, ident = v.funnel(frame), v.paired(frame), v.identity(frame)

    v.write_tables(tmp_path, frame, rungs, pairs, ident, v.forensics(frame), meta)
    v.write_html(tmp_path / "validation.html", frame, rungs, pairs, ident, meta)

    saved = json.loads((tmp_path / "validation.json").read_text())
    assert saved["episodes"] == 2
    assert saved["best_in_distribution"]["checkpoint"] == "180000"
    assert saved["scoring"]["containment"] == "geometric"
    assert (tmp_path / "episodes.csv").is_file()
    page = (tmp_path / "validation.html").read_text()
    assert "Validation report" in page and page.count("<svg") == page.count("</svg>")


def test_the_backstop_budget_exceeds_the_measured_cost_of_a_full_run():
    """A full episode measured 470 s wall for 60 s of sim; 100 of them must fit.

    The earlier ceiling assumed 300 s an episode and killed two runs short.
    """
    assert v.budget(100, 60.0) > 100 * 470
    assert v.budget(20, 60.0) < v.budget(100, 60.0)


def test_zone_order_expands_a_spec_into_one_zone_per_episode():
    assert v.zone_order("1:2,7:3") == [1, 1, 7, 7, 7]


def test_remaining_continues_a_part_finished_checkpoint():
    """84 of 100 done over ten zones: zone 12 owes 6 and zone 17 owes all 10."""
    spec, offset = v.remaining("1,2,3,6,7,8,0,4,12,17", 10, 84)

    assert spec == "12:6,17:10"
    assert offset == 84


def test_remaining_shifts_the_seed_so_resumed_scenes_are_not_redrawn():
    """Seeds run straight through the plan, so the offset is the episode count."""
    _spec, offset = v.remaining("1,2", 10, 13)

    assert offset == 13


def test_remaining_on_an_untouched_checkpoint_is_the_whole_plan():
    spec, offset = v.remaining("1,2", 10, 0)

    assert v.zone_order(spec) == [1] * 10 + [2] * 10
    assert offset == 0


def test_remaining_on_a_finished_checkpoint_asks_for_nothing():
    spec, _offset = v.remaining("1,2", 10, 20)

    assert spec == ""


def test_the_resumed_plan_covers_exactly_what_is_missing():
    """Whatever the split point, done plus remaining reconstructs the full plan."""
    whole = v.zone_order(",".join(f"{z}:10" for z in (1, 2, 3, 6, 7, 8, 0, 4, 12, 17)))
    for done in (0, 7, 40, 84, 99):
        spec, _ = v.remaining("1,2,3,6,7,8,0,4,12,17", 10, done)
        assert whole[:done] + v.zone_order(spec) == whole
