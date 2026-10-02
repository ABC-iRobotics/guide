"""The sweep's arithmetic and its report, on records no simulator was needed for.

Rolling out is the expensive half and it is ``eval_policy_pink``'s, already tested.
What is worth checking here is everything that happens to the numbers afterwards --
where a checkpoint's identity comes from, what counts as finished, and whether the PDF
renders at all, which is the sort of thing that only fails after a two-hour sweep.
"""

import json
import signal
import subprocess
import threading
from types import SimpleNamespace

import pandas as pd
import pytest

from block_bin import sweep_checkpoints as sweep


def episode(
    policy,
    zone=3,
    success=True,
    seconds=12.0,
    steps=90,
    home=True,
    travelled=1.4,
    grip_commanded_min=0.001,
    grip_measured_min=0.024,
):
    return {
        "max_home_error": 0.7,
        "travelled": travelled,
        "grip_commanded_min": grip_commanded_min,
        "grip_measured_min": grip_measured_min,
        "policy": policy,
        "episode": 0,
        "time": 1.0,
        "zone": zone,
        "task": "put the block in the bin",
        "success": success,
        "succeeded_in_loop": success,
        "returned_home": home,
        "aborted": False,
        "steps": steps,
        "success_step": 60 if success else None,
        "wall_seconds": 40.0,
        "sim_seconds": 18.0,
        "success_seconds": seconds if success else None,
        "median_step_ms": 30.0,
        "replans": 9,
        "unreachable": 0,
        "clamped": 2,
        "held": 0,
        "home_error": 0.1,
        "home_settle_error": 0.002,
    }


@pytest.fixture
def swept(tmp_path):
    """An output directory as a finished two-checkpoint, two-zone sweep leaves it."""
    raw = tmp_path / "raw"
    raw.mkdir()
    scores = (("010000", [False, False, True, False]), ("100000", [True, True, True, False]))
    for name, wins in scores:
        policy = f"/models/run/checkpoints/{name}/pretrained_model"
        lines = [
            episode(policy, zone=3 if index < 2 else 8, success=win)
            for index, win in enumerate(wins)
        ]
        (raw / f"{name}.jsonl").write_text("\n".join(json.dumps(line) for line in lines) + "\n")
    return tmp_path


def test_a_checkpoint_is_identified_from_the_record_not_the_filename(swept):
    # So a hand-run evaluation dropped into raw/ aggregates with the rest.
    (swept / "raw" / "whatever.jsonl").write_text(
        json.dumps(episode("/models/run/checkpoints/050000/pretrained_model")) + "\n"
    )

    frame = sweep.load_records(swept / "raw")

    assert set(frame.checkpoint) == {"010000", "050000", "100000"}
    assert list(frame.step.unique()) == [10000, 50000, 100000]


def test_records_come_back_in_training_order(swept):
    frame = sweep.load_records(swept / "raw")

    assert list(frame.step) == sorted(frame.step)


def test_an_empty_directory_says_so_instead_of_rendering_an_empty_report(tmp_path):
    (tmp_path / "raw").mkdir()

    with pytest.raises(SystemExit, match="Nothing ran|nothing ran"):
        sweep.load_records(tmp_path / "raw")


def test_the_success_rate_is_per_checkpoint(swept):
    summary = sweep.summarize(sweep.load_records(swept / "raw"), ["checkpoint", "step"])

    rates = dict(zip(summary.checkpoint, summary.success_rate))
    assert rates == {"010000": 0.25, "100000": 0.75}


def test_a_failed_episode_does_not_score_as_an_instant_success(swept):
    # success_seconds is None for a failure. Counting it as 0 would make the worst
    # checkpoint look like the fastest one.
    summary = sweep.summarize(sweep.load_records(swept / "raw"), ["checkpoint", "step"])

    assert set(summary.median_success_s) == {12.0}


def test_zones_are_scored_separately(swept):
    summary = sweep.summarize(sweep.load_records(swept / "raw"), ["checkpoint", "zone"])

    assert sorted(summary.zone.unique()) == [3, 8]
    assert list(summary.episodes) == [2, 2, 2, 2]


def test_an_unrestricted_episode_keeps_a_usable_zone(tmp_path):
    # zone is None in the record; NaN would drop the row out of every groupby.
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "010000.jsonl").write_text(
        json.dumps(episode("/models/run/checkpoints/010000/pretrained_model", zone=None)) + "\n"
    )

    frame = sweep.load_records(raw)

    assert list(frame.zone) == [-1]
    assert len(sweep.summarize(frame, ["checkpoint", "zone"])) == 1


@pytest.mark.parametrize(
    "spec,episodes,expected",
    [
        ("", 5, 5),
        ("3", 5, 5),
        ("3,8,12", 5, 15),
        ("3:10,8:4", 5, 14),
        ("3:10,8", 5, 15),
    ],
)
def test_how_many_episodes_a_finished_checkpoint_leaves(spec, episodes, expected):
    assert sweep.expected_episodes(spec, episodes) == expected


def test_the_interval_does_not_collapse_on_a_clean_sweep():
    # The reason this is Wilson: the normal interval is +-0 at 5/5 and would report a
    # five-episode checkpoint as certainly perfect.
    low, high = sweep.wilson(5, 5)

    assert 0.5 < low < 1.0
    assert high == 1.0


def test_the_interval_is_honest_about_a_shutout():
    low, high = sweep.wilson(0, 5)

    assert low == 0.0
    assert 0.0 < high < 0.6


def test_more_episodes_narrow_the_interval():
    few = sweep.wilson(3, 5)
    many = sweep.wilson(30, 50)

    assert (many[1] - many[0]) < (few[1] - few[0])


def scored(rates, episodes=20):
    """A per-checkpoint summary from {checkpoint: success rate}, for the advisers."""
    rows = []
    for index, (name, rate) in enumerate(rates.items()):
        wins = round(rate * episodes)
        low, high = sweep.wilson(wins, episodes)
        rows.append(
            {
                "checkpoint": name,
                "step": (index + 1) * 10000,
                "episodes": episodes,
                "successes": wins,
                "success_rate": wins / episodes,
                "ci_low": low,
                "ci_high": high,
                "held": 0,
                "clamped": 0,
                "unreachable": 0,
                "total_steps": episodes * 100,
            }
        )
    return pd.DataFrame(rows)


def test_the_best_checkpoint_is_the_one_to_ship():
    notes = sweep.advise_checkpoints(scored({"a": 0.2, "b": 0.9, "c": 0.5}))

    assert notes[0].startswith("Ship b")


def test_a_last_checkpoint_that_is_still_the_best_says_train_longer():
    notes = " ".join(sweep.advise_checkpoints(scored({"a": 0.2, "b": 0.5, "c": 0.9})))

    assert "training longer" in notes


def test_a_late_collapse_is_called_a_regression():
    notes = " ".join(sweep.advise_checkpoints(scored({"a": 0.2, "b": 0.95, "c": 0.15})))

    assert "regression" in notes
    assert "do not ship the final checkpoint" in notes


def test_training_past_an_indistinguishable_checkpoint_is_flagged():
    # b and c are 65% and 70% over 20 episodes -- overlapping intervals, so the extra
    # 10k steps that produced c bought nothing this sweep can see.
    notes = " ".join(sweep.advise_checkpoints(scored({"a": 0.1, "b": 0.65, "c": 0.7})))

    assert "indistinguishable" in notes


def test_a_zone_nobody_ever_solved_is_blamed_on_the_scene_not_the_model():
    per_zone = pd.DataFrame(
        [
            {"checkpoint": "a", "zone": 3, "episodes": 10, "successes": 6},
            {"checkpoint": "a", "zone": 8, "episodes": 10, "successes": 0},
        ]
    )

    notes = " ".join(sweep.advise_zones(per_zone, overall=0.3))

    assert "reachability or randomizer" in notes


def test_a_significantly_worse_zone_asks_for_demonstrations():
    per_zone = pd.DataFrame(
        [
            {"checkpoint": "a", "zone": 3, "episodes": 40, "successes": 36},
            {"checkpoint": "a", "zone": 8, "episodes": 40, "successes": 12},
        ]
    )

    notes = " ".join(sweep.advise_zones(per_zone, overall=0.6))

    assert "Collect more demonstrations" in notes


def test_even_coverage_says_so_rather_than_inventing_a_gap():
    per_zone = pd.DataFrame(
        [
            {"checkpoint": "a", "zone": 3, "episodes": 20, "successes": 12},
            {"checkpoint": "a", "zone": 8, "episodes": 20, "successes": 11},
        ]
    )

    notes = " ".join(sweep.advise_zones(per_zone, overall=0.575))

    assert "No zone is significantly worse" in notes


def test_held_steps_are_called_a_pipeline_fault_not_a_policy_one():
    summary = scored({"a": 0.5})
    summary.loc[0, "held"] = 50  # of 2000 steps, i.e. 2.5%

    notes = " ".join(sweep.advise_motion(summary))

    assert "not a policy problem" in notes


def test_heavy_clamping_points_at_fps_first():
    summary = scored({"a": 0.5})
    summary.loc[0, "clamped"] = 400  # of 2000 steps

    notes = " ".join(sweep.advise_motion(summary))

    assert "--fps" in notes


def test_a_clean_pipeline_sends_you_back_to_the_data():
    notes = " ".join(sweep.advise_motion(scored({"a": 0.3})))

    assert "More or better data" in notes


def test_overlapping_checkpoints_are_reported_as_unrankable():
    notes = " ".join(sweep.advise_power(scored({"a": 0.6, "b": 0.7})))

    assert "CANNOT be told apart" in notes


def test_a_tie_is_reported_as_a_tie_not_as_zero_more_episodes():
    # Equal rates make the sample-size formula divide by zero; "about 0 episodes each"
    # is the nonsense that produced this test.
    notes = " ".join(sweep.advise_power(scored({"a": 0.7, "b": 0.7})))

    assert "IDENTICALLY" in notes
    assert "about 0 episodes" not in notes


def test_a_handful_of_held_steps_is_not_worth_an_alarm():
    summary = scored({"a": 0.5})
    summary.loc[0, "held"] = 4  # of 2000 steps

    assert "HELD" not in " ".join(sweep.advise_motion(summary))


def test_a_clear_winner_is_reported_as_one():
    notes = " ".join(sweep.advise_power(scored({"a": 0.05, "b": 0.95})))

    assert "beats" in notes


def test_the_episodes_needed_grow_as_the_gap_shrinks():
    assert sweep.episodes_to_separate(0.5, 0.9) < sweep.episodes_to_separate(0.5, 0.6)
    assert sweep.episodes_to_separate(0.5, 0.5) == 0


@pytest.mark.parametrize(
    "kwargs,mode",
    [
        ({"travelled": 0.02}, "never moved"),
        ({"grip_commanded_min": 0.04}, "never closed"),
        ({"grip_measured_min": 0.001}, "closed on nothing"),
        ({}, "grasped, not placed"),
    ],
)
def test_a_failure_is_described_by_what_the_arm_was_doing(kwargs, mode):
    frame = pd.DataFrame([episode("/m/checkpoints/010000/pretrained_model", success=False,
                                  **kwargs)])

    assert list(sweep.with_failure_modes(frame).failure_mode) == [mode]


def test_a_success_gets_no_failure_mode(swept):
    frame = sweep.with_failure_modes(sweep.load_records(swept / "raw"))

    assert (frame.loc[frame.success, "failure_mode"] == "").all()
    assert (frame.loc[~frame.success, "failure_mode"] != "").all()


def test_an_arm_that_never_moved_is_not_also_accused_of_missing_its_grasp():
    # Ordered, not scored: a rollout that never left home has no grasp to have missed.
    row = pd.DataFrame(
        [episode("/m/checkpoints/010000/pretrained_model", success=False, travelled=0.01,
                 grip_commanded_min=0.04, grip_measured_min=0.0)]
    )

    assert list(sweep.with_failure_modes(row).failure_mode) == ["never moved"]


def test_a_run_recorded_before_the_telemetry_existed_still_reports(swept):
    # Old raw/ files have no travelled or grip columns; the failure page is skipped
    # rather than the whole report dying.
    frame = sweep.load_records(swept / "raw").drop(columns=["travelled"])

    assert list(sweep.with_failure_modes(frame).failure_mode.unique()) == [""]
    assert sweep.advise_failures(sweep.with_failure_modes(frame)) == []


def test_the_dominant_failure_mode_comes_with_its_advice(swept):
    frame = sweep.with_failure_modes(sweep.load_records(swept / "raw"))

    notes = " ".join(sweep.advise_failures(frame))

    assert "Dominant mode" in notes
    assert "grasped, not placed" in notes


def test_the_score_so_far_is_read_back_off_disk(swept):
    assert sweep.checkpoint_score(swept / "raw" / "010000.jsonl") == (1, 4)
    assert sweep.checkpoint_score(swept / "raw" / "100000.jsonl") == (3, 4)


def test_a_checkpoint_that_never_ran_scores_nothing(tmp_path):
    assert sweep.checkpoint_score(tmp_path / "missing.jsonl") == (0, 0)


def test_a_complete_shutout_stops_the_sweep():
    assert sweep.is_hopeless(wins=0, ran=15, wanted=15)


def test_one_win_anywhere_keeps_it_going():
    # "for all of the designated zones" -- a single success in a single zone is enough.
    assert not sweep.is_hopeless(wins=1, ran=15, wanted=15)


def test_a_timed_out_child_does_not_stop_the_sweep():
    # 0/2 because the child was killed, not because the checkpoint is bad. Stopping
    # here would discard every earlier checkpoint over an infrastructure problem.
    assert not sweep.is_hopeless(wins=0, ran=2, wanted=15)


def test_a_checkpoint_that_never_started_does_not_stop_the_sweep():
    assert not sweep.is_hopeless(wins=0, ran=0, wanted=15)


def test_the_newest_checkpoint_is_evaluated_first(tmp_path):
    for name in ("010000", "050000", "100000"):
        (tmp_path / name / "pretrained_model").mkdir(parents=True)

    order = [path.name for path in reversed(sweep.discover_checkpoints(tmp_path))]

    assert order == ["100000", "050000", "010000"]


def test_last_is_not_evaluated_twice(tmp_path):
    for name in ("010000", "020000"):
        (tmp_path / name / "pretrained_model").mkdir(parents=True)
    (tmp_path / "last").symlink_to(tmp_path / "020000")

    found = sweep.discover_checkpoints(tmp_path)

    assert [path.name for path in found] == ["010000", "020000"]


def test_the_sweep_walks_back_from_the_newest_and_stops_at_the_first_shutout(
    tmp_path, monkeypatch
):
    """The whole loop, with every checkpoint already recorded so nothing is launched."""
    checkpoints, output = tmp_path / "ckpt", tmp_path / "out"
    (output / "raw").mkdir(parents=True)
    scores = {"010000": [True] * 3, "020000": [False] * 3, "030000": [True, False, True]}
    for name, wins in scores.items():
        (checkpoints / name / "pretrained_model").mkdir(parents=True)
        policy = f"{checkpoints}/{name}/pretrained_model"
        (output / "raw" / f"{name}.jsonl").write_text(
            "\n".join(json.dumps(episode(policy, success=win)) for win in wins) + "\n"
        )

    monkeypatch.setattr(
        sweep, "run_checkpoint", lambda *a, **k: pytest.fail("nothing should have been launched")
    )
    monkeypatch.setattr(
        "sys.argv",
        ["sweep", "--checkpoints", str(checkpoints), "--output", str(output),
         "--zones", "3", "--episodes", "3"],
    )
    sweep.main()

    # 030000 has a win so the walk continues; 020000 is a complete shutout and ends it,
    # leaving 010000 -- the best of the three, and deliberately so -- unevaluated.
    summary = json.loads((output / "summary.json").read_text())
    assert summary["run"]["stopped_at"] == "020000"


def test_a_directory_without_checkpoints_is_an_error_not_an_empty_sweep(tmp_path):
    with pytest.raises(SystemExit, match="pretrained_model"):
        sweep.discover_checkpoints(tmp_path)


def test_the_whole_report_renders(swept):
    meta = {
        "checkpoints_root": "/models/run/checkpoints",
        "namespace": "/Sim_0/Scene_0",
        "zones": "3,8",
        "episodes": 2,
        "seconds": 60.0,
        "extra": [],
        "finished": "2026-08-04 10:00",
    }

    per_checkpoint, suggestions = sweep.build_outputs(swept, meta)

    assert (swept / "report.pdf").stat().st_size > 10_000
    assert list(per_checkpoint.checkpoint) == ["010000", "100000"]
    assert suggestions["Which checkpoint to ship"][0].startswith("Ship 100000")
    written = pd.read_csv(swept / "episodes.csv")
    assert len(written) == 8
    summary = json.loads((swept / "summary.json").read_text())
    assert summary["run"]["zones"] == "3,8"
    assert len(summary["per_zone"]) == 4
    assert summary["suggestions"]["Policy or pipeline"]


def test_a_single_zone_report_that_stopped_early_renders_too(tmp_path):
    # Two branches at once: the zone heatmap is skipped with one zone, and the cover
    # grows the "stopped early" note. This is the shape a bad model produces, so it is
    # exactly the report that must not crash.
    raw = tmp_path / "raw"
    raw.mkdir()
    lost = episode("/models/run/checkpoints/010000/pretrained_model", success=False)
    (raw / "010000.jsonl").write_text(json.dumps(lost) + "\n")

    sweep.build_outputs(
        tmp_path,
        {
            "checkpoints_root": "x",
            "namespace": "n",
            "zones": "3",
            "episodes": 1,
            "seconds": 60.0,
            "extra": [],
            "finished": "-",
            "stopped_at": "010000",
        },
    )

    assert (tmp_path / "report.pdf").stat().st_size > 10_000


def checkpoint_tree(tmp_path, names):
    """A checkpoints directory the way LeRobot writes one."""
    for name in names:
        (tmp_path / name / sweep.CHECKPOINT_SUBDIR).mkdir(parents=True)
    return sweep.discover_checkpoints(tmp_path)


def test_no_selector_keeps_the_whole_run(tmp_path):
    found = checkpoint_tree(tmp_path, ["005000", "010000"])

    assert sweep.select_checkpoints(found, "") == found
    assert sweep.select_checkpoints(found, "   ") == found


def test_selector_ignores_the_trainers_zero_padding(tmp_path):
    """'5000' has to find '005000'.

    The padding is a formatting choice of the trainer, and a selector that demands it
    back silently matches nothing -- which would look like a sweep that ran.
    """
    found = checkpoint_tree(tmp_path, ["005000", "010000", "015000"])

    picked = sweep.select_checkpoints(found, "5000,15000")

    assert [p.name for p in picked] == ["005000", "015000"]


def test_selector_returns_training_order_whatever_order_it_was_given(tmp_path):
    found = checkpoint_tree(tmp_path, ["005000", "010000", "015000"])

    picked = sweep.select_checkpoints(found, "15000, 5000")

    order = [p.name for p in picked]
    assert order == ["005000", "015000"], "report order must not depend on --only"


def test_selector_names_one_checkpoint_once(tmp_path):
    # 5000 and 005000 are the same directory; evaluating it twice would plot it twice.
    found = checkpoint_tree(tmp_path, ["005000", "010000"])

    assert [p.name for p in sweep.select_checkpoints(found, "5000,005000")] == ["005000"]


def test_an_unknown_checkpoint_stops_the_sweep(tmp_path):
    """Skipping it silently would evaluate less than asked and still look complete."""
    found = checkpoint_tree(tmp_path, ["005000", "010000"])

    with pytest.raises(SystemExit, match="99999"):
        sweep.select_checkpoints(found, "5000,99999")


def test_nothing_recorded_is_not_a_report(tmp_path):
    # A sweep stopped by hand can end before the first episode finishes; every
    # aggregate downstream assumes at least one row.
    raw = tmp_path / "raw"
    raw.mkdir()
    assert sweep.recorded_episodes(raw) == 0
    assert sweep.recorded_episodes(tmp_path / "absent") == 0

    (raw / "005000.jsonl").write_text("\n")  # a child that wrote nothing
    assert sweep.recorded_episodes(raw) == 0

    (raw / "010000.jsonl").write_text(json.dumps(episode("010000")) + "\n")
    assert sweep.recorded_episodes(raw) == 1


class FakeChild:
    """A child that never exits on its own, so the poll loop has to end it."""

    def __init__(self):
        self.signals = []
        self.returncode = None

    def wait(self, timeout=None):
        if self.signals:
            self.returncode = -2
            return self.returncode
        raise subprocess.TimeoutExpired("child", timeout)

    def send_signal(self, sig):
        self.signals.append(sig)

    def kill(self):
        self.signals.append(signal.SIGKILL)


def sweep_args(tmp_path, timeout=600.0):
    return SimpleNamespace(
        namespace="/Sim_0/Scene_0",
        episodes=1,
        seconds=10.0,
        seed_base=0,
        zones="",
        timeout=timeout,
    )


def run_with_fake_child(tmp_path, monkeypatch, stop, timeout=600.0):
    child = FakeChild()
    monkeypatch.setattr(sweep.subprocess, "Popen", lambda *a, **k: child)
    status = sweep.run_checkpoint(
        tmp_path / "005000",
        tmp_path / "results.jsonl",
        tmp_path / "log.txt",
        sweep_args(tmp_path, timeout),
        [],
        stop,
    )
    return child, status


def test_a_stop_request_interrupts_the_running_checkpoint(tmp_path, monkeypatch):
    """The point of polling instead of subprocess.run(timeout=...).

    Waiting for the child to finish would make the service useless on exactly the
    run worth stopping -- a checkpoint with half an hour of episodes left. SIGINT is
    the signal eval_policy_pink already unwinds cleanly on, and the episodes it
    finished are on disk already.
    """
    stop = threading.Event()
    stop.set()

    child, status = run_with_fake_child(tmp_path, monkeypatch, stop)

    assert status == "stopped on request"
    assert child.signals == [signal.SIGINT], "a stop must not SIGKILL a child mid-write"


def test_a_wedged_checkpoint_is_still_killed(tmp_path, monkeypatch):
    # The timeout is a hang detector and has to survive the rewrite that added --stop.
    child, status = run_with_fake_child(tmp_path, monkeypatch, threading.Event(), timeout=0.0)

    assert status.startswith("TIMED OUT")
    assert child.signals == [signal.SIGKILL]


def test_a_checkpoint_nobody_stopped_is_left_alone(tmp_path, monkeypatch):
    child = FakeChild()
    child.returncode = 0
    monkeypatch.setattr(sweep.subprocess, "Popen", lambda *a, **k: child)
    monkeypatch.setattr(FakeChild, "wait", lambda self, timeout=None: 0)

    status = sweep.run_checkpoint(
        tmp_path / "005000",
        tmp_path / "results.jsonl",
        tmp_path / "log.txt",
        sweep_args(tmp_path),
        [],
        threading.Event(),
    )

    assert status == "ok"
    assert child.signals == []


def test_last_selects_the_newest_checkpoint(tmp_path):
    """`last` is LeRobot's symlink, and discovery resolves the name away.

    It is still the selection people reach for most -- "score the one I would ship" --
    so the selector offers it back rather than making them look the number up.
    """
    found = checkpoint_tree(tmp_path, ["005000", "010000", "015000"])

    assert [p.name for p in sweep.select_checkpoints(found, "last")] == ["015000"]
    picked = sweep.select_checkpoints(found, "5000,last")
    assert [p.name for p in picked] == ["005000", "015000"]


def test_a_hand_stopped_report_says_so_rather_than_blaming_the_model(tmp_path):
    """"Scored 0 everywhere" is a verdict; "the operator stopped it" is not.

    Both leave the earlier checkpoints unevaluated, so a cover that cannot tell them
    apart invites reading an unfinished sweep as a finding about the policy.
    """
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "010000.jsonl").write_text(
        json.dumps(episode("/models/run/checkpoints/010000/pretrained_model")) + "\n"
    )

    sweep.build_outputs(
        tmp_path,
        {
            "checkpoints_root": "x",
            "namespace": "/Sim_0/Scene_0",
            "zones": "3",
            "episodes": 1,
            "seconds": 60.0,
            "extra": [],
            "finished": "-",
            "stopped_at": "010000",
            "stopped_reason": "stop requested through /Sim_0/Scene_0/stop_sweep",
        },
    )

    assert (tmp_path / "report.pdf").stat().st_size > 10_000
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert "stop requested" in summary["run"]["stopped_reason"]


def sweep_over(tmp_path, monkeypatch, run_checkpoint, episodes=3):
    """Drive main() over three empty checkpoints with a stubbed child, and count launches."""
    checkpoints, output = tmp_path / "ckpt", tmp_path / "out"
    (output / "raw").mkdir(parents=True)
    for name in ("010000", "020000", "030000"):
        (checkpoints / name / "pretrained_model").mkdir(parents=True)

    launched = []

    def launch(checkpoint, results, *a, **k):
        launched.append(checkpoint.name)
        return run_checkpoint(checkpoint, results)

    monkeypatch.setattr(sweep, "run_checkpoint", launch)
    monkeypatch.setattr(
        "sys.argv",
        ["sweep", "--checkpoints", str(checkpoints), "--output", str(output),
         "--zones", "3", "--episodes", str(episodes)],
    )
    sweep.main()
    return launched, output


def test_a_child_that_records_nothing_stops_the_sweep(tmp_path, monkeypatch):
    """A misconfigured simulator greets every checkpoint identically -- one is enough.

    This is the eval-without-camera-topics case: the child dies before its first
    episode, so there is no score to read and no reason to think the next checkpoint
    fares better. Without the guard the sweep pays a 450M-parameter load per
    checkpoint to reprint the same traceback, then writes an empty report.
    """
    launched, _ = sweep_over(tmp_path, monkeypatch, lambda ckpt, results: "exited 1")

    assert launched == ["030000"]


def test_a_child_that_recorded_something_does_not_stop_the_sweep(tmp_path, monkeypatch):
    """A timeout mid-checkpoint is not a setup failure: the episodes prove it ran."""

    def half_a_checkpoint(checkpoint, results):
        policy = f"{checkpoint}/pretrained_model"
        results.write_text(json.dumps(episode(policy, success=True)) + "\n")
        return "TIMED OUT after 600s"

    launched, output = sweep_over(tmp_path, monkeypatch, half_a_checkpoint)

    assert launched == ["030000", "020000", "010000"]
    assert json.loads((output / "summary.json").read_text())["run"]["stopped_reason"] == ""


def test_the_setup_failure_is_named_in_the_report(tmp_path, monkeypatch):
    """One recorded episode elsewhere means the report renders and must explain itself."""

    def barren_after_the_first(checkpoint, results):
        if checkpoint.name == "030000":
            results.write_text(json.dumps(episode(f"{checkpoint}/pretrained_model")) + "\n")
            return "ok"
        return "exited 1"

    launched, output = sweep_over(tmp_path, monkeypatch, barren_after_the_first, episodes=1)

    assert launched == ["030000", "020000"]
    reason = json.loads((output / "summary.json").read_text())["run"]["stopped_reason"]
    assert "setup" in reason and "exited 1" in reason


def test_a_dead_context_does_not_take_the_report_with_it():
    """Reproduces the failure that lost a finished 200-episode campaign's report:
    rclpy.shutdown() raises RCLError when the context is already down, and it was
    raising out of a context manager whose data was entirely intact."""
    rclpy = pytest.importorskip("rclpy")

    with sweep.stop_service("/test_teardown"):
        rclpy.shutdown()  # something else tore the context down first

    assert not rclpy.ok()


def test_the_stop_service_still_signals_before_that():
    rclpy = pytest.importorskip("rclpy")

    with sweep.stop_service("/test_signal") as stop:
        assert not stop.is_set()

    if rclpy.ok():
        rclpy.shutdown()


def run_with_stall(tmp_path, monkeypatch, stall, results_grow=False):
    """Drive the poll loop with a child that never exits, watching the stall guard."""
    child = FakeChild()
    monkeypatch.setattr(sweep.subprocess, "Popen", lambda *a, **k: child)
    results = tmp_path / "results.jsonl"
    args = sweep_args(tmp_path, timeout=600.0)
    args.stall = stall

    if results_grow:
        # A slow but working child: one more episode lands on every poll.
        calls = {"n": 0}

        def growing(_path):
            calls["n"] += 1
            return (0, calls["n"])

        monkeypatch.setattr(sweep, "checkpoint_score", growing)

    status = sweep.run_checkpoint(
        tmp_path / "005000", results, tmp_path / "log.txt", args, [], None)
    return child, status


def test_a_child_that_finishes_no_episodes_is_killed_as_stalled(tmp_path, monkeypatch):
    """The guard that replaces guessing a per-episode cost."""
    _child, status = run_with_stall(tmp_path, monkeypatch, stall=0.001)

    assert "STALLED" in status


def test_a_slow_child_that_keeps_finishing_episodes_is_left_alone(tmp_path, monkeypatch):
    """Slow is not hung. Killing this is how the earlier campaigns lost episodes."""
    child = FakeChild()
    monkeypatch.setattr(sweep.subprocess, "Popen", lambda *a, **k: child)
    calls = {"n": 0}

    def growing(_path):
        calls["n"] += 1
        if calls["n"] > 3:
            child.send_signal(signal.SIGTERM)      # let wait() return, ending the loop
        return (0, calls["n"])

    monkeypatch.setattr(sweep, "checkpoint_score", growing)
    args = sweep_args(tmp_path, timeout=600.0)
    args.stall = 0.001                              # would fire instantly without progress

    status = sweep.run_checkpoint(
        tmp_path / "005000", tmp_path / "results.jsonl", tmp_path / "log.txt",
        args, [], None)

    assert "STALLED" not in status


def test_without_a_stall_setting_the_loop_behaves_as_before(tmp_path, monkeypatch):
    """Callers that never heard of --stall still get the elapsed-time backstop."""
    _child, status = run_with_fake_child(tmp_path, monkeypatch, threading.Event(),
                                         timeout=0.0)

    assert "TIMED OUT" in status
