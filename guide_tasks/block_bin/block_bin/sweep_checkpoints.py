"""Score every checkpoint of a training run over the same zones, and report on it.

A wrapper around ``eval_policy_pink``, not a second copy of it: every rollout is a
child process running that exact module, so whatever the sweep measures is what a
hand-run evaluation would do. One child per checkpoint (each pays the ~450M-parameter
load once, and a fresh process guarantees no policy state or GPU memory carries
between checkpoints), evaluating every zone in ``--zones``, ``--episodes`` times each.

    ~/ros2_ws/.venv/bin/python -m block_bin.sweep_checkpoints \
        --namespace /Sim_0/Scene_0 --zones 3 --episodes 5

Isaac has to be up with the scene, and the description publisher running, exactly as
for a single evaluation::

    ros2 launch block_bin eval_pink.launch.py

Output, all under ``--output``::

    episodes.csv     one row per episode -- the raw scores, for a spreadsheet or pandas
    summary.json     configuration, aggregates, and the suggestions as structured text
    report.pdf       the same thing as tables and charts, led by a page of suggestions:
                     which checkpoint to ship, which zones are short of data, whether
                     the failures are the policy's or the motion pipeline's, how the
                     failures failed, and whether the ranking survives its own error bars
    raw/<ckpt>.jsonl what eval_policy_pink appended as it went, one JSON per episode
    logs/<ckpt>.log  that child's full console output

Checkpoints are evaluated NEWEST FIRST, and the sweep stops as soon as one of them
scores 0 on a complete evaluation -- every zone, every episode, no wins. Walking
backwards, the remaining checkpoints are earlier in training than a model that cannot
do the task at all, and paying an hour of simulator time to confirm that is the least
interesting hour available. ``--no-early-stop`` runs them anyway, which is what you
want if the run might be non-monotonic or the whole curve is the point.

Each episode is written the moment it finishes, so a sweep that is interrupted -- or
one child that hangs and hits ``--timeout`` -- keeps everything up to that point, and
re-running with the same ``--output`` skips the checkpoints that already have all
their episodes. ``--report-only`` rebuilds the CSV, JSON and PDF from a finished
directory without touching the simulator, which is the cheap way to change a chart.

Timing note: a checkpoint that always fails burns the full ``--seconds`` per episode,
so budget by the worst case, not by how fast a good rollout ends.
"""

import argparse
import json
import math
import subprocess
import sys
import textwrap
import time
from datetime import datetime
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

# What LeRobot writes inside each checkpoint directory, and what --policy wants.
CHECKPOINT_SUBDIR = "pretrained_model"

PAGE = (11.69, 8.27)  # A4 landscape, so the tables have room for their columns


def checkpoint_step(path: Path) -> int:
    """Training step from the directory name; -1 for anything not numbered."""
    return int(path.name) if path.name.isdigit() else -1


def discover_checkpoints(root: Path) -> list[Path]:
    """Every checkpoint under ``root``, in training order.

    ``last`` is a symlink onto the newest numbered checkpoint in every run LeRobot
    writes. Resolving before de-duplicating keeps it from being evaluated twice and
    plotted as if it were a separate model -- the numbered name wins because the
    directory listing is sorted and ``last`` sorts after the digits.
    """
    if not root.is_dir():
        raise SystemExit(f"--checkpoints {root}: no such directory.")
    found: dict[Path, Path] = {}
    for entry in sorted(root.iterdir()):
        if (entry / CHECKPOINT_SUBDIR).is_dir():
            found.setdefault(entry.resolve(), entry)
    if not found:
        raise SystemExit(f"No */{CHECKPOINT_SUBDIR} directories under {root}.")
    return sorted(found.values(), key=checkpoint_step)


def expected_episodes(zones: str, episodes: int) -> int:
    """Rows a finished checkpoint leaves behind. Mirrors ``eval_policy.episode_plan``.

    Only what the sweep needs to decide "did this one finish": a plain list is
    ``--episodes`` per zone, and ``2:4`` overrides the count for that zone.
    """
    if not zones:
        return episodes
    return sum(
        int(count) if count else episodes
        for _, _, count in (item.partition(":") for item in zones.split(","))
    )


def run_checkpoint(checkpoint: Path, results: Path, log: Path, args, extra: list) -> str:
    """One child evaluation. Returns how it ended, for the console line."""
    command = [
        sys.executable,
        "-u",  # so tail -f on the log shows the rollout as it happens
        "-m",
        "block_bin.eval_policy_pink",
        "--policy",
        str(checkpoint / CHECKPOINT_SUBDIR),
        "--namespace",
        args.namespace,
        "--episodes",
        str(args.episodes),
        "--seconds",
        str(args.seconds),
        "--results",
        str(results),
        "--seed-base",
        str(args.seed_base),
        *(["--zone", args.zones] if args.zones else []),
        *extra,
    ]
    log.parent.mkdir(parents=True, exist_ok=True)
    with open(log, "w") as stream:
        stream.write(" ".join(command) + "\n\n")
        stream.flush()
        try:
            finished = subprocess.run(
                command, stdout=stream, stderr=subprocess.STDOUT, timeout=args.timeout
            )
        except subprocess.TimeoutExpired:
            # Killed, not waited on: a wedged child (Isaac stalling, a service that
            # never answers) must not take the rest of the sweep with it. Its finished
            # episodes are already on disk.
            return f"TIMED OUT after {args.timeout:.0f}s"
        except KeyboardInterrupt:
            raise
    return "ok" if finished.returncode == 0 else f"exited {finished.returncode}"


def checkpoint_score(results: Path) -> tuple[int, int]:
    """(successes, episodes) recorded for one checkpoint so far."""
    if not results.is_file():
        return (0, 0)
    rows = [json.loads(line) for line in results.read_text().splitlines() if line.strip()]
    return (sum(bool(row["success"]) for row in rows), len(rows))


def is_hopeless(wins: int, ran: int, wanted: int) -> bool:
    """Whether a checkpoint scored zero on a COMPLETE evaluation.

    Complete matters more than zero does. A checkpoint whose child timed out or was
    aborted after two episodes has also "scored 0", and stopping the sweep on that
    would throw away every earlier checkpoint over what is an infrastructure problem --
    so the whole plan has to have run: every zone, every episode.
    """
    return ran >= wanted and wins == 0


def load_records(raw: Path) -> pd.DataFrame:
    """Every episode of every checkpoint, as one table."""
    rows = []
    for path in sorted(raw.glob("*.jsonl")):
        for line in path.read_text().splitlines():
            if line.strip():
                rows.append(json.loads(line))
    if not rows:
        raise SystemExit(f"No episodes recorded under {raw}. Check logs/ -- nothing ran.")

    frame = pd.DataFrame(rows)
    # Identity comes from the record, not the filename, so a hand-run evaluation
    # dropped into raw/ aggregates the same way.
    directories = [Path(policy).parent for policy in frame["policy"]]
    frame["checkpoint"] = [directory.name for directory in directories]
    frame["step"] = [checkpoint_step(directory) for directory in directories]
    frame["zone"] = frame["zone"].fillna(-1).astype(int)  # -1 = unrestricted placement
    return frame.sort_values(["step", "checkpoint", "episode"]).reset_index(drop=True)


def wilson(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    """95% interval on a success rate, Wilson rather than normal.

    Five episodes per zone is a small sample and the interesting checkpoints sit near
    0 and 1, which is exactly where the textbook +-1.96*sqrt(p(1-p)/n) collapses to
    zero width and claims a 5/5 checkpoint is perfect.
    """
    if total <= 0:
        return (0.0, 0.0)
    rate = successes / total
    denominator = 1 + z * z / total
    centre = (rate + z * z / (2 * total)) / denominator
    spread = z * math.sqrt(rate * (1 - rate) / total + z * z / (4 * total * total)) / denominator
    return (max(0.0, centre - spread), min(1.0, centre + spread))


def summarize(frame: pd.DataFrame, by: list) -> pd.DataFrame:
    """Aggregate scores, grouped however the caller asks."""
    summary = frame.groupby(by, as_index=False, dropna=False).agg(
        episodes=("success", "size"),
        successes=("success", "sum"),
        success_rate=("success", "mean"),
        # Only successful episodes have a time to success, so median skips the NaNs of
        # the failures rather than scoring them as instant.
        median_success_s=("success_seconds", "median"),
        median_steps=("steps", "median"),
        home_return_rate=("returned_home", "mean"),
        median_step_ms=("median_step_ms", "median"),
        held=("held", "sum"),
        clamped=("clamped", "sum"),
        unreachable=("unreachable", "sum"),
        total_steps=("steps", "sum"),
        wall_minutes=("wall_seconds", lambda column: column.sum() / 60.0),
    )
    intervals = [wilson(int(s), int(n)) for s, n in zip(summary.successes, summary.episodes)]
    summary["ci_low"] = [low for low, _ in intervals]
    summary["ci_high"] = [high for _, high in intervals]
    return summary


# The block is a 0.05 m cube (config/randomize.yaml sets its centre at z = 0.025), and
# GRIPPER_OPEN is 0.04. So fingers that stall near the 0.025 m half-width have a block
# between them, and fingers that reach ~0 shut on air. NEVER_MOVED is metres of tool
# path: a rollout that walks less than this has not so much failed the task as declined
# to attempt it.
GRIP_CLOSE_ATTEMPT = 0.030
GRIP_EMPTY = 0.010
NEVER_MOVED = 0.15

FAILURE_MODES = ("never moved", "never closed", "closed on nothing", "grasped, not placed")

# 95% two-sided, 80% power -- the usual pair, and the one behind "how many episodes
# would it take to prove these two checkpoints differ".
Z_CONFIDENCE = 1.96
Z_POWER = 0.84


def failure_mode(row) -> str:
    """Which way one failed episode failed, from the gripper and distance telemetry.

    Deliberately ordered, not scored: an arm that never moved cannot be said to have
    missed its grasp, and one that never commanded a close cannot have closed on
    nothing. The first test that fires is the honest description.
    """
    if row.travelled < NEVER_MOVED:
        return FAILURE_MODES[0]
    if row.grip_commanded_min > GRIP_CLOSE_ATTEMPT:
        return FAILURE_MODES[1]
    if row.grip_measured_min < GRIP_EMPTY:
        return FAILURE_MODES[2]
    return FAILURE_MODES[3]


def with_failure_modes(frame: pd.DataFrame) -> pd.DataFrame:
    """``failure_mode`` per row, blank on successes. Empty if the run predates it."""
    frame = frame.copy()
    if not {"travelled", "grip_commanded_min", "grip_measured_min"} <= set(frame.columns):
        frame["failure_mode"] = ""
        return frame
    usable = frame[["travelled", "grip_commanded_min", "grip_measured_min"]].notna().all(axis=1)
    frame["failure_mode"] = [
        "" if row.success or not ok else failure_mode(row)
        for row, ok in zip(frame.itertuples(), usable)
    ]
    return frame


def episodes_to_separate(first: float, second: float) -> int:
    """Episodes per checkpoint needed to call two success rates apart, 95%/80%.

    The standard two-proportion sample size. It is what turns "these two overlap" into
    a decision: either run that many, or accept that the sweep cannot rank them.
    """
    if first == second:
        return 0
    spread = first * (1 - first) + second * (1 - second)
    return math.ceil((Z_CONFIDENCE + Z_POWER) ** 2 * spread / (first - second) ** 2)


def advise_checkpoints(per_checkpoint: pd.DataFrame) -> list[str]:
    """Which checkpoint to ship, and where training stopped paying."""
    ranked = per_checkpoint.sort_values(["success_rate", "step"], ascending=[False, True])
    best = ranked.iloc[0]
    latest = per_checkpoint.sort_values("step").iloc[-1]
    notes = [
        f"Ship {best.checkpoint}: {best.success_rate:.0%} "
        f"({int(best.successes)}/{int(best.episodes)}), 95% CI "
        f"{best.ci_low:.0%}-{best.ci_high:.0%}."
    ]

    # The earliest checkpoint that could be as good as the best one. Everything after
    # it is training time that bought nothing this sweep can measure.
    reachable = per_checkpoint[per_checkpoint.ci_high >= best.success_rate].sort_values("step")
    if len(reachable) and reachable.iloc[0].step < best.step:
        first = reachable.iloc[0]
        notes.append(
            f"{first.checkpoint} is already statistically indistinguishable from the best "
            f"(its CI reaches {first.ci_high:.0%}). Training past step {first.step:,} bought "
            f"nothing this sweep can measure -- consider stopping there, or evaluating more "
            f"episodes before believing the difference."
        )
    if best.checkpoint == latest.checkpoint and len(per_checkpoint) > 1:
        notes.append(
            "The last checkpoint is the best one, so the curve had not turned over yet: "
            "training longer is worth trying."
        )
    elif latest.ci_high < best.success_rate:
        notes.append(
            f"{latest.checkpoint} is significantly WORSE than {best.checkpoint} "
            f"({latest.success_rate:.0%} against {best.success_rate:.0%}, and its CI tops out "
            f"below it). That is a late regression -- overfitting or the LR schedule -- so do "
            f"not ship the final checkpoint, and shorten the run or add regularisation."
        )
    return notes


def advise_zones(per_zone: pd.DataFrame, overall: float) -> list[str]:
    """Where the dataset is thin, in a form you can go and record."""
    pooled = per_zone.groupby("zone", as_index=False).agg(
        episodes=("episodes", "sum"), successes=("successes", "sum")
    )
    if len(pooled) < 2:
        return []

    notes = []
    for row in pooled.itertuples():
        rate = row.successes / row.episodes
        _, high = wilson(int(row.successes), int(row.episodes))
        if row.successes == 0:
            notes.append(
                f"Zone {row.zone}: 0/{int(row.episodes)} across EVERY checkpoint. A zone no "
                f"checkpoint has ever solved is more likely a reachability or randomizer "
                f"problem than a training one -- check it by hand before collecting data."
            )
        elif high < overall:
            notes.append(
                f"Zone {row.zone}: {rate:.0%} against {overall:.0%} overall, significantly "
                f"worse. Collect more demonstrations there -- this is a coverage gap in the "
                f"dataset, not something more training steps will fix."
            )
    if not notes:
        notes.append(
            f"No zone is significantly worse than the {overall:.0%} pooled rate; coverage "
            f"looks even, so zone-targeted data collection would not be the best next move."
        )
    return notes


def advise_motion(per_checkpoint: pd.DataFrame) -> list[str]:
    """Whether the failures are the policy's or the motion pipeline's."""
    held = 100 * per_checkpoint.held.sum() / max(1, per_checkpoint.total_steps.sum())
    clamped = 100 * per_checkpoint.clamped.sum() / max(1, per_checkpoint.total_steps.sum())
    unreachable = 100 * per_checkpoint.unreachable.sum() / max(1, per_checkpoint.total_steps.sum())

    notes = []
    if held > 0.5:
        # Above half a percent it is systematic rather than the odd step near a
        # singularity, and systematic held steps mean the arm is not being driven at all.
        notes.append(
            f"{held:.1f}% of steps were HELD ({int(per_checkpoint.held.sum())} of "
            f"{int(per_checkpoint.total_steps.sum())}) -- the IK wanted a joint jump and "
            f"nothing was published. This is not a policy problem: check joint_states is live "
            f"and that --base-frame matches the description before reading anything else here."
        )
    if clamped > 5:
        notes.append(
            f"{clamped:.1f}% of steps were clamped: the policy is predicting displacements "
            f"bigger than anything in its training set. Either the checkpoint is "
            f"under-trained, or --fps disagrees with the rate the demonstrations were "
            f"recorded at (each delta spans one control period, so a wrong rate rescales "
            f"every one of them). Check --fps first, it is free."
        )
    if unreachable > 5:
        notes.append(
            f"{unreachable:.1f}% of steps missed the IK tolerance -- targets off the arm's "
            f"reachable set. Usually --action-scale overshooting, or the policy driving into "
            f"a singularity; neither is fixed by more training."
        )
    if not notes:
        notes.append(
            "The motion pipeline is clean (held, clamped and unreachable all low), so the "
            "failures belong to the policy. More or better data, not tuning."
        )
    return notes


def advise_power(per_checkpoint: pd.DataFrame) -> list[str]:
    """Whether the ranking above can be believed at this episode count."""
    if len(per_checkpoint) < 2:
        return []
    ranked = per_checkpoint.sort_values(["success_rate", "step"], ascending=[False, True])
    best, second = ranked.iloc[0], ranked.iloc[1]
    width = best.ci_high - best.ci_low

    if second.ci_high < best.ci_low:
        return [
            f"{best.checkpoint} beats {second.checkpoint} outright -- the intervals do not "
            f"overlap, so the ranking holds at this episode count."
        ]
    needed = episodes_to_separate(best.success_rate, second.success_rate)
    if needed == 0:
        verdict = (
            f"{best.checkpoint} and {second.checkpoint} scored IDENTICALLY "
            f"({best.success_rate:.0%} each) over {int(best.episodes)} episodes. No number of "
            f"episodes separates equal rates, so the tie is the answer: pick on other grounds "
            f"-- the earlier checkpoint is the cheaper one to reproduce."
        )
    else:
        cost = (
            "; that is the honest cost of picking between them."
            if needed < 500
            else ", which is not worth it -- treat them as equivalent and pick on other grounds."
        )
        verdict = (
            f"{best.checkpoint} and {second.checkpoint} ({best.success_rate:.0%} against "
            f"{second.success_rate:.0%}) CANNOT be told apart at {int(best.episodes)} episodes "
            f"-- their intervals overlap. Separating them at 95% would take about {needed} "
            f"episodes each{cost}"
        )
    return [
        verdict,
        f"The best checkpoint's interval is {width:.0%} wide. Every rate on these pages "
        f"carries that much uncertainty, so read the shape of the curve rather than the "
        f"ordering of adjacent points.",
    ]


def advise_failures(frame: pd.DataFrame) -> list[str]:
    """What the failures were doing, from the gripper and distance telemetry."""
    modes = frame.loc[frame.failure_mode != "", "failure_mode"]
    if modes.empty:
        return []
    counts = modes.value_counts()
    total = int(counts.sum())
    dominant = counts.index[0]
    share = counts.iloc[0] / total
    advice = {
        "never moved": "The arm barely left home. That is a policy producing near-zero "
        "deltas -- an under-trained checkpoint, or a state it does not recognise. Check the "
        "first-state line in the logs against the dataset before blaming training.",
        "never closed": "The arm moved but never commanded a close, so it is not finding "
        "the block. This is a perception or approach failure: more demonstrations from "
        "varied starting layouts.",
        "closed on nothing": "The arm reached and closed on empty air -- the approach is "
        "close but the grasp pose is off. Grasp-phase data is what is missing, not more "
        "episodes of the whole task.",
        "grasped, not placed": "The block was grasped and then not placed. The hard part "
        "is working; the transport and release are what need demonstrations.",
    }
    breakdown = ", ".join(f"{count} {name}" for name, count in counts.items())
    return [
        f"{total} failures: {breakdown}.",
        f"Dominant mode is '{dominant}' at {share:.0%}. {advice[dominant]}",
    ]


def build_suggestions(frame: pd.DataFrame, per_checkpoint, per_zone) -> dict:
    """Every generator, keyed by section, ready for the page and the JSON."""
    overall = frame.success.mean()
    return {
        "Which checkpoint to ship": advise_checkpoints(per_checkpoint),
        "Where the data is thin": advise_zones(per_zone, overall),
        "Policy or pipeline": advise_motion(per_checkpoint),
        "How the failures failed": advise_failures(frame),
        "Can this ranking be believed": advise_power(per_checkpoint),
    }


def rate_of(summary: pd.DataFrame, column: str) -> pd.Series:
    """A per-step count as a percentage of the steps it could have happened on."""
    return 100.0 * summary[column] / summary["total_steps"].replace(0, pd.NA)


ROWS_PER_PAGE = 24


def table_page(pdf: PdfPages, title: str, columns: list, values: list, note: str = "") -> None:
    """Pages that are nothing but a table, split so no row falls off the bottom.

    Column widths are set from the content rather than shared out evenly: a dozen
    columns of an even split gives "median s to success" the same room as "step", and
    the header then writes straight over its neighbours.
    """
    values = [[str(cell) for cell in row] for row in values]
    for start in range(0, max(1, len(values)), ROWS_PER_PAGE):
        page = values[start : start + ROWS_PER_PAGE]
        figure, axes = plt.subplots(figsize=PAGE)
        axes.axis("off")
        heading = title if start == 0 else f"{title} (continued)"
        axes.set_title(heading, fontsize=15, loc="left", pad=24)
        if note and start == 0:
            axes.text(0, 1.015, note, transform=axes.transAxes, fontsize=8, color="0.35")
        table = axes.table(cellText=page, colLabels=columns, loc="upper center", cellLoc="center")
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.auto_set_column_width(list(range(len(columns))))
        table.scale(1, 1.45)
        for column in range(len(columns)):
            table[0, column].set_facecolor("#dbe4ee")
            table[0, column].set_text_props(weight="bold")
        pdf.savefig(figure, bbox_inches="tight")
        plt.close(figure)


def cover_page(pdf: PdfPages, meta: dict, per_checkpoint: pd.DataFrame) -> None:
    figure = plt.figure(figsize=PAGE)
    figure.text(0.06, 0.88, "Checkpoint sweep", fontsize=26)
    figure.text(0.06, 0.835, meta["checkpoints_root"], fontsize=12, color="0.35")

    best = per_checkpoint.sort_values(["success_rate", "step"], ascending=[False, True]).iloc[0]
    lines = [
        f"Finished          {meta['finished']}",
        f"Namespace         {meta['namespace']}",
        f"Zones             {meta['zones'] or 'unrestricted placement'}",
        f"Episodes          {meta['episodes']} per zone per checkpoint",
        f"Timeout           {meta['seconds']:g} s of sim time per episode",
        f"Checkpoints       {len(per_checkpoint)}",
        f"Episodes run      {int(per_checkpoint.episodes.sum())}",
        f"Simulator time    {per_checkpoint.wall_minutes.sum() / 60:.1f} h",
        "",
        f"Best              {best.checkpoint} at {best.success_rate:.0%} "
        f"({int(best.successes)}/{int(best.episodes)}, "
        f"95% CI {best.ci_low:.0%}-{best.ci_high:.0%})",
        f"Extra eval flags  {' '.join(meta['extra']) or '(none)'}",
    ]
    if meta.get("stopped_at"):
        lines += [
            "",
            f"STOPPED EARLY     at {meta['stopped_at']}, which scored 0 across every "
            f"zone.",
            "                  Checkpoints before it were not evaluated. --no-early-stop",
            "                  runs them.",
        ]
    figure.text(0.06, 0.72, "\n".join(lines), fontsize=11, family="monospace", va="top")
    figure.text(
        0.06,
        0.10,
        "Rates are per episode; a 95% Wilson interval is drawn on every point, because at "
        "five\nepisodes a zone the gap between 3/5 and 4/5 is noise. episodes.csv has every "
        "rollout.",
        fontsize=9,
        color="0.35",
    )
    pdf.savefig(figure)
    plt.close(figure)


def success_page(pdf: PdfPages, per_checkpoint: pd.DataFrame, per_zone: pd.DataFrame) -> None:
    figure, axes = plt.subplots(figsize=PAGE)
    x = range(len(per_checkpoint))
    rate = per_checkpoint.success_rate.to_numpy()
    axes.errorbar(
        x,
        rate,
        yerr=[rate - per_checkpoint.ci_low, per_checkpoint.ci_high - rate],
        marker="o",
        markersize=7,
        linewidth=2,
        capsize=5,
        color="#1f4e79",
        label="all zones",
        zorder=3,
    )
    for label, group in per_zone.groupby("zone"):
        group = group.set_index("checkpoint").reindex(per_checkpoint.checkpoint)
        axes.plot(
            x,
            group.success_rate.to_numpy(),
            marker=".",
            linestyle="--",
            linewidth=1,
            alpha=0.65,
            label=f"zone {label}" if label >= 0 else "unrestricted",
        )
    for position, row in zip(x, per_checkpoint.itertuples()):
        axes.annotate(
            f"{int(row.successes)}/{int(row.episodes)}",
            (position, row.success_rate),
            textcoords="offset points",
            xytext=(0, 11),
            ha="center",
            fontsize=8,
        )
    axes.set_xticks(list(x))
    axes.set_xticklabels(per_checkpoint.checkpoint, rotation=45, ha="right")
    axes.set_ylim(-0.05, 1.12)
    axes.set_ylabel("success rate")
    axes.set_xlabel("checkpoint")
    axes.set_title("Success rate by checkpoint", fontsize=15, loc="left")
    axes.grid(axis="y", alpha=0.3)
    if len(per_zone.zone.unique()) > 1:
        axes.legend(fontsize=8, ncols=4)
    pdf.savefig(figure, bbox_inches="tight")
    plt.close(figure)


def zone_page(pdf: PdfPages, per_zone: pd.DataFrame, order: list) -> None:
    """Success rate as checkpoint x zone. Which zones a checkpoint cannot reach."""
    grid = per_zone.pivot(index="zone", columns="checkpoint", values="success_rate")
    counts = per_zone.pivot(index="zone", columns="checkpoint", values="successes")
    totals = per_zone.pivot(index="zone", columns="checkpoint", values="episodes")
    grid, counts, totals = grid[order], counts[order], totals[order]

    figure, axes = plt.subplots(figsize=PAGE)
    image = axes.imshow(grid.to_numpy(dtype=float), vmin=0, vmax=1, cmap="RdYlGn", aspect="auto")
    axes.set_xticks(range(len(grid.columns)))
    axes.set_xticklabels(grid.columns, rotation=45, ha="right")
    axes.set_yticks(range(len(grid.index)))
    axes.set_yticklabels([f"zone {z}" if z >= 0 else "any" for z in grid.index])
    for row in range(grid.shape[0]):
        for column in range(grid.shape[1]):
            hits, tries = counts.iat[row, column], totals.iat[row, column]
            if pd.notna(tries):
                axes.text(
                    column, row, f"{int(hits)}/{int(tries)}", ha="center", va="center", fontsize=8
                )
    axes.set_title("Success rate by zone", fontsize=15, loc="left")
    figure.colorbar(image, ax=axes, shrink=0.7, label="success rate")
    pdf.savefig(figure, bbox_inches="tight")
    plt.close(figure)


def suggestions_page(pdf: PdfPages, suggestions: dict) -> None:
    """What to do next, in words. Read this page and skip the rest if you are busy.

    Everything on it is derived from the charts that follow, and every claim of
    significance is the same Wilson interval drawn there -- so a suggestion can be
    checked against the data on the next page rather than taken on trust.
    """
    figure = plt.figure(figsize=PAGE)
    figure.text(0.06, 0.93, "What this suggests", fontsize=22)
    figure.text(
        0.06,
        0.90,
        "Generated from the numbers on the following pages. Significance is the same 95% "
        "Wilson interval used there.",
        fontsize=9,
        color="0.35",
    )

    y = 0.845
    for heading, notes in suggestions.items():
        if not notes:
            continue
        figure.text(0.06, y, heading, fontsize=12, weight="bold", color="#1f4e79")
        y -= 0.028
        for note in notes:
            wrapped = textwrap.fill(note, 118)
            figure.text(0.075, y, "• " + wrapped, fontsize=9.5, va="top", linespacing=1.5)
            y -= 0.022 * (wrapped.count("\n") + 1) + 0.012
        y -= 0.016
    pdf.savefig(figure)
    plt.close(figure)


def failure_page(pdf: PdfPages, frame: pd.DataFrame, order: list) -> None:
    """Failures split by what the arm was actually doing when it ran out of time."""
    modes = frame[frame.failure_mode != ""]
    if modes.empty:
        return
    grid = (
        modes.groupby(["checkpoint", "failure_mode"]).size().unstack(fill_value=0)
    ).reindex(index=order, columns=list(FAILURE_MODES), fill_value=0)

    figure, axes = plt.subplots(figsize=PAGE)
    bottom = [0] * len(grid)
    colours = ("#8c8c8c", "#c0504d", "#e8a33d", "#4f81bd")
    for mode, colour in zip(FAILURE_MODES, colours):
        axes.bar(range(len(grid)), grid[mode], 0.6, bottom=bottom, label=mode, color=colour)
        bottom = [total + value for total, value in zip(bottom, grid[mode])]
    axes.set_xticks(range(len(grid)))
    axes.set_xticklabels(grid.index, rotation=45, ha="right")
    axes.set_ylabel("failed episodes")
    axes.set_title("How the failures failed", fontsize=15, loc="left")
    axes.legend(fontsize=9)
    axes.grid(axis="y", alpha=0.3)
    figure.text(
        0.125,
        0.02,
        "never moved: tool path under "
        f"{NEVER_MOVED} m.   never closed: no close command.   closed on nothing: fingers "
        f"reached {GRIP_EMPTY} m, so nothing was between them.",
        fontsize=8,
        color="0.35",
    )
    pdf.savefig(figure, bbox_inches="tight")
    plt.close(figure)


def timing_page(pdf: PdfPages, frame: pd.DataFrame, per_checkpoint: pd.DataFrame) -> None:
    """How long a success takes, and how long the loop takes to compute a step."""
    figure, (top, bottom) = plt.subplots(2, 1, figsize=PAGE, sharex=True)
    order = list(per_checkpoint.checkpoint)
    samples = [
        frame.loc[
            (frame.checkpoint == name) & frame.success_seconds.notna(), "success_seconds"
        ].to_numpy()
        for name in order
    ]
    positions = [index for index, values in enumerate(samples) if len(values)]
    if positions:
        top.boxplot([samples[index] for index in positions], positions=positions, widths=0.5)
        for index in positions:
            top.scatter(
                [index] * len(samples[index]),
                samples[index],
                s=14,
                alpha=0.55,
                color="#1f4e79",
                zorder=3,
            )
    top.set_ylabel("sim seconds to success")
    top.set_title(
        "Time to success (successful episodes only) and control-loop cost",
        fontsize=15,
        loc="left",
    )
    top.grid(axis="y", alpha=0.3)

    bottom.bar(range(len(order)), per_checkpoint.median_step_ms, color="#7a9cc6")
    bottom.set_ylabel("median step (ms)")
    bottom.set_xticks(range(len(order)))
    bottom.set_xticklabels(order, rotation=45, ha="right")
    bottom.grid(axis="y", alpha=0.3)
    pdf.savefig(figure, bbox_inches="tight")
    plt.close(figure)


def health_page(pdf: PdfPages, per_checkpoint: pd.DataFrame) -> None:
    """The rollout's own complaints. A checkpoint can fail the task in several ways.

    Held steps mean the IK asked for a joint jump and the command was dropped, clamped
    means the requested motion was bigger than anything in the training set, and
    unreachable means the QP could not hit the pose at all. A low success rate with
    all three at zero is a policy problem; with any of them high it is a motion
    problem, and they want different fixes.
    """
    figure, (top, bottom) = plt.subplots(2, 1, figsize=PAGE, sharex=True)
    x = range(len(per_checkpoint))
    width = 0.27
    for offset, column, colour in (
        (-width, "held", "#c0504d"),
        (0.0, "clamped", "#e8a33d"),
        (width, "unreachable", "#4f81bd"),
    ):
        top.bar(
            [position + offset for position in x],
            rate_of(per_checkpoint, column),
            width,
            label=column,
            color=colour,
        )
    top.set_ylabel("% of control steps")
    top.set_title("Motion guards, and the return home", fontsize=15, loc="left")
    top.legend(fontsize=9)
    top.grid(axis="y", alpha=0.3)

    bottom.bar(x, per_checkpoint.home_return_rate, color="#77933c")
    bottom.set_ylabel("returned home")
    bottom.set_ylim(0, 1.05)
    bottom.set_xticks(list(x))
    bottom.set_xticklabels(per_checkpoint.checkpoint, rotation=45, ha="right")
    bottom.grid(axis="y", alpha=0.3)
    pdf.savefig(figure, bbox_inches="tight")
    plt.close(figure)


def write_report(
    frame: pd.DataFrame,
    per_checkpoint: pd.DataFrame,
    per_zone: pd.DataFrame,
    meta: dict,
    path: Path,
    suggestions: dict,
) -> None:
    matplotlib.use("Agg", force=True)
    order = list(per_checkpoint.checkpoint)
    with PdfPages(path) as pdf:
        cover_page(pdf, meta, per_checkpoint)
        suggestions_page(pdf, suggestions)
        table_page(
            pdf,
            "Per checkpoint",
            [
                "checkpoint",
                "step",
                "episodes",
                "successes",
                "rate",
                "95% CI",
                "median s to success",
                "median steps",
                "home return",
                "median step ms",
                "held %",
                "clamped %",
            ],
            [
                [
                    row.checkpoint,
                    row.step,
                    int(row.episodes),
                    int(row.successes),
                    f"{row.success_rate:.0%}",
                    f"{row.ci_low:.0%}-{row.ci_high:.0%}",
                    "-" if pd.isna(row.median_success_s) else f"{row.median_success_s:.1f}",
                    f"{row.median_steps:.0f}",
                    f"{row.home_return_rate:.0%}",
                    "-" if pd.isna(row.median_step_ms) else f"{row.median_step_ms:.0f}",
                    f"{100 * row.held / row.total_steps:.1f}",
                    f"{100 * row.clamped / row.total_steps:.1f}",
                ]
                for row in per_checkpoint.itertuples()
            ],
            note="Time to success counts successful episodes only. Held and clamped are "
            "percentages of control steps.",
        )
        success_page(pdf, per_checkpoint, per_zone)
        if len(per_zone.zone.unique()) > 1:
            zone_page(pdf, per_zone, order)
        table_page(
            pdf,
            "Per checkpoint and zone",
            ["checkpoint", "zone", "episodes", "successes", "rate", "median s to success"],
            [
                [
                    row.checkpoint,
                    "any" if row.zone < 0 else row.zone,
                    int(row.episodes),
                    int(row.successes),
                    f"{row.success_rate:.0%}",
                    "-" if pd.isna(row.median_success_s) else f"{row.median_success_s:.1f}",
                ]
                for row in per_zone.itertuples()
            ],
        )
        failure_page(pdf, frame, order)
        timing_page(pdf, frame, per_checkpoint)
        health_page(pdf, per_checkpoint)


def build_outputs(output: Path, meta: dict) -> pd.DataFrame:
    """Turn the raw JSON Lines into the CSV, the JSON summary and the PDF."""
    frame = with_failure_modes(load_records(output / "raw"))
    per_checkpoint = summarize(frame, ["checkpoint", "step"]).sort_values("step")
    per_zone = summarize(frame, ["checkpoint", "zone"])
    # Same order as the charts, so the tables read alongside them.
    per_zone = per_zone.set_index("checkpoint").loc[list(per_checkpoint.checkpoint)].reset_index()
    suggestions = build_suggestions(frame, per_checkpoint, per_zone)

    frame.to_csv(output / "episodes.csv", index=False)
    (output / "summary.json").write_text(
        json.dumps(
            {
                "run": meta,
                "suggestions": suggestions,
                "per_checkpoint": per_checkpoint.to_dict(orient="records"),
                "per_zone": per_zone.to_dict(orient="records"),
            },
            indent=2,
            default=str,
        )
    )
    write_report(frame, per_checkpoint, per_zone, meta, output / "report.pdf", suggestions)
    return per_checkpoint, suggestions


def main():
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        epilog="Unrecognised arguments are forwarded to eval_policy_pink unchanged, so "
        "--fps, --action-scale, --n-action-steps and friends all work here.",
    )
    parser.add_argument(
        "--checkpoints",
        type=Path,
        default=Path.home() / "models/smolvla_fr3_07_29/checkpoints",
        help="Directory of checkpoint directories (each holding pretrained_model/).",
    )
    parser.add_argument("--namespace", type=str, default="/Sim_0/Scene_0")
    parser.add_argument(
        "--zones",
        type=str,
        default="",
        help="Zones to evaluate in, as eval_policy_pink's --zone: '3', '3,8,12' or "
        "'3:10,8:4' for per-zone counts. Blank is unrestricted placement. 'all' is not "
        "accepted here -- list them, so the report has stable columns.",
    )
    parser.add_argument(
        "--episodes", type=int, default=5, help="Episodes per zone per checkpoint."
    )
    parser.add_argument("--seconds", type=float, default=60.0, help="Rollout timeout per episode.")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Where everything is written (default ~/eval_sweeps/<model>_<timestamp>). "
        "Pass an existing directory to resume it.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=0.0,
        help="Wall-clock seconds one checkpoint may take before it is killed and the "
        "sweep moves on (default: 4x its sim-time budget plus a margin). Sim time is not "
        "wall time -- Isaac with three cameras and a VLA on the GPU runs well under real "
        "time, so this is a hang detector, not a schedule.",
    )
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="Rebuild episodes.csv, summary.json and report.pdf from --output without "
        "running anything. The simulator is not touched.",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Re-run checkpoints that already have all their episodes in --output.",
    )
    parser.add_argument(
        "--seed-base",
        type=int,
        default=0,
        help="Every checkpoint sees the SAME layouts: episode N of each one is drawn from "
        "seed <base>+N. Paired samples, so a difference between two checkpoints is the "
        "policy rather than which of them drew the easy scenes -- worth more than doubling "
        "the episode count. -1 gives every episode a fresh draw, which is only worth it if "
        "you want each checkpoint measured against independent scenes.",
    )
    parser.add_argument(
        "--no-early-stop",
        action="store_true",
        help="Keep going past a checkpoint that scored 0 on a complete evaluation. Worth "
        "setting if you expect the run to be non-monotonic, or if you want the whole "
        "curve for a report rather than the answer to 'is the latest model any good'.",
    )
    args, extra = parser.parse_known_args()

    if args.zones.strip() == "all":
        raise SystemExit("--zones all is not supported; list the zones ('0,1,2,3').")

    output = args.output or (
        Path.home()
        / "eval_sweeps"
        / f"{args.checkpoints.resolve().parent.name}_{datetime.now():%Y%m%d_%H%M}"
    )
    output.mkdir(parents=True, exist_ok=True)

    meta = {
        "checkpoints_root": str(args.checkpoints),
        "namespace": args.namespace,
        "zones": args.zones,
        "episodes": args.episodes,
        "seconds": args.seconds,
        "extra": extra,
        "finished": "",
        "stopped_at": "",
    }

    if not args.report_only:
        checkpoints = discover_checkpoints(args.checkpoints)
        wanted = expected_episodes(args.zones, args.episodes)
        # 4x is for the gap between sim time and wall time; the flat margin covers the
        # model load, the scene settling and the homing before each episode.
        args.timeout = args.timeout or wanted * (4 * args.seconds + 60) + 300
        if extra:
            print(f"Forwarding to eval_policy_pink: {' '.join(extra)}")
        print(
            f"{len(checkpoints)} checkpoints x {wanted} episodes "
            f"({args.zones or 'unrestricted'}) -> {output}\n"
            f"Worst case {len(checkpoints) * wanted * args.seconds / 3600:.1f} h of sim "
            f"time; per-checkpoint timeout {args.timeout / 60:.0f} min.\n"
        )

        # Newest first. The last checkpoint is the one you would ship, so the sweep
        # answers "is this model any good" in its first half-hour instead of its last,
        # and the walk backwards is then a search for where the ability appeared.
        for index, checkpoint in enumerate(reversed(checkpoints), start=1):
            results = output / "raw" / f"{checkpoint.name}.jsonl"
            done = len(results.read_text().splitlines()) if results.is_file() else 0
            head = f"[{index}/{len(checkpoints)}] {checkpoint.name}"
            if done >= wanted and not args.no_resume:
                print(f"{head}: {done} episodes already recorded, skipping.")
            else:
                if done and not args.no_resume:
                    # Partial results from a killed run: they stay, and this run appends
                    # to them. A checkpoint's rate is then over more episodes than
                    # --episodes, which the report shows honestly rather than hiding.
                    print(f"{head}: resuming, {done} episodes already recorded.")

                started = time.perf_counter()
                print(f"{head}: running...", flush=True)
                status = run_checkpoint(
                    checkpoint, results, output / "logs" / f"{checkpoint.name}.log", args, extra
                )
                print(
                    f"{head}: {status}, {checkpoint_score(results)[1] - done} episodes in "
                    f"{(time.perf_counter() - started) / 60:.1f} min"
                )

            wins, ran = checkpoint_score(results)
            if not args.no_early_stop and is_hopeless(wins, ran, wanted):
                print(
                    f"{head}: 0/{ran} across every zone. Stopping -- going backwards, "
                    f"the checkpoints left are earlier in training than one that cannot "
                    f"do the task at all. --no-early-stop runs them anyway."
                )
                meta["stopped_at"] = checkpoint.name
                break

    meta["finished"] = datetime.now().strftime("%Y-%m-%d %H:%M")
    per_checkpoint, suggestions = build_outputs(output, meta)
    print(f"\n{output}/report.pdf")
    for row in per_checkpoint.itertuples():
        print(
            f"  {row.checkpoint:>8}  {row.success_rate:5.0%}  "
            f"({int(row.successes)}/{int(row.episodes)})"
        )
    for note in suggestions["Which checkpoint to ship"]:
        print(f"\n{textwrap.fill(note, 88)}")


if __name__ == "__main__":
    main()
