"""Compare recorded debug runs -- what changed when you changed one setting.

Reads the trace directories ``debug_rollout`` wrote and reduces each to the handful of
numbers that separate the ways this task fails::

    python -m block_bin.compare_runs ~/eval_debug/horizon_* --plot ~/eval_debug/cmp.png

The metrics, and why these:

``closest``   how near the tool ever got to the target block. Says whether the policy
              can aim at all. A run that never gets inside 20 cm has a reaching
              problem and nothing downstream is worth reading.
``at close``  how far away it was when it actually commanded the gripper shut. The gap
              between this and ``closest`` is the whole story of a policy that arrives
              correctly and grasps anyway: aim is ``closest``, timing is the gap.
``late``      steps between the closest approach and the close command. Positive means
              it passed the block and shut afterwards. Compare it against
              ``n_action_steps`` -- a lateness of about one chunk is a policy acting on
              an observation one chunk old.
``tilt@near`` wrist tilt from the starting down-pose at the closest approach, and
``tilt@shut`` the same at the close. Measuring only the second one attributes a
              failure to orientation that may have happened after the miss.
``arm``       the paired left/right instruction test: does the tool swing toward the
              bin each trial was told? Grounding, independent of whether it grasps.

Nothing here re-runs the simulator; it is arithmetic over the traces.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from block_bin.replay_rollout import column, tilt_from_start
from block_bin.rollout_trace import find_pairs, load_pair

# Commanded gripper below this is a close. Same threshold the recorder dates the grasp
# with (eval_policy.GRIP_CLOSE_COMMAND), repeated rather than imported so this module
# reads traces without pulling in lerobot and the ROS workspace.
CLOSE_COMMAND = 0.02

# Metres the target must rise above where it started before a placement is believed.
#
# Not decoration. IsSuccess is an OBB containment test with a 5 cm tolerance, and a
# block SHOVED across the table until it rests against a bin that the arm has also
# shoved satisfies it -- observed: bin displaced 0.152 m, block dragged 0.459 m, peak
# lift 0.0155 m (table jitter), fingers closing through to -0.022 m on empty air, and
# the scene reported success. A real pick clears the bin wall by a wide margin, and
# nothing that is only pushed ever leaves the table. This is the cheap discriminator.
LIFT_MIN = 0.05


def to_state_frame(meta: dict, position) -> np.ndarray:
    """A world-frame object position, in the frame the tool is recorded in.

    The policy's state is scene-relative; PoseRequest answers in world. Traces that
    record ``scene_origin`` state the difference, and older ones -- recorded when the
    tool was itself world-frame -- need no correction and get none.
    """
    origin = meta.get("scene_origin")
    position = np.asarray(position, dtype=float)
    return position - np.asarray(origin, dtype=float) if origin else position


def target_track(trace: dict) -> np.ndarray | None:
    """Target block position at every step, from the most recent pose poll.

    Poses are sampled on an interval, so this holds the last known position between
    polls -- and it follows the block rather than assuming it stayed where it started,
    which matters exactly when the arm knocks it.
    """
    meta, steps = trace["meta"], trace["steps"]
    key = str(meta.get("target", "")).rsplit("/", 1)[-1]
    start = (meta.get("objects_at_start") or {}).get(key)
    if not key or not steps:
        return None

    track, last = [], to_state_frame(meta, start) if start else None
    for step in steps:
        polled = (step.get("poses") or {}).get(key)
        if polled:
            last = to_state_frame(meta, polled)
        track.append(last if last is not None else [np.nan] * 3)
    return np.asarray(track, dtype=float)


def rollout_metrics(trace: dict) -> dict:
    """One rollout reduced to the numbers that distinguish its failure mode."""
    steps = trace["steps"]
    if not steps:
        return {}

    tool = np.stack([column(steps, "eef_position", i) for i in range(3)], axis=1)
    block = target_track(trace)
    tilt = tilt_from_start(steps)
    grips = column(steps, "grip_command")
    closed = np.flatnonzero(grips < CLOSE_COMMAND)
    grasp = int(closed[0]) if len(closed) else None

    period = float(np.median(np.diff(column(steps, "sim_seconds")))) if len(steps) > 1 else np.nan
    metrics = {
        "steps": len(steps),
        "period": period,
        "grasp": grasp,
        "reached": (trace["meta"].get("summary") or {}).get("reached"),
        "asked_for": trace["meta"].get("asked_for"),
        "tilt_end": float(tilt[-1]) if tilt is not None else np.nan,
    }

    if block is None or np.isnan(block).all():
        return metrics

    distance = np.linalg.norm(tool - block, axis=1)
    if np.isnan(distance).all():
        return metrics
    near = int(np.nanargmin(distance))
    heights = block[:, 2]
    metrics["lift"] = (
        float(np.nanmax(heights) - heights[0]) if not np.isnan(heights).all() else np.nan
    )
    metrics |= {
        "closest": float(distance[near]),
        "closest_step": near,
        "grip_at_closest": float(grips[near]),
        "tilt_at_closest": float(tilt[near]) if tilt is not None else np.nan,
        "at_close": float(distance[grasp]) if grasp is not None else np.nan,
        "tilt_at_close": float(tilt[grasp]) if grasp is not None and tilt is not None else np.nan,
        # Positive: it passed the block and shut afterwards.
        "late": (grasp - near) if grasp is not None else None,
    }
    return metrics


def pair_shift(pair: dict) -> float:
    """Differential lean of the two instructions, metres. Positive = followed.

    ``lean`` is closest-to-left minus closest-to-right, so following means the
    left-instructed trial leans left and the right-instructed one leans right. Paired
    because home is not equidistant from the bins: only the CHANGE means anything.
    """
    leans = {}
    for name in ("left", "right"):
        trace = pair.get(name)
        if not trace or not trace["steps"]:
            return np.nan
        objects = trace["meta"].get("objects_at_start") or {}
        if "left_bin" not in objects or "right_bin" not in objects:
            return np.nan
        tool = np.stack([column(trace["steps"], "eef_position", i) for i in range(2)], axis=1)
        centres = {
            side: to_state_frame(trace["meta"], objects[f"{side}_bin"])[:2]
            for side in ("left", "right")
        }
        closest = {
            side: float(np.min(np.linalg.norm(tool - centre, axis=1)))
            for side, centre in centres.items()
        }
        leans[name] = closest["left"] - closest["right"]
    return leans["right"] - leans["left"]


def summarise(directory) -> dict:
    """Every rollout of one run, reduced to medians."""
    directory = Path(directory)
    rollouts, shifts = [], []
    for pair_path in find_pairs(directory):
        pair = load_pair(pair_path)
        shifts.append(pair_shift(pair))
        for trace in pair.values():
            metrics = rollout_metrics(trace)
            if metrics:
                metrics["horizon"] = trace["meta"].get("n_action_steps")
                rollouts.append(metrics)

    def median(key):
        values = [r[key] for r in rollouts if r.get(key) is not None and not _nan(r.get(key))]
        return float(np.median(values)) if values else np.nan

    horizons = {r.get("horizon") for r in rollouts}
    return {
        "run": directory.name,
        "horizon": horizons.pop() if len(horizons) == 1 else None,
        "rollouts": len(rollouts),
        "pairs": len(shifts),
        "followed": sum(1 for s in shifts if s > 0.05),
        "shift": float(np.nanmean(shifts)) if shifts else np.nan,
        "period": median("period"),
        "closest": median("closest"),
        "at_close": median("at_close"),
        "late": median("late"),
        "tilt_at_closest": median("tilt_at_closest"),
        "tilt_at_close": median("tilt_at_close"),
        "grip_at_closest": median("grip_at_closest"),
        "lift": median("lift"),
        # Three conditions, and all three earn their place. "In a bin" alone counts a
        # block shoved against a displaced bin; "lifted" alone counts one picked up and
        # dropped anywhere; "the right bin" alone is what makes this the task rather
        # than a pick-and-drop. An earlier version of this counted any bin, and would
        # have scored a policy that always goes left as perfect.
        "placed": sum(1 for r in rollouts if _placed(r)),
        "wrong_bin": sum(
            1
            for r in rollouts
            if r.get("reached") in ("left", "right")
            and (r.get("lift") or 0) > LIFT_MIN
            and r.get("asked_for")
            and r["reached"] != r["asked_for"]
        ),
        "claimed": sum(1 for r in rollouts if r.get("reached") in ("left", "right")),
    }


def _placed(metrics: dict) -> bool:
    """Lifted, and put in the bin the instruction named."""
    reached, asked = metrics.get("reached"), metrics.get("asked_for")
    if reached not in ("left", "right") or (metrics.get("lift") or 0) <= LIFT_MIN:
        return False
    return reached == asked if asked else True


def _nan(value) -> bool:
    return isinstance(value, float) and np.isnan(value)


def table(summaries: list) -> str:
    """The comparison, one row per run."""
    head = (
        f"{'run':<14}{'horiz':>6}{'arm':>8}{'shift':>8}{'closest':>9}{'at close':>10}"
        f"{'late':>6}{'lift':>8}{'placed':>9}{'wrong':>7}"
    )
    lines = [head, "-" * len(head)]
    for summary in summaries:
        # Composed first, padded second. Interpolating "4" and "/4" through separate
        # width specs is how the horizon column ended up reading "14/4".
        arm = f"{summary['followed']}/{summary['pairs']}"
        placed = f"{summary['placed']}/{summary['rollouts']}"
        horizon = str(summary["horizon"] or "?")
        lines.append(
            f"{summary['run']:<14}{horizon:>6}{arm:>8}{summary['shift']:>+8.3f}"
            f"{summary['closest']:>9.3f}{summary['at_close']:>10.3f}{summary['late']:>6.0f}"
            f"{summary['lift']:>8.3f}{placed:>9}{summary['wrong_bin']:>7}"
        )
    lines += [
        "",
        "closest/at close: metres from the target block, at the nearest approach and at",
        "the moment the gripper was commanded shut. late: steps between the two -- compare",
        "it with the horizon. arm: paired left/right instruction test. lift: how far the",
        "block ever rose above the table. placed: lifted it into the bin the instruction",
        "NAMED -- a block shoved against a shoved bin satisfies IsSuccess, and a policy",
        "that always picks one bin would score perfectly on 'reached a bin'. wrong: put",
        "it in a bin, lifted, but the other one.",
    ]

    fooled = [s for s in summaries if s["claimed"] > s["placed"]]
    if fooled:
        # Loud, because this is the number every success rate in this project is built
        # on, and it is being satisfied without the task being done.
        lines += [
            "",
            "WARNING: the scene's own criterion counted placements that never left the",
            "table -- " + ", ".join(
                f"{s['run']}: {s['claimed']} claimed, {s['placed']} lifted" for s in fooled
            ),
            f"A block only pushed cannot rise {LIFT_MIN:g} m. Treat those as pushes.",
        ]
    return "\n".join(lines)


def plot(summaries: list, path: str) -> None:
    """Four panels against the horizon: aim, timing, tilt, grounding."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    order = sorted(summaries, key=lambda s: s["horizon"] or 0)
    x = [s["horizon"] or 0 for s in order]
    figure, axes = plt.subplots(2, 2, figsize=(11, 7.5))
    figure.suptitle("what changes with the action horizon", fontsize=11)

    panels = [
        (axes[0][0], "distance to the block (m)", [
            ("closest approach", [s["closest"] for s in order], "tab:blue"),
            ("when it closed", [s["at_close"] for s in order], "tab:red"),
        ]),
        (axes[0][1], "close command lateness (steps)", [
            ("steps after the closest approach", [s["late"] for s in order], "tab:purple"),
        ]),
        (axes[1][0], "wrist tilt (deg)", [
            ("at the closest approach", [s["tilt_at_closest"] for s in order], "tab:green"),
            ("at the close", [s["tilt_at_close"] for s in order], "tab:brown"),
        ]),
        (axes[1][1], "instruction following (m of differential lean)", [
            ("arm shift", [s["shift"] for s in order], "tab:orange"),
        ]),
    ]
    for axis, title, series in panels:
        for label, values, colour in series:
            axis.plot(x, values, "o-", color=colour, label=label)
        axis.set_title(title, fontsize=9)
        axis.set_xlabel("n_action_steps", fontsize=8)
        axis.grid(alpha=0.3)
        axis.legend(fontsize=7)
    # The lateness panel is read against the horizon itself: a run whose close lands one
    # chunk after the closest approach is acting on a chunk-old observation.
    axes[0][1].plot(x, x, ls=":", color="grey", label="one chunk")
    axes[0][1].legend(fontsize=7)

    figure.tight_layout()
    figure.savefig(path, dpi=110)
    print(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("runs", nargs="+", help="Run directories written by debug_rollout.")
    parser.add_argument("--plot", type=str, default="", help="Also write a PNG of the trends.")
    parser.add_argument("--json", type=str, default="", help="Also write the summaries as JSON.")
    args = parser.parse_args()

    summaries = [summarise(run) for run in args.runs]
    summaries = [s for s in summaries if s["rollouts"]]
    if not summaries:
        raise SystemExit("No rollouts found in any of those directories.")
    summaries.sort(key=lambda s: (s["horizon"] is None, s["horizon"] or 0))
    print(table(summaries))

    if args.json:
        Path(args.json).write_text(json.dumps(summaries, indent=2))
    if args.plot:
        plot(summaries, args.plot)


if __name__ == "__main__":
    main()
