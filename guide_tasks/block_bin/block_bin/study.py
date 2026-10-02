"""Sweep checkpoints, keep every rollout, and report on it offline or interactively.

``sweep_checkpoints`` answers "which checkpoint" with a table of rates.
``debug_rollout`` answers "what did it actually do" with one replayable pair. A study
needs both at once: a rate you can rank checkpoints by, and the rollouts behind it, so
a surprising number can be opened rather than argued about.

    python -m block_bin.study run --zones 1,2,3,6,7,8 --ood 0,4,12,17 --episodes 10
    python -m block_bin.study report ~/eval_studies/<run>      # offline: csv, json, pdf
    python -m block_bin.study gui    ~/eval_studies/<run>      # online: step through it

Two things it adds beyond running the two tools back to back.

**Success is gated on a lift.** ``IsSuccess`` is an OBB containment test with a 5 cm
tolerance, and a block shoved across the table until it rests against a bin the arm has
also shoved satisfies it -- measured: bin displaced 0.152 m, block dragged 0.459 m, peak
lift 0.0155 m, and the scene reported success. Every episode here traces the target
block's height, so ``placed`` means lifted clear of the table AND in a bin, and the gap
between that and the scene's own verdict is reported rather than hidden.

**In-distribution and out-of-distribution are scored separately.** ``--zones`` are the
cells the checkpoint trained on and ``--ood`` are cells it never saw. Pooling them
produces a number that is neither, and which moves when you change the mix.

Frames are sampled, not kept for every episode: the per-step numbers are ~100 kB a
rollout and the camera frames ~11 MB, so a long campaign traces everything numerically
and keeps video for every Nth episode (``--trace-frames-every``).
"""

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

from block_bin import sweep_checkpoints as sweep
from block_bin.rollout_trace import load_trace

# Metres the target must rise before a placement is believed. Nothing that is only
# pushed leaves the table; a real pick clears the bin wall by a wide margin.
LIFT_MIN = 0.05


def episode_traces(directory: Path) -> list:
    """Every traced episode under a directory, as (checkpoint, path), in run order.

    Two layouts. A study writes ``<dir>/traces/<checkpoint>/ep_*``; a direct
    eval_policy_pink or debug_rollout run writes ``<dir>/ep_*`` with no checkpoint
    level, and is labelled by the directory name instead so the rest of the pipeline
    needs no special case.
    """
    directory = Path(directory)
    found = []
    for checkpoint in sorted((directory / "traces").glob("*")):
        if checkpoint.is_dir():
            found += [(checkpoint.name, p) for p in sorted(checkpoint.glob("ep_*")) if p.is_dir()]
    if not found:
        found = [(directory.name, p) for p in sorted(directory.glob("ep_*"))
                 if p.is_dir() and (p / "meta.json").is_file()]
    return found


# Commanded gripper below this is a close; measured above this while closed means
# something is between the fingers. See eval_policy_pink.GRIP_STALL.
CLOSE_COMMAND = 0.02
GRIP_STALL = 0.015

# The bin a goal prim names.
GOAL_SIDE = {"/bin_0": "left", "/bin_1": "right"}


def grasped_something(steps: list) -> bool:
    """Did the fingers ever close on an object rather than on air?

    Per CONTIGUOUS close, not over the whole episode: a policy that grasps once and
    then opens and closes on nothing would otherwise be scored by its worst attempt.
    An empty close runs the fingers through to ~0; a cube stops them near its
    half-width.
    """
    best, run = 0.0, []
    for step in steps + [{}]:
        command, measured = step.get("grip_command"), step.get("grip_measured")
        if command is not None and measured is not None and command < CLOSE_COMMAND:
            run.append(measured)
        elif run:
            best = max(best, min(run))
            run = []
    return best > GRIP_STALL


def lifts(trace: dict) -> dict:
    """How far each cube rose above where it started, by name."""
    first, peak = {}, {}
    for step in trace["steps"]:
        for name, position in (step.get("poses") or {}).items():
            if not position or not name.endswith("_block"):
                continue
            first.setdefault(name, position[2])
            peak[name] = max(peak.get(name, position[2]), position[2])
    return {name: peak[name] - first[name] for name in first}


def rungs(trace: dict) -> dict:
    """The ladder between doing nothing and succeeding.

    Each rung allows ANY cube, because grasping the wrong one is a different failure
    from never closing at all -- the point is to locate where the behaviour stops, not
    to award partial credit. Only the last rung is the task.
    """
    summary = trace["meta"].get("summary") or {}
    # Traces recorded before the ladder existed followed only the target cube and never
    # asked which bin anything ended in. They can still answer "closed" and "lifted";
    # they cannot answer the target rungs, and must not be scored as though they said
    # no -- that silently rewrites an old study as 0%.
    graded = "bin_contents" in summary
    contents = summary.get("bin_contents") or {}
    raised = lifts(trace)
    lifted = {name for name, rise in raised.items() if rise > LIFT_MIN}
    target = str(summary.get("target") or "").rsplit("/", 1)[-1]
    goal = GOAL_SIDE.get(summary.get("goal"))

    binned = {name for name, side in contents.items() if side and name in lifted}
    others = {name for name in lifted if name != target}
    # The task executed correctly on the WRONG object: lifted a distractor and put it
    # in the bin the instruction named. Everything except identifying the cube worked,
    # which is a completely different defect from fumbling the grasp -- and it is
    # invisible if the ladder stops at "lifted the wrong cube".
    wrong_in_goal = {n for n in others & binned if goal and contents.get(n) == goal}
    return {
        "graded": graded,
        "closed": grasped_something(trace["steps"]),
        "lifted": bool(lifted),
        "lifted_target": target in lifted,
        "binned": bool(binned),
        "binned_target": target in binned,
        "wrong_lifted": bool(others),
        "wrong_binned": bool(others & binned),
        "wrong_in_goal_bin": bool(wrong_in_goal),
        # The task. Named apart from the scene's own "success" so the two can be
        # compared rather than silently merged.
        "task_success": bool(
            target and goal and contents.get(target) == goal and target in lifted
        ),
        "best_lift": max(raised.values()) if raised else float("nan"),
    }


def lift_of(trace: dict) -> float:
    """How far the TARGET cube rose above where it started. NaN if never polled.

    Named explicitly rather than taking the first entry of the poses dict: that was
    equivalent while only the target was followed, and silently became "whichever cube
    dict ordering put first" once all four were.
    """
    summary = trace["meta"].get("summary") or {}
    target = str(summary.get("target") or "").rsplit("/", 1)[-1]
    raised = lifts(trace)
    if target and target in raised:
        return raised[target]
    return raised[next(iter(raised))] if len(raised) == 1 else float("nan")


def collect(directory: Path) -> pd.DataFrame:
    """One row per episode: what the sweep scored, and what the trace shows."""
    rows = []
    for checkpoint, path in episode_traces(directory):
        try:
            trace = load_trace(path)
        except SystemExit:
            continue
        meta = trace["meta"]
        summary = meta.get("summary") or {}
        rows.append(
            {
                "checkpoint": checkpoint,
                "episode": meta.get("episode"),
                "zone": meta.get("zone"),
                "steps": len(trace["steps"]),
                "lift": lift_of(trace),
                "reached": summary.get("reached"),
                **rungs(trace),
            }
        )
    return pd.DataFrame(rows)


def merge_scores(directory: Path, traced: pd.DataFrame) -> pd.DataFrame:
    """Join the sweep's per-episode verdicts onto the traced lifts."""
    scored = sweep.load_records(directory / "raw")
    if scored.empty:
        return traced
    scored = scored.reset_index(drop=True)
    scored["order"] = scored.groupby("checkpoint").cumcount()
    if traced.empty:
        scored["lift"] = float("nan")
        return scored
    traced = traced.sort_values(["checkpoint", "episode"]).reset_index(drop=True)
    traced["order"] = traced.groupby("checkpoint").cumcount()
    carried = [
        "checkpoint", "order", "lift", "graded", "closed", "lifted", "lifted_target",
        "binned", "binned_target", "wrong_lifted", "wrong_binned", "wrong_in_goal_bin",
        "task_success", "best_lift",
    ]
    return scored.merge(
        traced[[c for c in carried if c in traced]],
        on=["checkpoint", "order"],
        how="left",
        suffixes=("", "_traced"),
    )


def label_zones(frame: pd.DataFrame, trained: set) -> pd.DataFrame:
    """Mark each episode as in- or out-of-distribution by its zone."""
    frame = frame.copy()
    frame["split"] = frame["zone"].apply(
        lambda z: "in-distribution" if z in trained else "held-out"
    )
    return frame


def score(frame: pd.DataFrame) -> pd.DataFrame:
    """Per checkpoint and split: the scene's rate, the lift-gated rate, and intervals."""
    if frame.empty:
        return pd.DataFrame()
    frame = frame.copy()
    frame["claimed"] = frame["success"].astype(bool) if "success" in frame else False
    for rung in ("closed", "lifted", "lifted_target", "binned", "binned_target",
                 "wrong_lifted", "wrong_binned", "wrong_in_goal_bin", "task_success"):
        if rung not in frame:
            frame[rung] = False
        frame[rung] = frame[rung].fillna(False).astype(bool)
    # The task, gated on a real lift the same way everything else here is. Where the
    # trace predates the ladder there is no bin_contents to judge against, so fall back
    # to the scene's own verdict plus the lift -- which is what this script scored on
    # before, and is still honest.
    # A Series, not a bare False: Series.where needs a conditional of matching shape.
    graded = (
        frame["graded"].fillna(False).astype(bool)
        if "graded" in frame
        else pd.Series(False, index=frame.index)
    )
    # task_success already requires the target to have cleared LIFT_MIN inside rungs();
    # re-gating it on a separate lift column can only subtract, and did.
    frame["placed"] = frame["task_success"].where(
        graded, frame["claimed"] & (frame["lift"].fillna(0) > LIFT_MIN)
    )
    frame["placed"] = frame["placed"].fillna(False).astype(bool)

    rows = []
    for (checkpoint, split), part in frame.groupby(["checkpoint", "split"]):
        placed, n = int(part["placed"].sum()), len(part)
        low, high = sweep.wilson(placed, n)
        rows.append(
            {
                "checkpoint": checkpoint,
                "split": split,
                "episodes": n,
                # The ladder, each rung counting ANY cube unless it says target.
                "closed": int(part["closed"].sum()),
                "lifted": int(part["lifted"].sum()),
                "lifted_target": int(part["lifted_target"].sum()),
                "binned": int(part["binned"].sum()),
                "binned_target": int(part["binned_target"].sum()),
                "wrong_in_goal_bin": int(part["wrong_in_goal_bin"].sum()),
                "placed": placed,
                "rate": placed / n if n else float("nan"),
                "ci_low": low,
                "ci_high": high,
                # Where the scene's own criterion and the physical check disagree.
                "claimed": int(part["claimed"].sum()),
                "graded": int(part["graded"].sum()) if "graded" in part else 0,
                "median_lift": float(part["lift"].median()),
            }
        )
    return pd.DataFrame(rows).sort_values(["split", "rate"], ascending=[True, False])


def ranking(scored: pd.DataFrame, split: str = "in-distribution") -> list:
    """Which checkpoint to take forward, and whether the data can tell."""
    part = scored[scored["split"] == split].sort_values("rate", ascending=False)
    if part.empty:
        return [f"No {split} episodes to rank."]

    best = part.iloc[0]
    notes = [
        f"Best on {split}: {best['checkpoint']} at {best['rate']:.0%} "
        f"({best['placed']}/{best['episodes']}), 95% CI {best['ci_low']:.0%}-{best['ci_high']:.0%}."
    ]
    if len(part) > 1:
        second = part.iloc[1]
        if second["ci_high"] >= best["ci_low"]:
            needed = sweep.episodes_to_separate(best["rate"], second["rate"])
            notes.append(
                f"NOT separable from {second['checkpoint']} ({second['rate']:.0%}): the "
                f"intervals overlap. Separating them needs about {needed} episodes per "
                f"checkpoint; there are {best['episodes']}. Treat the top group as tied "
                f"and pick on another axis."
            )
        else:
            notes.append(
                f"Clear of {second['checkpoint']} ({second['rate']:.0%}) -- the intervals "
                f"do not overlap, so this is a real ordering."
            )

    total = int(part["episodes"].sum())
    if not int(part["graded"].sum()):
        notes.append(
            "These traces predate the outcome ladder: no bin contents were recorded, so "
            "the target rungs below read 0 and the rate falls back to the scene's "
            "verdict plus the lift gate."
        )
    notes.append(
        "Where it stops, over all episodes of this split: "
        + " -> ".join(
            f"{name} {int(part[key].sum())}/{total}"
            for name, key in (
                ("closed on a cube", "closed"),
                ("lifted one", "lifted"),
                ("lifted the RIGHT one", "lifted_target"),
                ("got one in a bin", "binned"),
                ("RIGHT bin, WRONG cube", "wrong_in_goal_bin"),
                ("right cube in right bin", "placed"),
            )
        )
    )
    disagree = int(part["claimed"].sum()) - int(part["placed"].sum())
    if disagree:
        notes.append(
            f"IsSuccess and the physical check differ on {disagree} episode(s) -- "
            f"claimed {int(part['claimed'].sum())}, verified {int(part['placed'].sum())}."
        )
    return notes


def write_report(directory: Path) -> pd.DataFrame:
    """Offline: csv, json and a pdf, all from the traces and the raw records."""
    trained = set(json.loads((directory / "study.json").read_text())["zones"])
    frame = label_zones(merge_scores(directory, collect(directory)), trained)
    scored = score(frame)
    if scored.empty:
        print(f"No episodes recorded yet in {directory}.")
        return scored

    frame.to_csv(directory / "episodes.csv", index=False)
    scored.to_csv(directory / "checkpoints.csv", index=False)
    notes = ranking(scored)
    (directory / "ranking.json").write_text(
        json.dumps({"notes": notes, "checkpoints": scored.to_dict("records")}, indent=2)
    )

    from matplotlib.backends.backend_pdf import PdfPages

    with PdfPages(directory / "study.pdf") as pdf:
        sweep.table_page(
            pdf,
            "Checkpoints, scored on a real lift",
            ["checkpoint", "split", "n", "closed", "lifted", "right cube",
             "in a bin", "SUCCESS", "rate", "95% CI"],
            [
                [
                    r["checkpoint"], r["split"], r["episodes"], r["closed"], r["lifted"],
                    r["lifted_target"], r["binned"], r["placed"],
                    f"{r['rate']:.0%}", f"{r['ci_low']:.0%}-{r['ci_high']:.0%}",
                ]
                for r in scored.to_dict("records")
            ],
            note="Each rung counts ANY cube unless it says otherwise, so a policy that "
            "grasps the wrong one still registers -- the point is where the behaviour "
            "stops, not partial credit. SUCCESS is the right cube in the right bin, "
            "lifted clear of the table.",
        )
        placed = frame["task_success"].fillna(False).astype(bool) & (
            frame["lift"].fillna(0) > LIFT_MIN
        )
        counts = (
            frame.assign(placed=placed)
            .groupby(["checkpoint", "zone", "split"])["placed"]
            .agg(["sum", "size"])
            .reset_index()
        )
        if not counts.empty:
            sweep.table_page(
                pdf,
                "Per zone",
                ["checkpoint", "zone", "split", "placed", "episodes"],
                [
                    [r["checkpoint"], r["zone"], r["split"], int(r["sum"]), int(r["size"])]
                    for r in counts.to_dict("records")
                ],
                note="The zones a checkpoint trained on and the ones it did not, kept "
                "apart. A pooled rate is neither, and moves with the mix.",
            )

    for note in notes:
        print(note)
    print(f"\n{directory}/study.pdf")
    return scored


def open_gui(directory: Path, trial: str | None = None) -> None:
    """Online: step through every traced episode of the study."""
    import matplotlib.pyplot as plt

    from block_bin.replay_rollout import Replay

    traces = [path for _checkpoint, path in episode_traces(directory)]
    if not traces:
        raise SystemExit(f"No traced episodes under {directory} (neither traces/*/ep_* nor ep_*).")
    print(f"{len(traces)} traced episodes. n / b steps between them.")
    Replay(traces, trial or traces[0].name)
    plt.show()


def run(args, extra: list) -> Path:
    """Sweep every selected checkpoint over both zone sets, tracing as it goes."""
    checkpoints = sweep.select_checkpoints(sweep.discover_checkpoints(args.checkpoints), args.only)
    zones = ",".join(str(z) for z in args.zones.split(",") + args.ood.split(",") if z.strip())
    output = Path(args.output).expanduser() if args.output else (
        Path.home() / "eval_studies" / f"{args.checkpoints.parts[-2]}_"
        f"{datetime.now().strftime('%Y%m%d_%H%M')}"
    )
    (output / "raw").mkdir(parents=True, exist_ok=True)
    (output / "logs").mkdir(exist_ok=True)

    trained = [int(z) for z in args.zones.split(",") if z.strip()]
    held_out = [int(z) for z in args.ood.split(",") if z.strip()]
    (output / "study.json").write_text(
        json.dumps(
            {
                "checkpoints_root": str(args.checkpoints),
                "zones": trained,
                "ood": held_out,
                "episodes": args.episodes,
                "seconds": args.seconds,
                "extra": extra,
                "started": datetime.now().strftime("%Y-%m-%d %H:%M"),
            },
            indent=2,
        )
    )

    wanted = sweep.expected_episodes(zones, args.episodes)
    # Two numbers, because they differ by 3x and only one of them is what you wait.
    # A 60 s rollout costs about 3 min of wall clock once homing, the settle, the
    # success poll and the policy's own inference are counted.
    episodes = len(checkpoints) * wanted
    print(
        f"{len(checkpoints)} checkpoints x {wanted} episodes "
        f"({len(trained)} trained zones + {len(held_out)} held out) -> {output}\n"
        f"{episodes * args.seconds / 3600:.1f} h of rollout, about "
        f"{episodes * 3 / 60:.0f} h of wall clock at the measured 3 min an episode.\n"
    )

    args.zones = zones  # what run_checkpoint forwards as --zone to the child
    try:
        _sweep_checkpoints(args, checkpoints, output, extra, wanted)
    except Exception as error:  # noqa: BLE001 - hours of episodes outrank a clean exit
        print(f"\nSweep ended with {type(error).__name__}: {error}")
        print("Episodes already on disk are intact; reporting on those.")
    return output


def _sweep_checkpoints(args, checkpoints, output, extra, wanted) -> None:
    """The loop itself, so a failure anywhere in it still leaves a report behind."""
    with sweep.stop_service(args.namespace) as stop:
        for index, checkpoint in enumerate(reversed(checkpoints), start=1):
            results = output / "raw" / f"{checkpoint.name}.jsonl"
            done = len(results.read_text().splitlines()) if results.is_file() else 0
            head = f"[{index}/{len(checkpoints)}] {checkpoint.name}"
            if done >= wanted:
                print(f"{head}: {done} episodes already recorded, skipping.")
                continue

            started = time.perf_counter()
            print(f"{head}: running...", flush=True)
            status = sweep.run_checkpoint(
                checkpoint,
                results,
                output / "logs" / f"{checkpoint.name}.log",
                args,
                extra + [
                    "--trace-dir", str(output / "traces" / checkpoint.name),
                    "--trace-frames-every", str(args.trace_frames_every),
                ],
                stop,
            )
            print(
                f"{head}: {status}, {sweep.checkpoint_score(results)[1] - done} episodes "
                f"in {(time.perf_counter() - started) / 60:.1f} min"
            )
            if stop.is_set():
                print(f"{head}: stop requested. Reporting on what has run.")
                break
            if status != "ok" and sweep.checkpoint_score(results)[1] == done:
                print(
                    f"{head}: {status} with no episodes recorded. Stopping -- that is the "
                    f"setup, not the checkpoint. See {output / 'logs' / f'{checkpoint.name}.log'}"
                )
                break


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="mode", required=True)

    r = sub.add_parser("run", help="Sweep checkpoints over both zone sets, tracing.")
    r.add_argument("--checkpoints", type=Path,
                   default=Path.home() / "models/smolvla_fr3_1_2_3_6_7_8/checkpoints")
    r.add_argument("--namespace", type=str, default="/Sim_0/Scene_0")
    r.add_argument("--zones", type=str, default="1,2,3,6,7,8",
                   help="Cells the checkpoint TRAINED on.")
    r.add_argument("--ood", type=str, default="0,4,12,17",
                   help="Cells it never saw. Scored separately -- pooling the two gives a "
                        "number that is neither and that moves with the mix.")
    r.add_argument("--episodes", type=int, default=10, help="Episodes per zone.")
    r.add_argument("--seconds", type=float, default=60.0)
    r.add_argument("--only", type=str, default="")
    r.add_argument("--output", type=str, default="")
    r.add_argument("--timeout", type=float, default=8 * 3600)
    r.add_argument("--trace-frames-every", type=int, default=25,
                   help="Keep camera frames for every Nth episode (0 = never). Numbers "
                        "are always kept; frames are ~11 MB a rollout.")
    r.add_argument("--seed-base", type=int, default=0)

    p = sub.add_parser("report", help="Offline: csv, json and pdf from a finished study.")
    p.add_argument("directory", type=Path)

    g = sub.add_parser("gui", help="Online: step through every traced episode.")
    g.add_argument("directory", type=Path)
    g.add_argument("--trial", type=str, default=None)

    args, extra = parser.parse_known_args()
    if args.mode == "run":
        write_report(run(args, extra))
    elif args.mode == "report":
        write_report(args.directory.expanduser())
    else:
        open_gui(args.directory.expanduser(), args.trial)


if __name__ == "__main__":
    main()
