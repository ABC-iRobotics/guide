"""Everything known about one trained policy, from a single sweep of the simulator.

This is the one entry point for validating a checkpoint directory. It drives the sim
once, writes a trace per episode, and derives every analysis offline from those traces
-- so re-analysing costs nothing and a killed run loses only the episode in flight.

    python -m block_bin.validate run    --checkpoints ~/models/M/checkpoints ...
    python -m block_bin.validate report ~/eval_studies/M
    python -m block_bin.validate gui    ~/eval_studies/M

Four questions, answered from the same data:

  ladder     where episodes stop -- grasped, lifted, released, correct cube -- split
             in-distribution vs out-of-distribution, with Wilson intervals.
  curve      success against training step, with paired McNemar between checkpoints
             on identical scenes, which is the only way a 6-point difference on 60
             episodes is distinguishable from noise.
  identity   when it takes the wrong cube, WHICH one and from where: zone-to-zone
             flow and a colour confusion matrix. Separates object selection from
             manipulation, which is the distinction the ladder alone cannot make.
  forensics  the decisive frames of each failure, plus an interactive replay.

SCORING NOTE. ``bin_contents()`` in eval_policy_pink asks the simulator for a COLLISION
between cube and bin prim, so a cube resting against the outer wall, or perched on the
rim, counts as delivered. Measured final offsets are bimodal -- genuinely inside is
|dy| <= 0.09 m, against the wall is |dy| >= 0.165 m, with nothing between -- so this
module scores containment geometrically from the bin poses already in the trace, and
reports the collision verdict alongside it for comparison with older runs.
"""

import argparse
import copy
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest

from block_bin import study, sweep_checkpoints as sweep
from block_bin.rollout_trace import load_trace

# Half-extent of a bin's interior. The two clusters sit at <=0.09 and >=0.165, so
# anything in that gap would be ambiguous; nothing observed lands there.
BIN_HALF = 0.13
CUBES = ("red_block", "green_block", "blue_block", "yellow_block")
COLOURS = ("red", "green", "blue", "yellow")


def contents_geometric(trace: dict) -> dict:
    """Which bin each cube ended INSIDE, by position, or None.

    The trace polls ``left_bin``/``right_bin`` alongside the cubes, so this needs no
    extra service calls and works on already-recorded runs.
    """
    polled = [step["poses"] for step in trace["steps"] if step.get("poses")]
    if not polled:
        return {}
    last, out = polled[-1], {}
    for cube in CUBES:
        out[cube] = None
        if cube not in last:
            continue
        centre = np.asarray(last[cube][:3], dtype=float)
        for key, side in (("left_bin", "left"), ("right_bin", "right")):
            if key not in last:
                continue
            delta = centre - np.asarray(last[key][:3], dtype=float)
            if abs(delta[0]) < BIN_HALF and abs(delta[1]) < BIN_HALF:
                out[cube] = side
                break
    return out


def first_lift_step(trace: dict, cube: str) -> int | None:
    """The step at which ``cube`` first rises LIFT_MIN above where it started."""
    base = None
    for index, step in enumerate(trace["steps"]):
        pose = (step.get("poses") or {}).get(cube)
        if not pose:
            continue
        if base is None:
            base = pose[2]
        elif pose[2] - base > study.LIFT_MIN:
            return index
    return None


def outcome(trace: dict, checkpoint: str = "") -> dict:
    """One episode, fully scored: rungs, what it carried, and when."""
    meta = trace["meta"]
    summary = meta.get("summary") or {}
    collision = study.rungs(trace)

    task = str(summary.get("task") or "")
    asked = next((c for c in COLOURS if c in task), None)
    side = "left" if "left" in task else ("right" if "right" in task else None)
    goal = study.GOAL_SIDE.get(summary.get("goal"))
    target = str(summary.get("target") or "").rsplit("/", 1)[-1] or None

    lifted = study.lifts(trace)
    inside = contents_geometric(trace)
    target_lifted = lifted.get(target, 0.0) > study.LIFT_MIN if target else False
    success = bool(target and target_lifted and inside.get(target) == goal)
    wrong = [c for c, rise in lifted.items()
             if c != target and rise > study.LIFT_MIN and inside.get(c) == goal]

    steps = trace["steps"]
    grasp = next((i for i, s in enumerate(steps)
                  if s.get("grip_command", 1.0) < study.CLOSE_COMMAND + 0.005), None)
    return {
        "checkpoint": checkpoint,
        "episode": meta.get("episode"),
        "zone": meta.get("zone"),
        "seed": meta.get("seed"),
        "task": task,
        "asked": asked,
        "side": side,
        "target": (target or "").replace("_block", "") or None,
        "steps": len(steps),
        "closed": bool(collision["closed"]),
        "lifted_any": bool(collision["lifted"]),
        "lifted_target": target_lifted,
        "binned_any": any(v is not None for v in inside.values()),
        "success": success,
        "wrong_cube": bool(wrong),
        "took": wrong[0].replace("_block", "") if wrong else None,
        "took_from_zone": None,          # filled by add_geometry
        "grasp_step": grasp,
        "decisive_step": (first_lift_step(trace, wrong[0]) if wrong
                          else (first_lift_step(trace, target) if target else None)),
        "success_collision": bool(collision["task_success"]),
        "wrong_collision": bool(collision["wrong_in_goal_bin"]),
        "binned_collision": bool(collision["binned"]),
        "trace": str(trace.get("directory", "")),
    }


def zone_of(world_x: float, world_y: float) -> int | None:
    """Which placement zone a world-frame point falls in, or None if outside the grid.

    ``/Scene_0/blocks`` carries a -90 degree z rotation, so a cube's world x is the
    grid's row axis and its world y runs opposite the column axis. Getting this
    backwards agrees with the recorded zone on about 4% of episodes; this agrees on 97%.
    """
    col = int((0.25 - world_y) // 0.1)
    row = int(world_x // 0.1)
    return row * 5 + col if 0 <= col < 5 and 0 <= row < 4 else None


def add_geometry(record: dict, trace: dict) -> dict:
    """Where the cube it actually carried was sitting when the episode began."""
    if not record["took"]:
        return record
    first = next((s["poses"] for s in trace["steps"] if s.get("poses")), None)
    pose = (first or {}).get(f"{record['took']}_block")
    if pose:
        record["took_from_zone"] = zone_of(pose[0], pose[1])
    return record


def gather(directory: Path, trained: set) -> pd.DataFrame:
    """Every traced episode of a study, scored. This is the only slow step."""
    rows = []
    for checkpoint, path in study.episode_traces(Path(directory)):
        try:
            trace = load_trace(path)
        except SystemExit as exc:                    # a run killed mid-episode
            print(f"  skipping {path.name}: {exc}", file=sys.stderr)
            continue
        rows.append(add_geometry(outcome(trace, checkpoint), trace))
    if not rows:
        raise SystemExit(f"No traced episodes under {directory}/traces.")
    frame = pd.DataFrame(rows)
    frame["split"] = np.where(frame["zone"].isin(trained),
                              "in-distribution", "out-of-distribution")
    frame["step"] = pd.to_numeric(frame["checkpoint"], errors="coerce")
    return frame.sort_values(["step", "zone", "seed"]).reset_index(drop=True)


# --------------------------------------------------------------------- 1. ladder
LADDER = [("closed", "grasped a cube"), ("lifted_any", "lifted it clear"),
          ("binned_any", "released into a bin"), ("success", "correct cube, correct bin")]


def funnel(frame: pd.DataFrame) -> pd.DataFrame:
    """Share of episodes reaching each rung, per checkpoint and split."""
    rows = []
    for (checkpoint, split), part in frame.groupby(["checkpoint", "split"], sort=True):
        n = len(part)
        wins = int(part["success"].sum())
        low, high = sweep.wilson(wins, n)
        row = {"checkpoint": checkpoint, "split": split, "episodes": n,
               "success": wins, "rate": wins / n, "ci_low": low, "ci_high": high,
               "wrong_cube": int(part["wrong_cube"].sum()),
               "success_collision": int(part["success_collision"].sum())}
        for key, _label in LADDER:
            row[key] = int(part[key].sum())
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------- 2. curve
def paired(frame: pd.DataFrame, split: str = "in-distribution") -> pd.DataFrame:
    """McNemar between every pair of checkpoints, on the scenes both actually ran.

    Unpaired rates move a lot on scene draw alone -- the same condition scored 57% and
    40% on two different seed blocks -- so any claim that one checkpoint beats another
    has to come from the matched scenes, not the headline percentages.
    """
    part = frame[frame["split"] == split]
    by = {c: g.set_index(["zone", "seed"])["success"]
          for c, g in part.groupby("checkpoint", sort=True)}
    rows = []
    names = sorted(by)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            shared = by[a].index.intersection(by[b].index)
            if not len(shared):
                continue
            first, second = by[a].loc[shared], by[b].loc[shared]
            only_a = int((first & ~second).sum())
            only_b = int((second & ~first).sum())
            p = binomtest(only_a, only_a + only_b, 0.5).pvalue if only_a + only_b else 1.0
            rows.append({"a": a, "b": b, "scenes": len(shared), "only_a": only_a,
                         "only_b": only_b, "p": p, "separable": p < 0.05})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------- 3. identity
def identity(frame: pd.DataFrame) -> dict:
    """What it grabbed instead: zone-to-zone flow and colour confusion.

    Only meaningful over the wrong-cube episodes, which is the failure the ladder
    reports but cannot explain.
    """
    wrong = frame[frame["wrong_cube"] & frame["took"].notna()]
    if wrong.empty:
        return {"flow": pd.DataFrame(), "confusion": pd.DataFrame(), "episodes": 0,
                "into_trained": 0, "chance": {}}
    flow = (wrong.dropna(subset=["took_from_zone"])
            .groupby(["zone", "took_from_zone"]).size()
            .reset_index(name="count").sort_values("count", ascending=False))
    confusion = (wrong.groupby(["asked", "took"]).size()
                 .reset_index(name="count").sort_values("count", ascending=False))
    # If it grabbed indifferently, each of the three non-target colours would come up
    # equally often; the gap between that and the observed counts is the colour bias.
    n = len(wrong)
    chance = {c: (n - int((wrong["asked"] == c).sum())) / 3 for c in COLOURS}
    return {"flow": flow, "confusion": confusion, "episodes": n,
            "into_trained": int(flow["count"].sum()) if not flow.empty else 0,
            "chance": chance}


# ------------------------------------------------------------------ 4. forensics
def forensics(frame: pd.DataFrame, limit: int = 40) -> pd.DataFrame:
    """The failures worth looking at, with the step where each went wrong."""
    bad = frame[~frame["success"]].copy()
    bad["kind"] = np.select(
        [bad["wrong_cube"], bad["binned_any"], bad["lifted_any"], bad["closed"]],
        ["wrong cube", "binned nothing asked for", "lifted, never released",
         "grasped, never lifted"],
        default="never grasped")
    bad["has_frames"] = [Path(t).joinpath("frames").is_dir() if t else False
                         for t in bad["trace"]]
    order = {"wrong cube": 0, "lifted, never released": 1, "grasped, never lifted": 2,
             "binned nothing asked for": 3, "never grasped": 4}
    bad["rank"] = bad["kind"].map(order)
    return (bad.sort_values(["has_frames", "rank"], ascending=[False, True])
            .head(limit)[["checkpoint", "zone", "seed", "split", "kind", "asked",
                          "took", "grasp_step", "decisive_step", "has_frames", "trace"]])


# ------------------------------------------------------------------- 5. outputs
def write_tables(directory: Path, frame: pd.DataFrame, rungs: pd.DataFrame,
                 pairs: pd.DataFrame, ident: dict, bad: pd.DataFrame, meta: dict) -> None:
    """CSV per table plus one JSON summary. Written before the PDF, which can fail."""
    directory.mkdir(parents=True, exist_ok=True)
    frame.to_csv(directory / "episodes.csv", index=False)
    rungs.to_csv(directory / "funnel.csv", index=False)
    if not pairs.empty:
        pairs.to_csv(directory / "paired.csv", index=False)
    if not ident["flow"].empty:
        ident["flow"].to_csv(directory / "wrong_cube_flow.csv", index=False)
        ident["confusion"].to_csv(directory / "colour_confusion.csv", index=False)
    bad.to_csv(directory / "failures.csv", index=False)

    best = rungs[rungs["split"] == "in-distribution"].sort_values("rate")
    summary = {
        "model": meta.get("model"),
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "episodes": int(len(frame)),
        "checkpoints": sorted(frame["checkpoint"].unique().tolist()),
        "zones_trained": sorted(meta.get("trained", [])),
        "zones_held_out": sorted(meta.get("held_out", [])),
        "scoring": {"containment": "geometric", "bin_half_extent_m": BIN_HALF,
                    "lift_min_m": study.LIFT_MIN},
        "best_in_distribution": (
            None if best.empty else
            {"checkpoint": best.iloc[-1]["checkpoint"],
             "success": int(best.iloc[-1]["success"]),
             "episodes": int(best.iloc[-1]["episodes"]),
             "rate": float(best.iloc[-1]["rate"]),
             "ci": [float(best.iloc[-1]["ci_low"]), float(best.iloc[-1]["ci_high"])]}),
        "funnel": json.loads(rungs.to_json(orient="records")),
        "paired": json.loads(pairs.to_json(orient="records")) if not pairs.empty else [],
        "wrong_cube_episodes": ident["episodes"],
        "separable_pairs": (int(pairs["separable"].sum()) if not pairs.empty else 0),
    }
    (directory / "validation.json").write_text(json.dumps(summary, indent=2, default=str))


def write_pdf(path: Path, frame: pd.DataFrame, rungs: pd.DataFrame,
              pairs: pd.DataFrame, ident: dict, meta: dict) -> None:
    """A standing report: funnel, curve, identity, and what is separable from what."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    with PdfPages(path) as pdf:
        # cover: the funnel, both splits
        fig, axes = plt.subplots(1, 2, figsize=(11.7, 8.3))
        fig.suptitle(f"{meta.get('model', 'policy')} — validation", fontsize=15)
        for ax, split in zip(axes, ("in-distribution", "out-of-distribution")):
            part = rungs[rungs["split"] == split]
            if part.empty:
                ax.set_axis_off()
                continue
            best = part.loc[part["rate"].idxmax()]
            values = [best[k] / best["episodes"] for k, _ in LADDER]
            ax.barh([lab for _, lab in LADDER], values,
                    color="#2C6A4E" if split == "in-distribution" else "#A8442A")
            ax.set_xlim(0, 1)
            ax.invert_yaxis()
            ax.set_title(f"{split}\nbest: {best['checkpoint']} · "
                         f"{best['success']}/{best['episodes']} = {best['rate']:.0%}")
            for i, v in enumerate(values):
                ax.text(min(v + 0.02, 0.92), i, f"{v:.0%}", va="center", fontsize=9)
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        # curve
        fig, ax = plt.subplots(figsize=(11.7, 8.3))
        for split, colour in (("in-distribution", "#2C6A4E"),
                              ("out-of-distribution", "#A8442A")):
            part = rungs[rungs["split"] == split].sort_values("checkpoint")
            if part.empty:
                continue
            steps = pd.to_numeric(part["checkpoint"], errors="coerce")
            ax.errorbar(steps, part["rate"],
                        yerr=[part["rate"] - part["ci_low"], part["ci_high"] - part["rate"]],
                        marker="o", capsize=4, color=colour, label=split)
        ax.set_xlabel("training steps")
        ax.set_ylabel("success")
        ax.set_ylim(0, 1)
        ax.legend()
        ax.grid(alpha=0.3)
        ax.set_title("Success against training step (95% Wilson)")
        pdf.savefig(fig)
        plt.close(fig)

        # what is separable
        fig, ax = plt.subplots(figsize=(11.7, 8.3))
        ax.set_axis_off()
        lines = ["Paired McNemar, in-distribution, identical scenes", ""]
        if pairs.empty:
            lines.append("Only one checkpoint — nothing to compare.")
        else:
            for _, r in pairs.iterrows():
                mark = "  <-- separable" if r["separable"] else ""
                lines.append(f"{r['a']} vs {r['b']}: {r['scenes']} scenes, "
                             f"{r['a']}-only {r['only_a']}, {r['b']}-only {r['only_b']}, "
                             f"p = {r['p']:.4f}{mark}")
            if not pairs["separable"].any():
                lines += ["", "No pair separates. The checkpoints tested are "
                          "statistically indistinguishable on this many episodes."]
        lines += ["", "", "Wrong-cube analysis", ""]
        if ident["episodes"]:
            lines.append(f"{ident['episodes']} episodes delivered the wrong cube.")
            if not ident["confusion"].empty:
                for _, r in ident["confusion"].head(8).iterrows():
                    lines.append(f"  asked {r['asked']:<7} took {r['took']:<7} x{r['count']}")
        else:
            lines.append("No wrong-cube deliveries.")
        ax.text(0.02, 0.98, "\n".join(lines), va="top", family="monospace", fontsize=9)
        pdf.savefig(fig)
        plt.close(fig)


def write_html(path: Path, frame: pd.DataFrame, rungs: pd.DataFrame,
               pairs: pd.DataFrame, ident: dict, meta: dict) -> None:
    """A self-contained page: the funnel, the curve, and what the failures were."""
    ident_rows = ""
    if not ident["confusion"].empty:
        for _, r in ident["confusion"].head(10).iterrows():
            ident_rows += (f"<tr><td>{r['asked']}</td><td>{r['took']}</td>"
                           f"<td class=n>{r['count']}</td></tr>")
    pair_rows = ""
    for _, r in pairs.iterrows():
        cls = ' class="sep"' if r["separable"] else ""
        pair_rows += (f"<tr{cls}><td>{r['a']} vs {r['b']}</td><td class=n>{r['scenes']}</td>"
                      f"<td class=n>{r['only_a']}</td><td class=n>{r['only_b']}</td>"
                      f"<td class=n>{r['p']:.4f}</td></tr>")

    bars = ""
    for split, colour in (("in-distribution", "var(--good)"),
                          ("out-of-distribution", "var(--bad)")):
        part = rungs[rungs["split"] == split]
        if part.empty:
            continue
        best = part.loc[part["rate"].idxmax()]
        bars += (f'<div class="lane"><h3>{split}</h3>'
                 f'<p class="sub">best checkpoint {best["checkpoint"]} '
                 f'· n={best["episodes"]}</p>')
        for key, label in LADDER:
            frac = best[key] / best["episodes"]
            bars += (f'<div class="row"><span class="lab">{label}</span>'
                     f'<span class="track"><span class="fill" style="width:{frac*100:.1f}%;'
                     f'background:{colour}"></span></span>'
                     f'<span class="pct n">{frac:.0%}</span></div>')
        bars += "</div>"

    curve = ""
    part = rungs[rungs["split"] == "in-distribution"].sort_values("checkpoint")
    if len(part) > 1:
        pts, labels = [], ""
        for i, (_, r) in enumerate(part.iterrows()):
            x = 140 + i / max(len(part) - 1, 1) * 1000
            y = 420 - r["rate"] * 340
            pts.append(f"{x:.0f},{y:.0f}")
            labels += (f'<circle cx="{x:.0f}" cy="{y:.0f}" r="7" fill="var(--paper)" '
                       f'stroke="var(--good)" stroke-width="3"/>'
                       f'<text x="{x:.0f}" y="{y-20:.0f}" text-anchor="middle" font-size="17" '
                       f'fill="var(--good)" font-weight="600">{r["rate"]:.0%}</text>'
                       f'<text x="{x:.0f}" y="452" text-anchor="middle" font-size="15" '
                       f'fill="var(--ink-soft)">{int(r["checkpoint"])//1000}k</text>')
        grid = "".join(f'<line x1="140" y1="{420-p/100*340:.0f}" x2="1140" '
                       f'y2="{420-p/100*340:.0f}" stroke="var(--rule)"/>'
                       f'<text x="120" y="{425-p/100*340:.0f}" text-anchor="end" '
                       f'font-size="14" fill="var(--ink-soft)">{p}%</text>'
                       for p in (0, 25, 50, 75, 100))
        curve = (f'<svg viewBox="0 0 1200 490" role="img" aria-label="In-distribution '
                 f'success against training step.">{grid}'
                 f'<polyline points="{" ".join(pts)}" fill="none" stroke="var(--good)" '
                 f'stroke-width="3"/>{labels}</svg>')

    empty_pairs = "<tr><td colspan=5>Single checkpoint — nothing to compare.</td></tr>"
    total = len(frame)
    ood = rungs[rungs["split"] == "out-of-distribution"]
    ood_success = int(ood["success"].sum()) if not ood.empty else 0
    ood_n = int(ood["episodes"].sum()) if not ood.empty else 0
    path.write_text(f"""<title>{meta.get('model', 'Policy')} Validation</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,600;12..96,700&family=Literata:opsz,wght@7..72,400&family=IBM+Plex+Mono:wght@400;600&display=swap">
<style>
  :root {{ --paper:#F2F1ED; --surface:#fff; --ink:#15161A; --ink-soft:#4C4E56;
          --rule:#DAD8D2; --good:#2C6A4E; --bad:#9B2B26; --sep-bg:#EFE6E5; }}
  @media (prefers-color-scheme: dark) {{ :root:not([data-theme="light"]) {{
    --paper:#0F1013; --surface:#17181D; --ink:#E7E7E4; --ink-soft:#A0A2AA;
    --rule:#2A2C32; --good:#6CC095; --bad:#E28378; --sep-bg:#2A1B1A; }} }}
  :root[data-theme="dark"] {{ --paper:#0F1013; --surface:#17181D; --ink:#E7E7E4;
    --ink-soft:#A0A2AA; --rule:#2A2C32; --good:#6CC095; --bad:#E28378; --sep-bg:#2A1B1A; }}
  *{{box-sizing:border-box}}
  body{{margin:0;background:var(--paper);color:var(--ink);font-family:Literata,Georgia,serif;
       font-size:17px;line-height:1.6}}
  .wrap{{max-width:1080px;margin:0 auto;padding:52px 24px 80px;display:flex;
        flex-direction:column;gap:34px}}
  h1{{font-family:'Bricolage Grotesque',sans-serif;font-size:clamp(34px,5vw,54px);
     font-weight:700;letter-spacing:-.02em;margin:0;line-height:1.05}}
  h2{{font-family:'Bricolage Grotesque',sans-serif;font-size:15px;font-weight:700;
     letter-spacing:.11em;text-transform:uppercase;margin:0;color:var(--ink-soft)}}
  h3{{font-family:'Bricolage Grotesque',sans-serif;font-size:20px;margin:0 0 2px}}
  p{{margin:0}} .eyebrow{{font-family:'Bricolage Grotesque',sans-serif;font-size:13px;
     font-weight:600;letter-spacing:.14em;text-transform:uppercase;color:var(--ink-soft)}}
  .n{{font-family:'IBM Plex Mono',monospace;font-variant-numeric:tabular-nums}}
  section{{background:var(--surface);border:1px solid var(--rule);padding:26px 24px;
          display:flex;flex-direction:column;gap:16px;overflow-x:auto}}
  .lane{{margin-bottom:18px}} .sub{{color:var(--ink-soft);font-size:14px;margin-bottom:10px}}
  .row{{display:flex;align-items:center;gap:14px;margin:7px 0}}
  .lab{{flex:0 0 250px;font-size:15px}}
  .track{{flex:1;height:24px;background:var(--rule);position:relative}}
  .fill{{position:absolute;left:0;top:0;bottom:0}}
  .pct{{flex:0 0 60px;text-align:right;font-weight:600}}
  table{{border-collapse:collapse;width:100%;min-width:460px}}
  th,td{{padding:8px 12px;text-align:right;border-bottom:1px solid var(--rule);font-size:15px}}
  th{{font-family:'Bricolage Grotesque',sans-serif;font-size:12px;letter-spacing:.06em;
     text-transform:uppercase;color:var(--ink-soft)}}
  td:first-child,th:first-child{{text-align:left}}
  tr.sep{{background:var(--sep-bg)}}
  svg{{width:100%;height:auto;display:block}}
  footer{{border-top:1px solid var(--rule);padding-top:16px;color:var(--ink-soft);font-size:14px}}
</style>
<div class="wrap">
<header>
  <p class="eyebrow">{meta.get('model', 'policy')} · {total} episodes · {len(rungs['checkpoint'].unique())} checkpoints</p>
  <h1>Validation report</h1>
  <p style="color:var(--ink-soft);font-size:19px">Trained zones {', '.join(map(str, sorted(meta.get('trained', []))))} ·
     unseen zones {', '.join(map(str, sorted(meta.get('held_out', []))))} ·
     containment scored geometrically.</p>
</header>
<section><h2>Where episodes end</h2>{bars}</section>
{f'<section><h2>Training curve · in-distribution</h2>{curve}</section>' if curve else ''}
<section><h2>Paired comparison · identical scenes</h2>
  <table><thead><tr><th>pair</th><th>scenes</th><th>first only</th>
  <th>second only</th><th>p</th></tr></thead>
  <tbody>{pair_rows or empty_pairs}</tbody></table>
  <p style="color:var(--ink-soft);font-size:14px">Shaded rows separate at p &lt; 0.05.
     Unpaired rates move on scene draw alone, so only these columns are evidence.</p>
</section>
<section><h2>Wrong-cube deliveries</h2>
  <p>{ident['episodes']} episodes carried a cube other than the one named, into the
     correct bin. Out-of-distribution success across this run:
     <span class="n">{ood_success}/{ood_n}</span>.</p>
  <table><thead><tr><th>asked</th><th>took</th><th>episodes</th></tr></thead>
  <tbody>{ident_rows or '<tr><td colspan=3>None.</td></tr>'}</tbody></table>
</section>
<footer>Success means the instructed cube lifted clear of the table and
  released <em>inside</em> the instructed bin, measured from the bin poses in
  each trace (|dx|,|dy| &lt; {BIN_HALF} m). The simulator's own check reports a
  collision, which also fires for a cube resting against an outer wall; that
  verdict is kept in <span class="n">episodes.csv</span> as
  <span class="n">success_collision</span>.</footer>
</div>
""")


# ----------------------------------------------------------------------- 6. run
# Measured on this machine: a full-length episode costs about 470 s of wall clock for
# 60 s of sim time, an effective real-time factor of 0.127 once resets and homing are
# counted. Budgeting by an assumed multiple of sim time is what truncated two runs, so
# the ceiling below is a backstop and --stall is the real supervisor.
EPISODE_STALL = 20 * 60


def budget(episodes: int, seconds: float, rtf: float = 0.11) -> float:
    """Backstop ceiling for a whole checkpoint, from the measured real-time factor.

    Deliberately pessimistic: rtf 0.11 against a measured 0.127, so a slow-but-working
    run is never killed. Progress is policed by ``stall`` instead, which does not need
    the cost of an episode to be known in advance.
    """
    return episodes * (seconds / rtf + 90) + 600


def zone_order(zones: str) -> list:
    """The zone each episode targets, in the order eval_policy will run them."""
    plan = []
    for item in zones.split(","):
        zone, _, count = item.partition(":")
        if zone.strip():
            plan += [int(zone)] * int(count or 0)
    return plan


def remaining(zones: str, episodes: int, done: int) -> tuple[str, int]:
    """The zone spec and seed offset that continue a part-finished checkpoint.

    Episodes are planned zone by zone with seeds running straight through, so the tail
    of the plan is itself a valid plan -- provided the seed base is shifted by however
    many already ran, or the resumed episodes would re-draw scenes already recorded.
    """
    full = zone_order(",".join(f"{z}:{episodes}" for z in zones.split(",") if z.strip()))
    tail = full[done:]
    if not tail:
        return "", done
    spec, run, current = [], 0, tail[0]
    for zone in tail:
        if zone == current:
            run += 1
            continue
        spec.append(f"{current}:{run}")
        current, run = zone, 1
    spec.append(f"{current}:{run}")
    return ",".join(spec), done


def sweep_once(args, checkpoints, output: Path, extra: list, wanted: int) -> None:
    """Run each checkpoint to completion, skipping whatever is already on disk."""
    with sweep.stop_service(args.namespace) as stop:
        for index, checkpoint in enumerate(reversed(checkpoints), start=1):
            results = output / "raw" / f"{checkpoint.name}.jsonl"
            done = len(results.read_text().splitlines()) if results.is_file() else 0
            head = f"[{index}/{len(checkpoints)}] {checkpoint.name}"
            if done >= wanted:
                print(f"{head}: {done} episodes already recorded, skipping.", flush=True)
                continue

            # Continue where a killed run stopped rather than repeating its episodes,
            # which would append duplicate scenes to the same results file.
            spec, offset = remaining(args.zones, args.episodes, done)
            child_args = copy.copy(args)
            child_args.zones = spec
            child_args.seed_base = args.seed_base + offset
            started = time.perf_counter()
            print(f"{head}: {wanted - done} episodes to go"
                  + (f" (resuming after {done}, zones {spec})" if done else "")
                  + f", stall guard {args.stall / 60:.0f} min", flush=True)
            status = sweep.run_checkpoint(
                checkpoint, results, output / "logs" / f"{checkpoint.name}.log", child_args,
                extra + ["--trace-dir", str(output / "traces" / checkpoint.name),
                         "--trace-frames-every", str(args.trace_frames_every),
                         "--trace-index-base", str(offset)],
                stop)
            ran = sweep.checkpoint_score(results)[1] - done
            print(f"{head}: {status}, {ran} episodes in "
                  f"{(time.perf_counter() - started) / 60:.1f} min", flush=True)
            if stop.is_set():
                print(f"{head}: stop requested; reporting on what has run.", flush=True)
                break
            if status != "ok" and ran == 0:
                print(f"{head}: {status} with nothing recorded -- that is the setup, "
                      f"not the checkpoint. See {output / 'logs' / f'{checkpoint.name}.log'}",
                      flush=True)
                break


def analyse(directory: Path, trained: set, held_out: set, model: str) -> pd.DataFrame:
    """Every analysis and every output, from traces already on disk."""
    frame = gather(directory, trained)
    rungs = funnel(frame)
    pairs = paired(frame)
    ident = identity(frame)
    bad = forensics(frame)
    meta = {"model": model, "trained": sorted(trained), "held_out": sorted(held_out)}

    write_tables(directory, frame, rungs, pairs, ident, bad, meta)
    try:
        write_pdf(directory / "validation.pdf", frame, rungs, pairs, ident, meta)
    except Exception as exc:                       # a plot failure must not lose the data
        print(f"PDF failed ({exc}); tables and JSON are written.", file=sys.stderr)
    write_html(directory / "validation.html", frame, rungs, pairs, ident, meta)

    print(f"\n{len(frame)} episodes over {frame['checkpoint'].nunique()} checkpoints")
    for split in ("in-distribution", "out-of-distribution"):
        part = rungs[rungs["split"] == split]
        if part.empty:
            continue
        print(f"\n{split}")
        for _, r in part.sort_values("checkpoint").iterrows():
            print(f"  {r['checkpoint']}  {r['success']:>3}/{r['episodes']:<3} "
                  f"{r['rate']:>4.0%} [{r['ci_low']:.0%}-{r['ci_high']:.0%}]   "
                  f"grasped {r['closed']:>3} lifted {r['lifted_any']:>3} "
                  f"binned {r['binned_any']:>3} wrong-cube {r['wrong_cube']:>3}"
                  + ("" if r["success"] == r["success_collision"]
                     else f"   (collision scoring would say {r['success_collision']})"))
    if not pairs.empty:
        sep = pairs[pairs["separable"]]
        print(f"\npaired: {len(sep)} of {len(pairs)} checkpoint pairs separate at p<0.05")
        for _, r in sep.iterrows():
            print(f"  {r['a']} vs {r['b']}: p = {r['p']:.4f}")
    if ident["episodes"]:
        print(f"\nwrong cube in {ident['episodes']} episodes; top confusions:")
        for _, r in ident["confusion"].head(5).iterrows():
            print(f"  asked {r['asked']:<7} took {r['took']:<7} x{r['count']}")
    print(f"\nwritten: {directory}/validation.json .html .pdf, episodes.csv, funnel.csv")
    return frame


def main():
    """Parse the command line and dispatch to run, report or gui."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="mode", required=True)

    r = sub.add_parser("run", help="Sweep checkpoints, trace every episode, report.")
    r.add_argument("--checkpoints", type=Path, required=True)
    r.add_argument("--namespace", type=str, default="/Sim_0/Scene_0")
    r.add_argument("--zones", type=str, required=True, help="Zones the policy TRAINED on.")
    r.add_argument("--ood", type=str, default="", help="Zones it never saw, scored apart.")
    r.add_argument("--episodes", type=int, default=10, help="Episodes per zone.")
    r.add_argument("--seconds", type=float, default=60.0)
    r.add_argument("--only", type=str, default="", help="Checkpoints to run, comma separated.")
    r.add_argument("--output", type=Path, required=True)
    r.add_argument("--timeout", type=float, default=0,
                   help="Backstop ceiling per checkpoint. 0 derives it from the "
                        "measured real-time factor.")
    r.add_argument("--stall", type=float, default=EPISODE_STALL,
                   help="Kill a checkpoint if no episode finishes in this many "
                        "seconds. This, not --timeout, is what catches a wedged run.")
    r.add_argument("--trace-frames-every", type=int, default=10)
    r.add_argument("--seed-base", type=int, default=0)

    a = sub.add_parser("report", help="Re-derive every analysis from existing traces.")
    a.add_argument("directory", type=Path)
    a.add_argument("--zones", type=str, default="")
    a.add_argument("--ood", type=str, default="")

    g = sub.add_parser("gui", help="Step through traced episodes interactively.")
    g.add_argument("directory", type=Path)
    g.add_argument("--trial", type=str, default=None)

    args, extra = parser.parse_known_args()

    if args.mode == "gui":
        study.open_gui(args.directory, args.trial)
        return

    if args.mode == "report":
        saved = args.directory / "validation.json"
        known = json.loads(saved.read_text()) if saved.is_file() else {}
        trained = ({int(z) for z in args.zones.split(",") if z.strip()}
                   or set(known.get("zones_trained", [])))
        held = ({int(z) for z in args.ood.split(",") if z.strip()}
                or set(known.get("zones_held_out", [])))
        analyse(args.directory, trained, held,
                known.get("model", args.directory.name))
        return

    trained = {int(z) for z in args.zones.split(",") if z.strip()}
    held = {int(z) for z in args.ood.split(",") if z.strip()}
    every = ",".join(str(z) for z in sorted(trained | held))
    checkpoints = sweep.select_checkpoints(
        sweep.discover_checkpoints(args.checkpoints), args.only)
    wanted = sweep.expected_episodes(every, args.episodes)
    args.timeout = args.timeout or budget(wanted, args.seconds)
    args.zones = every                       # what run_checkpoint forwards as --zone

    output = Path(args.output)
    for name in ("raw", "logs", "traces"):
        (output / name).mkdir(parents=True, exist_ok=True)
    model = args.checkpoints.parent.name

    print(f"{len(checkpoints)} checkpoints x {wanted} episodes "
          f"({len(trained)} trained + {len(held)} unseen zones)\n"
          f"stall guard {args.stall / 60:.0f} min, backstop "
          f"{args.timeout / 3600:.1f} h per checkpoint\n"
          f"stop early with: ros2 service call {args.namespace}/stop_sweep "
          f"std_srvs/srv/Trigger\n", flush=True)

    try:
        sweep_once(args, checkpoints, output, extra, wanted)
    finally:
        analyse(output, trained, held, model)


if __name__ == "__main__":
    main()
