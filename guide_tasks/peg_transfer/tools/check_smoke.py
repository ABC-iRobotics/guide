"""Check a peg_transfer smoke recording (plan T12, gates 1-6) from its LeRobot dataset and provenance.

Run (the package's sources importable):
    ~/ros2_ws/.venv/bin/python guide_tasks/peg_transfer/tools/check_smoke.py ~/dataset/peg_transfer [--n 10]

Reads <path>/dataset_*/ (meta/info.json, meta/guide_info.json, meta/tasks.parquet, data/**/*.parquet) and
<path>.provenance.jsonl (one line per attempt, solve_task.generate). Prints each gate PASS/FAIL, the failed
attempts per subtask and each episode's subtask timeline; exits 1 when a gate fails. Gate 7 (the videos by
eye) is not checked here.
"""
import argparse
import collections
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from peg_transfer.scene import IMAGE_SIDE
from peg_transfer.solve_task import SEED_BLOCK, SUBTASKS

RETRY = {"grasp": "regrasp", "handover": "rereceive", "place": "replace"}  # a planned miss's recovery prompt


def active(rows, style, t):
    """The `style` prompt active at time t: the last of that style at or before it."""
    hits = [r for r in rows if r["style"] == style and r["timestamp"] <= t + 1e-6]
    return max(hits, key=lambda r: r["timestamp"])["content"] if hits else None


def load(path: Path):
    frames, infos = [], []
    for d in sorted(path.glob("dataset_*")):
        files = sorted((d / "data").glob("**/*.parquet"))
        if not files:
            continue  # a run that saved nothing
        df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
        tasks = pd.read_parquet(d / "meta" / "tasks.parquet")
        names = dict(zip(tasks["task_index"], tasks.index)) if "task_index" in tasks else {}
        df["task"] = df["task_index"].map(names)
        df["run"] = d.name
        frames.append(df)
        infos.append((json.loads((d / "meta" / "info.json").read_text()),
                      json.loads((d / "meta" / "guide_info.json").read_text())))
    return frames, infos


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path", type=Path)
    ap.add_argument("--n", type=int, default=10)
    args = ap.parse_args()
    path = args.path.expanduser()
    lines = [json.loads(s) for s in Path(f"{str(path).rstrip('/')}.provenance.jsonl").read_text().splitlines() if s]
    saved = [p for p in lines if p.get("saved")]
    frames, infos = load(path)
    episodes = [g for df in frames for _, g in df.groupby("episode_index", sort=True)]
    gates = {}

    # 1. episodes on disk, one saved provenance line each: seed = seed_base + episode * SEED_BLOCK + attempt
    seeds_ok = all(p["seed"] % SEED_BLOCK == p["attempt"] and p["seed"] // SEED_BLOCK == p["episode"] for p in saved)
    gates["1 episodes + provenance"] = (len(episodes) == args.n == len(saved) and seeds_ok
                                        and all("recipe" in p and "misses" in p for p in saved),
                                        f"{len(episodes)} episodes, {len(saved)} saved lines of {len(lines)}, "
                                        f"seeds {[p['seed'] for p in saved]}")

    # 2. features in init.yaml's order, units, fps, cameras
    ok, why = True, []
    for info, guide in infos:
        f, ds = info["features"], guide["config"]["dataset"]
        for key, cfg in (("action", "action"), ("observation.state", "observations")):
            want = [next(iter(e)) for e in ds[cfg]]
            if f[key]["names"] != want:
                ok, why = False, why + [f"{key} names differ from init.yaml"]
            unit = f[key].get("info", {}).get("unit", {})
            bad = [k for k in want if k.endswith("main_insertion.pos") and unit.get(k) != "m"
                   or k.endswith("jaw.pos") and unit.get(k) != "rad"]
            if bad:
                ok, why = False, why + [f"{key} units {bad}"]
        cams = {k: f.get(f"observation.images.{k}", {}).get("shape") for k in ("left", "right")}
        if info["fps"] != 20 or any(s != [480, 720, 3] for s in cams.values()):
            ok, why = False, why + [f"fps {info['fps']} cameras {cams}"]
    gates["2 features/units/fps/cameras"] = (ok and bool(infos), "; ".join(why) or "T9 order, m/rad, 20 fps, 720x480")

    # 3. language on every frame; the task names the plan's colour and sides
    ok, why, timelines = True, [], []
    for ep, prov in zip(episodes, saved):
        rows = list(ep["language_persistent"].iloc[0]) if "language_persistent" in ep else []
        ts = ep["timestamp"].to_numpy()
        missing = sum(not (active(rows, "procedure", t) and active(rows, "subtask", t)) for t in ts)
        want = (f"Move the {prov['colour']} block from the {IMAGE_SIDE[prov['giver']]} to the "
                f"{IMAGE_SIDE['psm1' if prov['giver'] == 'psm2' else 'psm2']} side.")
        tasks = set(ep["task"])
        # a parallel block's subtask names both arms, image right (psm1) first: never half of it
        half = [r["content"] for r in rows if r["style"] == "subtask" and ": " in r["content"]
                and not (r["content"].startswith("Right: ") and " Left: " in r["content"])]
        if missing or tasks != {want} or half:
            ok, why = False, why + [f"ep {ep['episode_index'].iloc[0]}: {missing} frames unlabelled, task {tasks}, half prompts {half}"]
        timelines.append([(round(r["timestamp"], 1), r["content"]) for r in rows if r["style"] == "subtask"])
    gates["3 language"] = (ok and bool(episodes), "; ".join(why) or "procedure+task+subtask on every frame")

    # 4. DART: perturbed action pose deltas differ from the measured ones; all finite
    ok, why, diffs = True, [], []
    for ep, prov in zip(episodes, saved):
        a, s = np.stack(ep["action"].to_numpy()), np.stack(ep["observation.state"].to_numpy())
        if not (np.isfinite(a).all() and np.isfinite(s).all()):
            ok, why = False, why + [f"ep {ep['episode_index'].iloc[0]}: NaN/inf"]
        info = infos[0][0]["features"]
        an, sn = info["action"]["names"], info["observation.state"]["names"]
        for arm in ("psm1", "psm2"):
            ai, si = [an.index(f"{arm}.{k}") for k in "xyz"], [sn.index(f"{arm}.{k}") for k in "xyz"]
            d = np.abs(a[:-1, ai] - np.diff(s[:, si], axis=0)).mean()
            diffs.append(d)
            if not prov.get("clean") and d < 1e-6:
                ok, why = False, why + [f"ep {ep['episode_index'].iloc[0]} {arm}: action = measured delta"]
    gates["4 DART + finite"] = (ok and bool(episodes), "; ".join(why) or
                                f"mean |action - measured| {np.mean(diffs) * 1e3:.4f} mm/frame" if diffs else "")

    # 5. success rate, misses and their recoveries in the timeline
    first = sum(p["attempt"] == 0 for p in saved)
    failed = collections.Counter(" < ".join(p.get("failed") or ["?"]) for p in lines if not p.get("saved"))
    unseen = [(p["episode"], m) for p, tl in zip(saved, timelines) for m in p["misses"]
              if not any(SUBTASKS[RETRY[m]] in c for _, c in tl)]  # in a parallel block: within its joined prompt
    rate = len(saved) / max(len(lines), 1)
    gates["5 success + recoveries"] = (rate >= 0.8 and not unseen,
                                       f"{len(saved)}/{len(lines)} attempts saved ({first} first-try), "
                                       f"misses without their retry in the timeline: {unseen or 'none'}")

    # 6. both directions twice; episodes <= 90 s
    dirs = collections.Counter(p["giver"] for p in saved)
    longest = max((len(ep) / 20.0 for ep in episodes), default=math.inf)
    gates["6 directions + length"] = (min(dirs.get("psm1", 0), dirs.get("psm2", 0)) >= 2 and longest <= 90.0,
                                      f"givers {dict(dirs)}, longest {longest:.1f} s")

    for name, (ok, detail) in gates.items():
        print(f"{'PASS' if ok else 'FAIL'}  {name}: {detail}")
    print("failed attempts by node (innermost < its composites):")
    for chain, k in failed.most_common():
        print(f"  {k} x {chain}")
    for p, tl in zip(saved, timelines):
        print(f"ep {p['episode']} seed {p['seed']} {p['recipe']} {p['giver']} misses {p['misses']}: "
              + " | ".join(f"{t} {c}" for t, c in tl))
    sys.exit(0 if all(ok for ok, _ in gates.values()) else 1)


if __name__ == "__main__":
    main()
