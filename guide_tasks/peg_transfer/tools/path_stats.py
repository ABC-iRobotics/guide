"""How much a peg_transfer recording's tool tips stray and blend: per move, the largest distance of the tip from
the straight line between the move's start and goal; at each free end (a blend), how far the tip passes
from that corner; at each contact (a slow precise pose move), how far the tip stopped from its goal.

Run (the package's sources and guide_motion importable):
    ~/ros2_ws/.venv/bin/python guide_tasks/peg_transfer/tools/path_stats.py ~/dataset/peg_transfer_v2 \
        [--twin ~/dataset/peg_transfer_clean] [--plot out.png --episode 0] [--gate]

Reads <path>/dataset_*/ (observation.state psmN.x/y/z: each tip in its arm's base frame, 20 fps) and
<path>.provenance.jsonl, whose i-th saved attempt is episode i and lists its arm goals (`moves`,
solve_task.log_moves; a joint goal's tip by robots/psm's forward kinematics). Goals are matched to the path
in order: a precise one where the tip first comes within TOL of it, a free one where it passes closest.
--twin: a recording of the same seeds (a clean run): its moves' numbers alongside, and each move's
largest distance from the twin's path (the noise alone). --plot: both tips' paths seen from above (world,
davinci_sim's RCM poses), this run's over the twin's. --gate: exit 1 unless the perturbed moves' median
deviation is >= 3 mm and the free ends' median corner cut >= 5 mm.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

TOL = 0.005  # m: a precise goal is reached (the robot's own reach tolerance)
ARMS = ("psm1", "psm2")
SCENE_INFO = Path("~/ros2_ws/src/davinci_sim/scene/scene_info.json").expanduser()


def load(path):
    """[(episode's provenance record, {arm: tip path (n, 3) m})] of the saved episodes, in order."""
    path = Path(path).expanduser()
    saved = [r for r in map(json.loads, Path(f"{str(path).rstrip('/')}.provenance.jsonl").read_text().splitlines()) if r["saved"]]
    root = next(path.glob("dataset_*"))
    names = json.loads((root / "meta/info.json").read_text())["features"]["observation.state"]["names"]
    data = pd.concat(pd.read_parquet(f) for f in sorted(root.glob("data/**/*.parquet")))
    out = []
    for ep, rec in enumerate(saved):
        s = np.stack(data[data.episode_index == ep]["observation.state"].to_numpy())
        out.append((rec, {a: s[:, [names.index(f"{a}.{k}") for k in "xyz"]] for a in ARMS}))
    return out


def goals(rec, arm):
    """The arm's goals: (tip position, move record), a joint goal's tip by forward kinematics."""
    from guide_motion.model import Model

    model = Model.load("psm")
    out = []
    for m in rec.get("moves", []):
        if m["robot"] == arm:
            p = m["position"] if "position" in m else model.ee_pose(np.array([m["joints"][f"{j}.pos"] for j in model.arm_joints]))[0]
            out.append((np.asarray(p, float), m))
    return out


def free(m) -> bool:
    return "position" in m and m.get("precise_end", True) is False


def match(P, gs) -> list:
    """Each goal's frame: a precise goal's first approach within TOL after the last one (then down to its
    local minimum; never within TOL: the closest), a free goal's closest point between its neighbours'."""
    idx, cur = [None] * len(gs), 0
    for k, (g, m) in enumerate(gs):
        if free(m):
            continue
        d = np.linalg.norm(P[cur:] - g, axis=1)
        near = np.flatnonzero(d < TOL)
        j = int(near[0]) if len(near) else int(np.argmin(d))
        while j + 1 < len(d) and d[j + 1] < d[j]:
            j += 1
        idx[k] = cur = cur + j
    for k, (g, m) in enumerate(gs):
        if idx[k] is None:
            lo = max([i for i in idx[:k] if i is not None], default=0)
            hi = min([i for i in idx[k + 1:] if i is not None], default=len(P) - 1)
            idx[k] = lo + int(np.argmin(np.linalg.norm(P[lo:hi + 1] - g, axis=1)))
    return idx


def to_segment(X, a, b) -> np.ndarray:
    """Each point's distance from the segment a-b."""
    ab = b - a
    t = np.clip((X - a) @ ab / max(ab @ ab, 1e-12), 0.0, 1.0)
    return np.linalg.norm(X - (a + t[:, None] * ab), axis=1)


def to_path(X, Q) -> np.ndarray:
    """Each point's distance from the polyline Q."""
    return np.min([to_segment(X, a, b) for a, b in zip(Q[:-1], Q[1:])], axis=0) if len(Q) > 1 else np.linalg.norm(X - Q[0], axis=1)


def kind(m) -> str:
    if "joints" in m:
        return "joint"
    if m.get("perturb"):
        return "perturbed"
    if m.get("precise_end", True) and m.get("speed", 1.0) <= 0.05:
        return "contact"
    return "straight"


def moves(rec, P, twin=None):
    """Per move of each arm (after its first): dict(arm, k, kind, free, dev, cut, end, noise), metres."""
    rows = []
    for arm in ARMS:
        gs = goals(rec, arm)
        idx = match(P[arm], gs)
        tw = None
        if twin is not None:
            tg = goals(twin[0], arm)
            if [kind(m) for _, m in tg] == [kind(m) for _, m in gs]:
                tw = (twin[1][arm], match(twin[1][arm], tg))
        for k in range(1, len(gs)):
            (a, _), (b, m) = gs[k - 1], gs[k]
            X = P[arm][idx[k - 1]:idx[k] + 1]
            row = {"arm": arm, "k": k, "kind": kind(m), "free": free(m), "dev": float(to_segment(X, a, b).max()),
                   "end": float(np.linalg.norm(P[arm][idx[k]] - b))}
            if free(m) and k + 1 < len(gs):
                row["cut"] = float(np.linalg.norm(P[arm][idx[k - 1]:idx[k + 1] + 1] - b, axis=1).min())
            if tw is not None:
                row["noise"] = float(to_path(X, tw[0][tw[1][k - 1]:tw[1][k] + 1]).max())
            rows.append(row)
    return rows


def summary(rows) -> dict:
    med = lambda xs: round(float(np.median(xs)) * 1e3, 2) if len(xs) else None
    pert = [r for r in rows if r["kind"] == "perturbed"]
    return {"moves": len(rows), "perturbed": len(pert),
            "dev_perturbed_mm": med([r["dev"] for r in pert]),
            "noise_perturbed_mm": med([r["noise"] for r in pert if "noise" in r]),
            "dev_straight_mm": med([r["dev"] for r in rows if r["kind"] in ("straight", "contact")]),
            "cut_mm": med([r["cut"] for r in rows if "cut" in r]),
            "cut_n": sum("cut" in r for r in rows),
            "contact_end_max_mm": round(max([r["end"] for r in rows if r["kind"] == "contact"], default=0.0) * 1e3, 2)}


def plot(out, eps, twin, episode):
    """Both tips from above (world x, y), this run's episode over the twin's."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    info = json.loads(SCENE_INFO.read_text())["arms"]
    fig, ax = plt.subplots(figsize=(8, 7))
    for (rec, P), style, label in ((twin[episode] if twin else None, "--", "clean twin"), (eps[episode], "-", "this run")):
        if rec is None:
            continue
        for arm, colour in zip(ARMS, ("tab:blue", "tab:orange")):
            T = np.array(info[arm.upper()]["rcm_world"])
            W = P[arm] @ T[:3, :3].T + T[:3, 3]
            ax.plot(W[:, 0] * 1e3, W[:, 1] * 1e3, style, color=colour, lw=1.2 if style == "-" else 1.0,
                    alpha=1.0 if style == "-" else 0.6, label=f"{arm} {label}")
            G = np.array([g for g, m in goals(rec, arm)]) @ T[:3, :3].T + T[:3, 3]
            if style == "-":
                ax.plot(G[:, 0] * 1e3, G[:, 1] * 1e3, "x", color=colour, ms=5)
    ax.set_aspect("equal")
    ax.set_xlabel("world x (mm)")
    ax.set_ylabel("world y (mm)")
    ax.set_title(f"episode {episode}: tool tips from above (x: goals)")
    ax.legend(fontsize=8)
    fig.savefig(out, dpi=130, bbox_inches="tight")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("path")
    parser.add_argument("--twin")
    parser.add_argument("--plot")
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--gate", action="store_true")
    args = parser.parse_args()
    eps = load(args.path)
    twin = load(args.twin) if args.twin else None
    rows, twin_rows = [], []
    for ep, (rec, P) in enumerate(eps):
        pair = twin[ep] if twin and ep < len(twin) and twin[ep][0]["seed"] == rec["seed"] else None
        r = moves(rec, P, pair)
        rows += r
        line = {"episode": ep, "seed": rec["seed"], "recipe": rec["recipe"], "clean": rec["clean"], **summary(r)}
        if pair:
            t = moves(*pair)
            twin_rows += t
            line["twin"] = {"clean": pair[0]["clean"], **summary(t)}
        print(json.dumps(line))
    total = {"all": summary(rows), **({"twin": summary(twin_rows)} if twin_rows else {})}
    print(json.dumps(total))
    if args.plot:
        plot(args.plot, eps, twin, args.episode)
    if args.gate:
        s = total["all"]
        ok = (s["dev_perturbed_mm"] or 0) >= 3.0 and (s["cut_mm"] or 0) >= 5.0
        print("GATE", "PASS" if ok else "FAIL")
        raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
