"""Spike: measure Replicator-YAML randomization against the instruction executor.

Runs a headless Isaac Sim from the checked-out guide_core (not the installed one), registers
the checked-out block_bin by path, and randomizes it repeatedly. With ``--legacy`` the same
task is copied to a temp dir with the given (instruction-dialect) randomize.yaml, so both
paths are measured by the same code.

  PYTHONPATH=<guide>/guide_core ~/ros2_ws/.venv/bin/python <guide>/guide_core/scripts/spike_replicator.py \
      --task <guide>/guide_tasks/block_bin [--legacy <old randomize.yaml>] [--episodes 20]
"""
from __future__ import annotations

import argparse
import json
import shutil
import tempfile
import time
from pathlib import Path

import numpy as np
import yaml

COLORS = ("red", "yellow", "green", "blue")


def local_xy(rt, prim: str) -> np.ndarray:
    pose = rt._cmd_get_local_poses(prim_path=prim)
    return np.asarray(pose.position.to_numpy(), dtype=float)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--legacy", default=None, help="instruction-dialect randomize.yaml to test instead")
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    task = Path(args.task).resolve()
    if args.legacy:
        tmp = Path(tempfile.mkdtemp()) / task.name
        shutil.copytree(task, tmp, ignore=shutil.ignore_patterns("__pycache__", "*.egg-info"))
        shutil.copy(args.legacy, tmp / "config" / "randomize.yaml")
        task = tmp

    cfg = yaml.safe_load((Path(__file__).resolve().parents[1] / "config" / "init.yaml").read_text())
    cfg.setdefault("startup", {})["headless"] = True

    from guide_core.core.guide_simulator import GUIDESimulator

    sim = GUIDESimulator(sim_id=0, namespace="Sim_0")
    sim.init_runtime(config=cfg)
    sim.init_scene_manager()
    rt = sim._runtime

    t0 = time.perf_counter()
    sid, _ = rt._cmd_register_scene(str(task))
    t_register = time.perf_counter() - t0
    rt._cmd_start()
    rt.update(5)
    scene = sim._scene_manager._scenes[sid]
    dialect = "replicator" if getattr(scene, "replicator_yaml", None) else "instructions"
    blocks = [f"/Scene_{sid}/blocks/{c}_block" for c in COLORS]

    def randomize(**kw):
        t = time.perf_counter()
        rt._cmd_randomize_scene(scene_id=sid, **kw)
        return time.perf_counter() - t

    # 1. determinism: the same seed twice gives the same layout and the same record
    randomize(seed=123)
    a = {b: local_xy(rt, b) for b in blocks}
    rec_a = json.loads(sim._scene_manager.get_last_record_json(sid))
    randomize(seed=777)  # disturb
    randomize(seed=123)
    b = {p: local_xy(rt, p) for p in blocks}
    rec_b = json.loads(sim._scene_manager.get_last_record_json(sid))
    deterministic = all(np.allclose(a[p], b[p], atol=1e-6) for p in blocks)
    same_record = rec_a.get("record") == rec_b.get("record")

    # 2. independence: four blocks, four distinct positions
    pts = np.array([a[p][:2] for p in blocks])
    dists = [np.linalg.norm(pts[i] - pts[j]) for i in range(4) for j in range(i + 1, 4)]
    independent = min(dists) > 1e-3

    # 3. cost per episode (free draws)
    times = [randomize() for _ in range(args.episodes)]

    # 4. zones: the target block lands in its cell (local xy inside cell bounds)
    zone_ok = {}
    grid = getattr(scene, "_grid", None)
    for z in (0, 7, 19):
        randomize(use_zone=True, zone=z)
        target = scene.zone_target()
        low, high = grid.cell_bounds(z)
        xy = local_xy(rt, target)[:2]
        zone_ok[z] = bool(np.all(xy >= low[:2] - 1e-3) and np.all(xy <= high[:2] + 1e-3))

    # 5. write-back: poses read straight after the call sit inside the region
    randomize()
    region_lo, region_hi = (grid.low, grid.high) if grid is not None else (None, None)
    inside = (
        all(
            np.all(local_xy(rt, p)[:2] >= region_lo[:2] - 1e-3) and np.all(local_xy(rt, p)[:2] <= region_hi[:2] + 1e-3)
            for p in blocks
        )
        if grid is not None
        else None
    )

    report = {
        "dialect": dialect,
        "register_s": round(t_register, 2),
        "deterministic_same_seed": deterministic,
        "record_identical": same_record,
        "per_prim_independent": independent,
        "randomize_ms_mean": round(1000 * float(np.mean(times)), 1),
        "randomize_ms_p95": round(1000 * float(np.percentile(times, 95)), 1),
        "zone_target_in_cell": zone_ok,
        "poses_inside_region_after_call": inside,
        "sample_record_keys": sorted(rec_a.get("record", {}).get("values", {}).keys())[:8],
    }
    print(json.dumps(report, indent=1))
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=1))
    rt.simulation_app.close()


if __name__ == "__main__":
    main()
