"""Spike: measure Replicator-YAML randomization against the instruction executor.

Runs a headless Isaac Sim from the checked-out guide_core (not the installed one), registers
the checked-out block_bin by path (once per ``--scenes``), and randomizes it repeatedly. With
``--legacy`` the task is staged with the given instruction-dialect randomize.yaml instead, so
both paths are measured by the same code. Poses are read from PhysX, not from USD/Fabric.

  PYTHONPATH=<guide>/guide_core ~/ros2_ws/.venv/bin/python <guide>/guide_core/scripts/spike_replicator.py \
      --task <guide>/guide_tasks/block_bin [--legacy <old randomize.yaml>] [--episodes 20] [--scenes 2]
"""
from __future__ import annotations

import argparse
import json
import logging
import shutil
import tempfile
import time
from pathlib import Path

import numpy as np
import yaml

COLORS = ("red", "yellow", "green", "blue")


def physx_local(prim: str) -> np.ndarray:
    """Position local to the prim's parent as PhysX sees it (RigidPrim view), i.e. the frame
    set_local_poses and the yaml bounds are written in -- not the USD/Fabric xformOp."""
    from isaacsim.core.prims import RigidPrim, XFormPrim
    from scipy.spatial.transform import Rotation as R

    try:
        view = RigidPrim(prim_paths_expr=prim)
    except Exception:  # not a rigid body (the bins): the xform is all there is
        view = XFormPrim(prim_paths_expr=prim)
    world = np.asarray(view.get_world_poses()[0], dtype=float).reshape(-1)[:3]
    parent = XFormPrim(prim_paths_expr=prim.rsplit("/", 1)[0])
    p_pos, p_quat = parent.get_world_poses()
    p_pos = np.asarray(p_pos, dtype=float).reshape(-1)[:3]
    w, x, y, z = np.asarray(p_quat, dtype=float).reshape(-1)[:4]
    return R.from_quat([x, y, z, w]).inv().apply(world - p_pos)


def local_scale(prim: str) -> np.ndarray:
    from isaacsim.core.prims import XFormPrim

    return np.asarray(XFormPrim(prim_paths_expr=prim).get_local_scales(), dtype=float).reshape(-1)[:3]


def stage_task(src: Path, legacy: str | None, legacy_reset: str | None = None) -> Path:
    """SceneManager.add_scene's filesystem branch wants scene.py beside config/ and assets/."""
    task = Path(tempfile.mkdtemp()) / src.name
    task.mkdir()
    shutil.copy(src / src.name / "scene.py", task / "scene.py")
    shutil.copytree(src / "config", task / "config")
    shutil.copytree(src / "assets", task / "assets")
    if legacy:
        shutil.copy(legacy, task / "config" / "randomize.yaml")
    if legacy_reset:
        shutil.copy(legacy_reset, task / "config" / "reset.yaml")
    return task


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--legacy", default=None, help="instruction-dialect randomize.yaml to test instead")
    ap.add_argument("--legacy-reset", default=None, help="instruction-dialect reset.yaml to test instead")
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--scenes", type=int, default=1)
    ap.add_argument("--reset", action="store_true", help="also measure the reset file")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg = yaml.safe_load((Path(__file__).resolve().parents[1] / "config" / "init.yaml").read_text())
    cfg.setdefault("startup", {})["headless"] = True

    from guide_core.core.guide_simulator import GUIDESimulator

    logging.basicConfig(level=logging.INFO)
    sim = GUIDESimulator(sim_id=0, namespace="Sim_0")
    sim.init_runtime(config=cfg, logger=logging.getLogger("spike"))
    sim.init_scene_manager()
    rt = sim._runtime

    task = stage_task(Path(args.task).resolve(), args.legacy, args.legacy_reset)
    sids, t_register = [], []
    for _ in range(args.scenes):
        t0 = time.perf_counter()
        sid, _ = rt._cmd_register_scene(str(task))
        t_register.append(round(time.perf_counter() - t0, 2))
        sids.append(sid)
    rt._cmd_start()
    rt.update(5)

    scenes = {sid: sim._scene_manager._scenes[sid] for sid in sids}
    dialect = "replicator" if scenes[sids[0]].replicator_files else "instructions"
    blocks = {sid: [f"/Scene_{sid}/blocks/{c}_block" for c in COLORS] for sid in sids}

    def randomize(sid, **kw):
        t = time.perf_counter()
        rt._cmd_randomize_scene(scene_id=sid, **kw)
        return time.perf_counter() - t

    def layout(sid):
        return {p: physx_local(p) for p in blocks[sid]}

    def inside(xy, low, high):
        return bool(np.all(xy >= low[:2] - 1e-3) and np.all(xy <= high[:2] + 1e-3))

    def rtf(frames: int = 120) -> float:
        """Sim seconds per wall second while the app updates freely (physics at step_freq)."""
        hz = float(cfg.get("startup", {}).get("step_freq", 60.0))
        t = time.perf_counter()
        rt.update(frames)
        return round((frames / hz) / (time.perf_counter() - t), 3)

    report = {"dialect": dialect, "scenes": args.scenes, "register_s": t_register, "rtf_idle": rtf()}
    scale_before = {sid: local_scale(blocks[sid][0]) for sid in sids}

    for sid in sids:
        grid = scenes[sid]._grid
        # determinism: the same seed twice gives the same layout and the same record
        randomize(sid, seed=123)
        a = layout(sid)
        rec_a = json.loads(sim._scene_manager.get_last_record_json(sid))
        randomize(sid, seed=777)
        randomize(sid, seed=123)
        b = layout(sid)
        rec_b = json.loads(sim._scene_manager.get_last_record_json(sid))
        # a different seed moves things; four blocks land at four places
        randomize(sid, seed=1)
        c = layout(sid)
        pts = np.array([a[p][:2] for p in blocks[sid]])
        dists = [np.linalg.norm(pts[i] - pts[j]) for i in range(4) for j in range(i + 1, 4)]
        # cost
        times = [randomize(sid) for _ in range(args.episodes)]
        # zones
        zone_ok = {}
        for z in (0, 7, 19):
            randomize(sid, use_zone=True, zone=z)
            low, high = grid.cell_bounds(z)
            zone_ok[z] = inside(physx_local(scenes[sid].zone_target())[:2], low, high)
        # region + scale after a free draw
        randomize(sid)
        report[f"scene_{sid}"] = {
            "deterministic_same_seed": bool(all(np.allclose(a[p], b[p], atol=1e-3) for p in blocks[sid])),
            "record_identical": rec_a.get("record") == rec_b.get("record"),
            "layout_changes_with_seed": bool(any(not np.allclose(a[p], c[p], atol=1e-3) for p in blocks[sid])),
            "per_prim_independent": bool(min(dists) > 1e-3),
            "randomize_ms_mean": round(1000 * float(np.mean(times)), 1),
            "randomize_ms_p95": round(1000 * float(np.percentile(times, 95)), 1),
            "zone_target_in_cell": zone_ok,
            "poses_inside_region_after_call": bool(all(inside(physx_local(p)[:2], grid.low, grid.high) for p in blocks[sid])),
            "scale_preserved": bool(np.allclose(local_scale(blocks[sid][0]), scale_before[sid])),
            "block_scale": np.round(local_scale(blocks[sid][0]), 4).tolist(),
            "sample_record_keys": sorted(rec_a.get("record", {}).get("values", {}).keys())[:6],
        }

    # RTF while randomizing every 60 frames (one call per sim second), all scenes
    hz = float(cfg.get("startup", {}).get("step_freq", 60.0))
    t = time.perf_counter()
    for _ in range(5):
        for sid in sids:
            rt._cmd_randomize_scene(scene_id=sid)
        rt.update(60)
    report["rtf_randomizing_every_60_frames"] = round((5 * 60 / hz) / (time.perf_counter() - t), 3)

    if args.reset:
        import re as _re

        home = yaml.safe_load((task / "config" / "reset.yaml").read_text())
        rt.update(30)
        for sid in sids:
            want = {}
            if "instructions" in home:
                for ins in home["instructions"]:
                    kw = ins.get("kwargs", {})
                    if ins["cmd"] == "set_local_poses":
                        want[f"/Scene_{sid}{kw['prim_path']}"] = np.asarray(kw["pose"]["position"]["value"], dtype=float)
            else:  # every registered body: with.<group> -> modify.pose constants
                for group in home.values():
                    for body in (group.get("randomizer.register") or {}).values():
                        for key, entry in body.items():
                            m = _re.match(r"with\.(\w+)$", key)
                            if m and isinstance(entry, dict) and "modify.pose" in entry:
                                pat = body[m.group(1)]["get.prims"]["path_pattern"].rstrip("$")
                                want[f"/Scene_{sid}{pat}"] = np.asarray(entry["modify.pose"]["position"], dtype=float)
            robot = rt._robots[f"/Scene_{sid}/fr3"]
            try:  # Isaac 6.0 Robot has no is_initialized; initialize() is idempotent enough
                robot.initialize()
            except Exception as e:  # noqa: BLE001
                print(f"[spike] robot.initialize: {e}", flush=True)
            print(f"[spike] dof_names Scene_{sid}: {list(robot.dof_names)}", flush=True)
            # disturb: a free randomize, then move the arm off home
            randomize(sid)
            rt.update(30)
            t = time.perf_counter()
            rt._cmd_reset_scene(scene_id=sid)
            ms = 1000 * (time.perf_counter() - t)
            rt.update(2)
            def at_home(p):
                # reset.yaml parks bin_1 on the block row, so PhysX pushes the blocks 1-3 cm
                # aside on either dialect; a block written at z=0.025 rests at 0.041.
                got = physx_local(p)
                return bool(np.allclose(got[:2], want[p][:2], atol=0.05) and abs(got[2] - want[p][2]) < 0.03)

            pose_ok = {p: at_home(p) for p in want}
            print(f"[spike] after reset: " + "; ".join(f"{p.split('/')[-1]} got {physx_local(p).round(3).tolist()} want {want[p].tolist()}" for p in want), flush=True)
            rt.update(30)  # still at home after half a second: no residual motion
            settled = {p: at_home(p) for p in want}
            # arm: frames until every DOF is within 1 degree of the reset targets
            targets = np.deg2rad([0.0, -45.0, 0.0, -135.0, 0.0, 90.0, 45.0])
            frames = None
            for f in range(600):
                q = np.asarray(robot.get_joint_positions(), dtype=float).ravel()[:7]
                if np.all(np.abs(q - targets) < np.deg2rad(1.0)):
                    frames = f
                    break
                rt.update(1)
            # idempotence: reset -> randomize -> reset lands on the same poses
            randomize(sid)
            rt._cmd_reset_scene(scene_id=sid)
            rt.update(2)
            again = {p: at_home(p) for p in want}
            report[f"reset_scene_{sid}"] = {
                "poses_at_home": pose_ok,
                "still_at_home_after_30_frames": settled,
                "arm_frames_to_home_1deg": frames,
                "idempotent": all(again.values()),
                "reset_ms": round(ms, 1),
            }
        if len(sids) > 1:
            s0, s1 = sids[0], sids[1]
            randomize(s1)
            rt.update(60)
            before = layout(s1)
            rt._cmd_reset_scene(scene_id=s0)
            rt.update(2)
            after = layout(s1)
            report["reset_cross_talk_free"] = bool(all(np.allclose(before[p], after[p], atol=1e-3) for p in blocks[s1]))

    # cross-talk: randomizing one scene must not move another
    if len(sids) > 1:
        s0, s1 = sids[0], sids[1]
        rt.update(60)  # let PhysX finish resolving any overlapping blocks from the last draw
        before = layout(s1)
        randomize(s0, seed=99)
        after = layout(s1)
        report["cross_talk_free"] = bool(all(np.allclose(before[p], after[p], atol=1e-3) for p in blocks[s1]))

    print(json.dumps(report, indent=1))
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=1))
    rt.simulation_app.close()


if __name__ == "__main__":
    main()
