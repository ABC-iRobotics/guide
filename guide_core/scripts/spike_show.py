"""Spike: show a few Replicator-YAML randomizations of block_bin in the Isaac viewport.

Opens Isaac Sim with its window (needs DISPLAY), registers the checked-out block_bin, then
randomizes every few seconds — free draws first, then zoned ones — capturing the viewport
to PNG after each layout. Closes on its own.

  DISPLAY=:1 PYTHONPATH=<guide>/guide_core ~/ros2_ws/.venv/bin/python \
      <guide>/guide_core/scripts/spike_show.py --task <guide>/guide_tasks/block_bin --out /tmp/show
"""
from __future__ import annotations

import argparse
import logging
import shutil
import tempfile
import time
from pathlib import Path

import yaml


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--scenes", type=int, default=1)
    ap.add_argument("--free", type=int, default=4, help="free-draw layouts to show")
    ap.add_argument("--zones", default="0,7,19", help="zoned layouts to show after the free ones")
    ap.add_argument("--hold", type=float, default=3.0, help="seconds to hold each layout")
    ap.add_argument("--linger", type=float, default=15.0, help="seconds to keep the window at the end")
    args = ap.parse_args()

    src = Path(args.task).resolve()
    task = Path(tempfile.mkdtemp()) / src.name  # flat layout, see spike_replicator.py
    task.mkdir()
    shutil.copy(src / src.name / "scene.py", task / "scene.py")
    shutil.copytree(src / "config", task / "config")
    shutil.copytree(src / "assets", task / "assets")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    cfg = yaml.safe_load((Path(__file__).resolve().parents[1] / "config" / "init.yaml").read_text())
    cfg.setdefault("startup", {})["headless"] = False

    from guide_core.core.guide_simulator import GUIDESimulator

    logging.basicConfig(level=logging.INFO)
    sim = GUIDESimulator(sim_id=0, namespace="Sim_0")
    sim.init_runtime(config=cfg, logger=logging.getLogger("show"))
    sim.init_scene_manager()
    rt = sim._runtime
    sids = [rt._cmd_register_scene(str(task))[0] for _ in range(args.scenes)]
    rt._cmd_start()
    rt.update(60)
    scenes = {sid: sim._scene_manager._scenes[sid] for sid in sids}
    print("[show] ROS 2 image topics: " + ", ".join(f"/Sim_0/Scene_{sid}/cam_{c}" for sid in sids for c in ("top", "base", "wrist")), flush=True)

    from omni.kit.viewport.utility import capture_viewport_to_file, get_active_viewport

    viewport = get_active_viewport()

    def hold_and_capture(label: str) -> None:
        end = time.time() + args.hold
        while time.time() < end:  # keep rendering so the window stays live
            rt.update(1)
        path = out / f"{label}.png"
        capture_viewport_to_file(viewport, str(path))
        rt.update(10)  # let the capture flush
        tasks = {sid: getattr(scenes[sid], "task", "") for sid in sids}
        print(f"[show] {label}: {tasks} -> {path}", flush=True)

    hold_and_capture("00_initial")
    for i in range(args.free):
        for sid in sids:
            rt._cmd_randomize_scene(scene_id=sid)
        hold_and_capture(f"{i + 1:02d}_free")
    for z in (int(x) for x in args.zones.split(",") if x.strip()):
        for sid in sids:
            rt._cmd_randomize_scene(scene_id=sid, use_zone=True, zone=z)
        hold_and_capture(f"zone_{z:02d}")

    end = time.time() + args.linger
    while time.time() < end:
        rt.update(1)
    rt.simulation_app.close()


if __name__ == "__main__":
    main()
