"""Diagnostic: after the Replicator reset fires, where did the writes land?"""
from __future__ import annotations

import faulthandler
import logging
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import yaml

src = Path(sys.argv[1]).resolve()
task = Path(tempfile.mkdtemp()) / src.name
task.mkdir()
shutil.copy(src / src.name / "scene.py", task / "scene.py")
shutil.copytree(src / "config", task / "config")
shutil.copytree(src / "assets", task / "assets")

faulthandler.dump_traceback_later(240, repeat=False, exit=True)  # guard against a hung build
cfg = yaml.safe_load((Path(__file__).resolve().parents[1] / "config" / "init.yaml").read_text())
cfg.setdefault("startup", {})["headless"] = True

from guide_core.core.guide_simulator import GUIDESimulator  # noqa: E402

logging.basicConfig(level=logging.WARNING)
sim = GUIDESimulator(sim_id=0, namespace="Sim_0")
sim.init_runtime(config=cfg, logger=logging.getLogger("diag"))
sim.init_scene_manager()
rt = sim._runtime
sid = rt._cmd_register_scene(str(task))[0]
rt._cmd_start()
rt.update(30)

import omni.usd  # noqa: E402
from isaacsim.core.prims import RigidPrim, XFormPrim  # noqa: E402
from pxr import Usd  # noqa: E402

stage = omni.usd.get_context().get_stage()
red = f"/Scene_{sid}/blocks/red_block"
j2 = [p.GetPath().pathString for p in Usd.PrimRange(stage.GetPrimAtPath(f"/Scene_{sid}/fr3")) if p.GetName() == "fr3_joint2"]
print("[diag] joint2 prims:", j2, flush=True)
jp = stage.GetPrimAtPath(j2[0]) if j2 else None
print("[diag] joint2 type:", jp.GetTypeName() if jp else None, "| drive attrs:",
      [a.GetName() for a in jp.GetAttributes() if "drive" in a.GetName()][:8] if jp else None, flush=True)

import usdrt  # noqa: E402

rt_stage = usdrt.Usd.Stage.Attach(omni.usd.get_context().get_stage_id())
robot = rt._robots[f"/Scene_{sid}/fr3"]
robot.initialize()


def snapshot(tag):
    usd_local = np.asarray(XFormPrim(prim_paths_expr=red).get_local_poses()[0], dtype=float).ravel()[:3]
    physx_world = np.asarray(RigidPrim(prim_paths_expr=red).get_world_poses()[0], dtype=float).ravel()[:3]
    tgt = jp.GetAttribute("drive:angular:physics:targetPosition").Get() if jp else None
    fab = rt_stage.GetPrimAtPath(j2[0]).GetAttribute("drive:angular:physics:targetPosition")
    fab = fab.Get() if fab else "n/a"
    q2 = float(np.asarray(robot.get_joint_positions()).ravel()[1])
    print(f"[diag] {tag}: red USD-local {usd_local.round(3).tolist()} | PhysX-world {physx_world.round(3).tolist()} | joint2 target USD {tgt} / Fabric {fab} | joint2 q {np.degrees(q2):.1f} deg", flush=True)

snapshot("start")
rt._cmd_randomize_scene(scene_id=sid)
rt.update(5)
snapshot("after randomize")
rt._cmd_reset_scene(scene_id=sid)
rt.update(2)
snapshot("after reset +2")
rt.update(60)
snapshot("after reset +62")
# what the legacy set_joint does, for comparison of the drive-target attribute
import carb  # noqa: E402

carb.settings.get_settings().set("/app/useFabricSceneDelegate", False)  # diag only: force USD writes
rt._cmd_reset_scene(scene_id=sid)
rt.update(60)
snapshot("after reset with FSD off, +60")
rt.simulation_app.close()
