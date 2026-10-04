"""Diagnostic: which triggers exist, and who fires when one scene's event is sent."""
from __future__ import annotations

import logging
import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import yaml

src = Path(sys.argv[1]).resolve()
task = Path(tempfile.mkdtemp()) / src.name
task.mkdir()
shutil.copy(src / src.name / "scene.py", task / "scene.py")
shutil.copytree(src / "config", task / "config")
shutil.copytree(src / "assets", task / "assets")

cfg = yaml.safe_load((Path(__file__).resolve().parents[1] / "config" / "init.yaml").read_text())
cfg.setdefault("startup", {})["headless"] = True

from guide_core.core.guide_simulator import GUIDESimulator  # noqa: E402

logging.basicConfig(level=logging.WARNING)
sim = GUIDESimulator(sim_id=0, namespace="Sim_0")
sim.init_runtime(config=cfg, logger=logging.getLogger("diag"))
sim.init_scene_manager()
rt = sim._runtime
sids = [rt._cmd_register_scene(str(task))[0] for _ in range(2)]
rt._cmd_start()
rt.update(5)

import omni.graph.core as og  # noqa: E402
import omni.replicator.core as rep  # noqa: E402
from guide_core.types.randomization import replicator_guide as rg  # noqa: E402

scenes = {sid: sim._scene_manager._scenes[sid] for sid in sids}
print("[diag] events:", {sid: scenes[sid].replicator["randomize"]["event"] for sid in sids}, flush=True)
graph = rep.utils.get_graph()
for node in graph.get_nodes():
    if "OnCustomEvent" in node.get_type_name():
        print("[diag] trigger node", node.get_prim_path(), "eventName =", og.AttributeValueHelper(node.get_attribute("inputs:eventName")).get(), flush=True)

def samples(sid):
    st = scenes[sid].replicator["randomize"]
    return {k: np.asarray(rg._get(v, "outputs:samples")).ravel()[:3].round(3).tolist() for k, v in st["named"].items()}

if not rep.orchestrator.get_is_started():
    rep.orchestrator.run()
rep.orchestrator.step(rt_subframes=1, pause_timeline=False, wait_for_render=False)
print("[diag] before:", {sid: samples(sid) for sid in sids}, flush=True)
t = time.perf_counter()
rep.utils.send_og_event(scenes[sids[0]].replicator["randomize"]["event"])
rep.orchestrator.step(rt_subframes=1, pause_timeline=False, wait_for_render=False)
print(f"[diag] step took {time.perf_counter() - t:.2f}s", flush=True)
print("[diag] after scene0 event:", {sid: samples(sid) for sid in sids}, flush=True)
t = time.perf_counter()
rep.orchestrator.step(rt_subframes=1, pause_timeline=False, wait_for_render=False)
print(f"[diag] plain step took {time.perf_counter() - t:.2f}s", flush=True)
print("[diag] after plain step:", {sid: samples(sid) for sid in sids}, flush=True)
rt.simulation_app.close()
