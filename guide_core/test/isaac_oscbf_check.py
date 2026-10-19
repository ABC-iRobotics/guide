"""Headless Isaac checks for the OSCBF host (S2b spec 7). Run by hand, one Isaac at a time:

    OMNI_KIT_ACCEPT_EULA=YES env -u CUDA_DEVICE_ORDER ~/ros2_ws/.venv/bin/python \\
        ~/ros2_ws/src/guide/guide_core/test/isaac_oscbf_check.py --robot fr3 --mode torque --check hold --out /tmp/x.json

hold:   no targets for 5 s, so the host's stale hold: mean tool error over the last 4 s <= 0.1 mm.
square: 2 s hold, then a 10 cm square at 10 cm/s as a 20 Hz zero-order hold: mean path deviation
        on lap 2 <= 3 mm (torque) / 2 mm (velocity); no NaN; |command| <= the URDF limits. Saves q.
home:   start 1 rad off home on joints 1, 2 and 4, then host.home() (Reset under oscbf, Ruling 2):
        peak |qd| per joint against the URDF velocity limits (controller, B3b); final error <= 0.01 rad.
blockbin: block_bin.usd's FR3 (its wrist camera included), registered before play as GUIDE does;
        then the restart check's 2 s stale hold and joint target, with the same gates.
restart: 1 s, then World.stop() and play() as GUIDE's second RegisterScene does (the joints revert
        to the asset's authored state), then a 2 s stale hold (mean <= 0.1 mm over its last second),
        3 s of a JOINTS target with joint 1 at home + 0.3 rad and the hand open, then GUIDE's Reset
        (reset.yaml closes the fingers, the host homes: 2 s) and 2 s of a JOINTS target with joint 1 at
        home - 0.3 rad and the hand open. Each joint error <= 0.01 rad, the finger within 2 mm.
rtf:    frame time over the square; --mode baseline records an awake arm without a controller,
        a controller run given --baseline must add <= 2.5 ms (torque) / 1.2 ms (velocity).
Writes --out JSON, exits 1 when a threshold fails. Not collected by pytest (no test_ prefix).
"""

import argparse
import json
import time
from pathlib import Path
from types import SimpleNamespace

parser = argparse.ArgumentParser()
parser.add_argument("--robot", choices=["fr3", "panda"], required=True)
parser.add_argument("--mode", choices=["torque", "velocity", "baseline"], required=True)
parser.add_argument("--check", choices=["hold", "square", "rtf", "home", "restart", "blockbin"], required=True)
parser.add_argument("--baseline", help="rtf: the baseline run's JSON")
parser.add_argument("--out", required=True)
args = parser.parse_args()
assert args.mode != "baseline" or args.check == "rtf", "the baseline exists for the rtf check only"
assert args.check != "blockbin" or args.robot == "fr3", "block_bin's robot is the FR3"

from isaacsim import SimulationApp  # noqa: E402

# GUIDE's startup settings (config/init.yaml), headless, rendering on the A4000.
app = SimulationApp({"headless": True, "renderer": "RaytracedLighting", "anti_aliasing": 0,
                     "width": 640, "height": 360, "active_gpu": 1, "multi_gpu": False})

import numpy as np  # noqa: E402
import omni.replicator.core as rep  # noqa: E402
from isaacsim.core.api import World  # noqa: E402  (deprecated, but it is what GUIDE steps with)
from isaacsim.core.experimental.prims import Articulation  # noqa: E402
from isaacsim.core.experimental.utils import stage as stage_utils  # noqa: E402
from isaacsim.core.simulation_manager import SimulationEvent, SimulationManager  # noqa: E402
from pxr import UsdLux  # noqa: E402

from guide_core.control.oscbf_host import OscbfHost  # noqa: E402
from guide_motion import description  # noqa: E402
from guide_motion.model import Model  # noqa: E402
from guide_motion.targets import JOINTS, Target  # noqa: E402

PHYSICS_HZ, FRAME_HZ, TARGET_HZ = 120, 60, 20.0
HOLD_S, LAP_S, SETTLE_FRAMES = 2.0, 4.0, 30
SQUARE = np.array([[0, 0, 0], [0, 0.1, 0], [0.1, 0.1, 0], [0.1, 0, 0], [0, 0, 0]], float)  # +y, +x, -y, -x
LIMIT = {"hold": 0.1, "square": {"torque": 3.0, "velocity": 2.0}, "rtf": {"torque": 2.5, "velocity": 1.2}}
FAR = np.array([1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0])  # home check: start offset from home, rad
MOVED = np.array([0.3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])  # restart check: the joint target's offset from home, rad
RESET_S, MOVE2_S = 5.0, 7.0  # restart check: Reset, then the second joint target


def square_offset(t):
    """The start for HOLD_S, then the square at 10 cm/s, sampled at 20 Hz and held."""
    if t < HOLD_S:
        return np.zeros(3)
    k = int((t - HOLD_S) * TARGET_HZ)
    s = (k / TARGET_HZ) % LAP_S / LAP_S * 4
    i = int(s)
    return SQUARE[i] + (s - i) * (SQUARE[i + 1] - SQUARE[i])


world = World(stage_units_in_meters=1.0, physics_dt=1.0 / PHYSICS_HZ, rendering_dt=1.0 / FRAME_HZ)
world.get_physics_context().enable_gpu_dynamics(False)
world.scene.add_default_ground_plane()
UsdLux.DistantLight.Define(world.stage, "/World/light").CreateIntensityAttr(3000)
BLOCKBIN = args.check == "blockbin"
ROOT = "/Scene_0/fr3" if BLOCKBIN else "/World/robot"
if BLOCKBIN:  # GUIDE's scene, its FR3 as authored (the host applies the overrides at registration)
    stage_utils.add_reference_to_stage(str(Path(__file__).resolve().parents[2] / "guide_tasks/block_bin/assets/block_bin.usd"), "/Scene_0")
else:
    source = str(description.ROBOTS_DIR / args.robot / f"{args.robot}.usda")  # Ruling 10: corrected masses
    stage_utils.add_reference_to_stage(source, "/World/robot")
camera = rep.create.camera(position=(1.6, 1.2, 1.1), look_at=(0.3, 0.0, 0.3))
annotator = rep.AnnotatorRegistry.get_annotator("rgb")
annotator.attach([rep.create.render_product(camera, (640, 480))])  # one camera: the measurement's light load

model = Model.load(args.robot)
arm = Articulation(ROOT)
arm.set_sleep_thresholds(0.0)  # awake in every run: a sleeping arm's physics is free and fakes the baseline


def register():
    """GUIDE's RegisterScene: the host compiles (the persistent cache makes reruns quicker)."""
    say = lambda message: print(message, flush=True)  # noqa: E731  the [oscbf] lines Step 2 greps
    return OscbfHost(ROOT, robot=args.robot, mode=args.mode, q_rest=model.home,  # the host's default rate, as GUIDE runs it
                     logger=SimpleNamespace(info=say, warning=say, error=say))


host = register() if BLOCKBIN else None  # before play, as in GUIDE: the arm starts at block_bin's authored joint state
world.reset()
dofs = list(arm.dof_names)
ARM = [dofs.index(j) for j in model.arm_joints]
FINGER, FINGERS = dofs.index(model.gripper_joint), [i for i, name in enumerate(dofs) if "finger" in name]
if not BLOCKBIN:
    home = np.zeros(arm.num_dofs)
    home[ARM] = model.home + (FAR if args.check == "home" else 0.0)
    arm.set_dof_positions(home[None])
    arm.set_dof_position_targets(home[None])
    for _ in range(SETTLE_FRAMES):  # position-driven settle at home
        world.step(render=True)
        annotator.get_data()
    if args.mode != "baseline":
        host = register()  # after the settle, with the timeline playing: <robot>.usda already carries the overrides
        if args.check == "home":
            host.home()  # what GUIDE's Reset does under controller: oscbf
p0, R0 = model.ee_pose(arm.get_dof_positions().numpy()[0][ARM].astype(float))

log = []
if args.check != "rtf":  # a per-substep read would bill the RTF run

    def on_post_step(dt, context):
        q = arm.get_dof_positions().numpy()[0][ARM].astype(float)
        qd = arm.get_dof_velocities().numpy()[0][ARM].astype(float)
        log.append((q, np.full(len(ARM), np.nan) if host._u is None else np.array(host._u, float), qd))

    SimulationManager.register_callback(on_post_step, SimulationEvent.PHYSICS_POST_STEP)

if args.check == "restart":  # GUIDE's second RegisterScene: stop and play with the host registered
    for _ in range(FRAME_HZ):
        world.step(render=True)
    world.stop()
    world.play()
    log.clear()
frames = int(9.0 * FRAME_HZ) if args.check in ("restart", "blockbin") else int(5.0 * FRAME_HZ) if args.check in ("hold", "home") else int((HOLD_S + 2 * LAP_S) * FRAME_HZ)
frame_ms, last_k = [], -1
for i in range(frames):
    t = i / FRAME_HZ
    k = int(t * TARGET_HZ)
    if host is not None and args.check in ("square", "rtf") and k != last_k:
        host.holder.put(Target(position=p0 + square_offset(t), rotation=R0))  # GUIDE's ROS thread does this
        last_k = k
    if args.check in ("restart", "blockbin"):
        if t >= RESET_S and t - 1.0 / FRAME_HZ < RESET_S:  # GUIDE's Reset: reset.yaml closes the fingers, then home()
            arm.set_dof_position_targets(np.zeros((1, len(FINGERS))), dof_indices=FINGERS)
            host.home()
        goal = model.home + MOVED if HOLD_S <= t < RESET_S else model.home - MOVED if t >= MOVE2_S else None
        if goal is not None and k != last_k:
            host.holder.put(Target(mode=JOINTS, joints=goal, gripper_width=0.08))
            last_k = k
    start = time.perf_counter()
    world.step(render=True)
    annotator.get_data()
    frame_ms.append(1e3 * (time.perf_counter() - start))
finger = float(arm.get_dof_positions().numpy()[0][FINGER])

result = {"robot": args.robot, "mode": args.mode, "check": args.check, "frames": frames}
ok = True
if args.check == "rtf":
    ft = np.array(frame_ms[30:])
    result.update(frame_ms_mean=float(ft[ft < 40.0].mean()), spikes_over_40ms=int((ft >= 40.0).sum()))
    if host is not None:  # a refused robot is never commanded: its frame time would pass for free
        result.update(infeasible=host.diagnostics["infeasible"], refused=host.diagnostics["refused"])
        ok = not result["refused"]
    if args.baseline:
        result["added_ms"] = result["frame_ms_mean"] - json.loads(Path(args.baseline).read_text())["frame_ms_mean"]
        ok = ok and result["added_ms"] <= LIMIT["rtf"][args.mode]
else:
    q, u, qd = (np.array([e[i] for e in log]) for i in range(3))
    p = np.array([model.ee_pose(x)[0] for x in q])
    t = np.arange(len(q)) / PHYSICS_HZ
    limit = model.max_effort if args.mode == "torque" else model.max_velocity
    result.update(nan=bool(np.isnan(q).any() or np.isnan(u[1:]).any()), u_over_limit=float(np.nanmax(np.abs(u) / limit)),
                  infeasible=host.diagnostics["infeasible"], refused=host.diagnostics["refused"])
    ok = not result["nan"] and result["u_over_limit"] <= 1.0 + 1e-6 and not result["refused"]
    if args.check == "hold":
        error = 1e3 * np.linalg.norm(p - host._hold.position, axis=1)[t >= 1.0]
        result.update(hold_mm_mean=float(error.mean()), hold_mm_max=float(error.max()), stale=host.diagnostics["stale"])
        ok = ok and result["hold_mm_mean"] <= LIMIT["hold"] and result["stale"]
    elif args.check == "home":
        peak = np.abs(qd).max(axis=0)
        result.update(qd_peak=peak.round(3).tolist(), qd_limit=model.max_velocity.round(3).tolist(),
                      qd_over_limit=float((peak / model.max_velocity).max()),
                      final_error_rad=float(np.abs(q[-1] - model.home).max()), stale=host.diagnostics["stale"])
        ok = ok and result["qd_over_limit"] <= 1.0 + 1e-6 and result["final_error_rad"] <= 1e-2 and not result["stale"]  # 1e-6: float32 PhysX cap (2.62f vs 2.61999977), as u_over_limit
    elif args.check in ("restart", "blockbin"):
        error = 1e3 * np.linalg.norm(p - p[0], axis=1)[(t >= 1.0) & (t < HOLD_S)]  # p[0]: the pose held after the restart
        before = lambda s: q[t < s][-1]  # noqa: E731  the last sample of a phase
        result.update(restart_q=q[0].round(4).tolist(), hold_mm_mean=float(error.mean()), hold_mm_max=float(error.max()),
                      moved_error_rad=float(np.abs(before(RESET_S) - (model.home + MOVED)).max()),
                      home_error_rad=float(np.abs(before(MOVE2_S) - model.home).max()),
                      moved2_error_rad=float(np.abs(q[-1] - (model.home - MOVED)).max()),
                      finger_error_m=abs(finger - 0.04), stale=host.diagnostics["stale"])
        ok = (ok and result["hold_mm_mean"] <= LIMIT["hold"] and not result["stale"] and result["finger_error_m"] <= 2e-3
              and max(result["moved_error_rad"], result["home_error_rad"], result["moved2_error_rad"]) <= 1e-2)
    else:
        path = np.array([p0 + square_offset(HOLD_S + x) for x in np.linspace(0, LAP_S, 4000, endpoint=False)])
        lap2 = t >= HOLD_S + LAP_S
        deviation = 1e3 * np.linalg.norm(p[lap2, None] - path[None], axis=2).min(axis=1)
        result.update(path_mm_mean=float(deviation.mean()), path_mm_peak=float(deviation.max()))
        ok = ok and result["path_mm_mean"] <= LIMIT["square"][args.mode]
        np.save(Path(args.out).with_suffix(".q.npy"), q)
result["pass"] = bool(ok)
Path(args.out).write_text(json.dumps(result, indent=1))
print("RESULT", json.dumps(result), flush=True)
app.close(exit_code=0 if ok else 1)  # Kit's fast shutdown ends the process here: sys.exit would never run
