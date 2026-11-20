"""T10: the peg_transfer scene brought up in a running GUIDE (Isaac 6 + ROS 2), checked by hand:

    ros2 launch guide_core bringup.launch.py > guide.log 2>&1     # GUIDE up, ROS + the worktree sourced
    ~/ros2_ws/.venv/bin/python guide_tasks/peg_transfer/test/isaac_peg_transfer_check.py --log guide.log --out <report.json>

It registers peg_transfer itself, then gates (plan T10):
1. three OSCBF hosts attached (psm1, psm2, ecm), none refused (the dynamics check is part of the attach);
2. the scene USD holds 4 articulation roots, and each robot's joint_states names its model's joints and no
   other arm's;
3. 10 s at home: each PSM tip within 0.1 mm of the model's FK of home (the scene's authored home agrees
   with the model), the ECM joints within 0.002 rad;
6. over the same 10 s every block on its peg moves at most 0.5 mm;
4. motion_target JOINTS and POSE (frame <psm>/psm_base_link) reach their goals on both PSMs; the ECM
   drops a POSE target and still follows JOINTS;
5. the jaws open to 1.0 rad and close; each PSM grasps a block from its peg at T11's grasp pose
   (peg_transfer.targets.grasp_pose: tool down across a wall, 2 cm/s), lifts it LIFT and holds it 5 s with at
   most 1 mm slip in the tip frame; CollisionRequest (T11's holding check, jaw vs block bounding boxes) reads
   False with the jaws closed 20 mm beside the block and True while holding;
7. Randomize moves the board within randomize.yaml's range and puts the six blocks on the giver's pegs;
   IsSuccess answers (False); 2 s of recording give init.yaml's feature list in order, two 480 x 720
   images and fps 20; one left frame is saved as PNG beside the report (the image check is by eye).
Writes --out JSON and exits 1 on a failed gate. Not collected by pytest (no test_ prefix).
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import yaml
from scipy.spatial.transform import Rotation as R

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[3] / "guide_core" / "test"))
from isaac_s3_scene_check import Io  # noqa: E402  (the same rclpy service/clock helper)

PKG = HERE.parents[1]
CONFIG = yaml.safe_load((PKG / "config" / "init.yaml").read_text())
USD = PKG / "assets" / "peg_transfer.usd"
S = "/Scene_0"
PSMS = ("psm1", "psm2")
GRASP = {"psm1": "red", "psm2": "purple"}  # purple sits on peg 5, the one nearest PSM2
SPEED = 0.02  # m/s, tool moves in the grasp
OPEN = None  # rad, jaw half opening for the grasp (--open, default 0.082); the asset clamps at 30 deg = 0.524


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--log", type=Path, required=True, help="GUIDE's log (stdout + stderr)")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--skip", default="", help="comma list of check numbers to skip, e.g. 5,7")
    parser.add_argument("--open", type=float, default=0.082, help="jaw half opening for the grasp, rad (0.082: 0.77 mm per jaw tip, straddles the 0.7 mm wall clear of the peg)")
    args = parser.parse_args()
    global OPEN
    OPEN = args.open
    return check(args.log, args.out.expanduser(), {int(s) for s in args.skip.split(",") if s})


def check(log: Path, out: Path, skip: set) -> int:
    import rclpy
    from guide_motion import description
    from guide_motion.model import Model
    from guide_msgs.msg import MotionTarget
    from guide_msgs.srv import CheckSuccess, Collision, FinalizeRecording, Randomize, RegisterScene, StartRecording, StopRecording
    from guide_msgs.srv import Pose as PoseSrv
    from rclpy.qos import QoSProfile
    from sensor_msgs.msg import JointState

    rclpy.init()
    io = Io(rclpy)
    node = io.node
    out.parent.mkdir(parents=True, exist_ok=True)
    report, failed = {}, []
    models = {r: Model.load(CONFIG["robots"][r]["profile"]) for r in CONFIG["robots"]}
    home = {r: np.array(description.ROBOTS[CONFIG["robots"][r]["profile"]]["home"]) for r in models}
    base = {r: f"{r}/{models[r].base_link}" for r in models}
    states, pubs, held = {}, {}, {}  # held: the target each robot is sent at 20 Hz

    def gate(name, ok, **data):
        report[name] = {"ok": bool(ok), **data}
        print(f"{'PASS' if ok else 'FAIL'} {name} {data}", flush=True)
        if not ok:
            failed.append(name)

    def log_text():
        return log.read_text(errors="replace")

    def pose(path, frame):
        reply = io.call(PoseSrv, "/Sim_0/PoseRequest", PoseSrv.Request(path=path, frame_id=frame))
        if not reply.success:
            raise RuntimeError(f"PoseRequest {path} in {frame}: {reply.message}")
        p, o = reply.pose.position, reply.pose.orientation
        return np.array([p.x, p.y, p.z]), R.from_quat([o.x, o.y, o.z, o.w])

    def tip(r):
        return pose(f"{S}{CONFIG['robots'][r]['path']}/psm_tool_tip_link", base[r])

    def q(r, names=None):
        msg = states[r]
        names = names or models[r].arm_joints
        return np.array([msg.position[list(msg.name).index(n)] for n in names])

    def target(r, joints=None, position=None, rotation=None, width=float("nan")):
        msg = MotionTarget(frame_id=base[r], gripper_width=float(width))
        msg.clean.x = msg.clean.y = msg.clean.z = float("nan")
        if joints is not None:
            msg.mode, msg.joints = MotionTarget.MODE_JOINTS, [float(x) for x in joints]
        else:
            msg.mode = MotionTarget.MODE_POSE
            msg.pose.position.x, msg.pose.position.y, msg.pose.position.z = (float(x) for x in position)
            x, y, z, w = rotation.as_quat()
            msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z, msg.pose.orientation.w = x, y, z, w
        held[r] = msg

    def spin(seconds):
        """Sim seconds, publishing every held target at 20 Hz of the sim clock (stamped with it)."""
        start, last = None, -1.0
        deadline = time.monotonic() + 120.0 + 30.0 * seconds
        while start is None or io.now < start + seconds:
            if time.monotonic() > deadline:
                raise RuntimeError(f"/clock did not advance {seconds} s")
            rclpy.spin_once(node, timeout_sec=0.01)
            if io.now is None:
                continue
            start = io.now if start is None else start
            if io.now - last >= 0.05:
                last = io.now
                for r, msg in held.items():
                    msg.stamp.sec, msg.stamp.nanosec = int(io.now), int((io.now % 1) * 1e9)
                    pubs[r].publish(msg)

    def move(r, position, rotation, width=float("nan")):
        """Straight line from the current tip pose at SPEED (rotation slerped along), then 1 s to settle."""
        p0, r0 = tip(r)
        n = max(1, int(np.linalg.norm(position - p0) / SPEED / 0.05))
        n = max(n, int(np.degrees((rotation * r0.inv()).magnitude()) / 20.0 / 0.05))  # 20 deg/s
        from scipy.spatial.transform import Slerp
        slerp = Slerp([0, 1], R.concatenate([r0, rotation]))
        for k in range(1, n + 1):
            s = k / n
            target(r, position=p0 + s * (position - p0), rotation=slerp(s), width=width)
            spin(0.05)
        spin(1.0)

    def holding(r):
        jaw = f"{S}{CONFIG['robots'][r]['path']}/psm_tool_gripper1_link"  # solve_task's jaw_prim
        return io.call(Collision, "/Sim_0/CollisionRequest", Collision.Request(prim1=jaw, prim2=f"{S}/PegTransfer/Blocks/{GRASP[r]}")).collision

    def grasp(r):
        """Air miss beside the block, then T11's grasp, lift and 5 s hold; the block put back after."""
        from guide_core.types.geometry import Pose
        from peg_transfer.targets import APPROACH, LIFT, grasp_pose
        block = f"{S}/PegTransfer/Blocks/{GRASP[r]}"
        bp, brot = pose(block, base[r])
        up = pose(f"{S}/PegTransfer", base[r])[1].as_matrix()[:, 2]
        g = grasp_pose(Pose.from_numpy(bp, brot), up)
        gp, gr = g.position.to_numpy(), R.from_matrix(g.orientation.to_numpy())
        side = gp + 0.02 * gr.as_matrix()[:, 0] + 0.01 * up  # 20 mm along the closing axis, clear of the block
        move(r, side + APPROACH * up, gr, width=0.0)
        move(r, side, gr, width=0.0)
        miss, miss_jaw = holding(r), float(q(r, ["psm_tool_gripper2_joint"])[0])
        move(r, gp + APPROACH * up, gr, width=2 * OPEN)
        move(r, gp, gr, width=2 * OPEN)
        reached = np.linalg.norm(tip(r)[0] - gp)
        target(r, position=gp, rotation=gr, width=0.0)
        spin(1.5)
        jaw = float(q(r, ["psm_tool_gripper2_joint"])[0])
        held_now = holding(r)
        b0 = pose(block, base[r])[0]
        move(r, gp + LIFT * up, gr, width=0.0)
        def local():
            p, rot = tip(r)
            return rot.inv().apply(pose(block, base[r])[0] - p)
        l0, lifted = local(), float((pose(block, base[r])[0] - b0) @ up)
        spin(5.0)
        l1 = local()
        slip = float(np.linalg.norm(l1 - l0))
        held_after = holding(r)
        bp1, brot1 = pose(block, base[r])
        jaws = {}  # each jaw link's pose in the block frame at the end of the hold, for a penetration check offline
        for j in ("gripper1", "gripper2"):
            jp, jrot = pose(f"{S}{CONFIG['robots'][r]['path']}/psm_tool_{j}_link", base[r])
            jaws[j] = [(brot1.inv().apply(jp - bp1)).tolist(), (brot1.inv() * jrot).as_quat().tolist()]
        gate(f"5_{r}_air_miss_not_holding", not miss, holding=miss, jaw_rad=miss_jaw)
        gate(f"5_{r}_grasp_{GRASP[r]}", held_now and held_after and lifted >= LIFT - 0.003 and slip <= 1e-3,
             open_rad=OPEN, jaws_in_block=jaws, holding_closed=held_now, holding_after_hold=held_after, reach_mm=1e3 * reached,
             jaw_closed_rad=jaw, lift_mm=1e3 * lifted, slip_mm=1e3 * slip, block_in_tip_mm=(1e3 * l1).round(2).tolist(),
             grasp_offset_mm=(1e3 * gr.inv().apply(bp - gp)).round(2).tolist())
        move(r, gp + 0.001 * up, gr, width=0.0)  # back down over its peg, then let go
        target(r, position=gp + 0.001 * up, rotation=gr, width=2 * OPEN)
        spin(1.0)
        move(r, gp + APPROACH * up, gr, width=2 * OPEN)
        target(r, joints=home[r], width=2 * np.radians(12))
        spin(3.0)
        report[f"5_{r}_grasp_{GRASP[r]}"]["block_back_mm"] = float(1e3 * np.linalg.norm(pose(block, base[r])[0] - bp))

    try:
        # ---- registration (1)
        reply = io.call(RegisterScene, "/Sim_0/Register", RegisterScene.Request(path="peg_transfer"))
        gate("1_register", reply.success and reply.id == 0 and not reply.message, id=int(reply.id), message=reply.message)
        for r in models:
            topic = f"/Sim_0{S}/{r}/motion_target"
            pubs[r] = node.create_publisher(MotionTarget, topic, QoSProfile(depth=1))
        deadline = time.monotonic() + 60
        while len(states) < len(models) and time.monotonic() < deadline:
            for r in models:
                if r not in states:
                    found = [n for n in (x for x, _ in node.get_topic_names_and_types()) if n.endswith(f"/{r}/joint_states")]
                    if found and not hasattr(node, f"_js_{r}"):
                        setattr(node, f"_js_{r}", node.create_subscription(
                            JointState, found[0], lambda m, r=r: states.__setitem__(r, m), 10))
            rclpy.spin_once(node, timeout_sec=0.1)
        names = sorted(n for n, _ in node.get_topic_names_and_types())
        report["topics"] = [n for n in names if n.startswith(S)]
        spin(1.0)
        text = log_text()
        attached, refused = text.count("[oscbf] attached:"), text.count("refused:")
        gate("1_three_hosts_attached", attached == 3 and refused == 0, attached=attached, refused=refused,
             dynamics=[line.split("[oscbf]")[-1].strip() for line in text.splitlines() if "mass properties differ" in line or "joins the articulation" in line][:8])

        # ---- articulations and DOF names (2)
        from pxr import Usd, UsdPhysics
        stage = Usd.Stage.Open(str(USD))
        roots = sorted(str(p.GetPath()) for p in stage.Traverse() if p.HasAPI(UsdPhysics.ArticulationRootAPI))
        gate("2_four_articulation_roots", len(roots) == 4, roots=roots)
        for r, m in models.items():
            names = list(states[r].name) if r in states else []
            mine = set(m.arm_joints) | set(getattr(m, "gripper_joints", []) or [])
            other = "ecm_" if r.startswith("psm") else "psm_"
            gate(f"2_{r}_dofs", mine <= set(names) and not any(n.startswith(other) for n in names), names=names)

        # ---- hold at home (3) and blocks on their pegs (6)
        for r in models:
            target(r, joints=home[r], width=2 * np.radians(12) if r in PSMS else float("nan"))
        spin(2.0)
        blocks = [f"{S}/PegTransfer/Blocks/{c}" for c in ("red", "green", "blue", "yellow", "orange", "purple")]
        b0 = np.array([pose(b, "Scene_0")[0] for b in blocks])
        errors = {r: [] for r in models}
        fk = {r: models[r].ee_pose(home[r]) for r in PSMS}
        t_end = io.now + 10.0
        while io.now < t_end:
            spin(0.5)
            for r in PSMS:
                p, rot = tip(r)
                errors[r].append((np.linalg.norm(p - fk[r][0]), np.degrees((rot * R.from_matrix(np.asarray(fk[r][1])).inv()).magnitude())))
            errors["ecm"].append((np.max(np.abs(q("ecm") - home["ecm"])), 0.0))
        for r in PSMS:
            e = np.array(errors[r])
            gate(f"3_{r}_hold", e[:, 0].max() <= 1e-4, max_mm=1e3 * e[:, 0].max(), max_deg=e[:, 1].max(),
                 q=q(r).round(5).tolist())
        e = np.array(errors["ecm"])
        gate("3_ecm_hold", e[:, 0].max() <= 0.002, max_rad=e[:, 0].max(), q=q("ecm").round(5).tolist())
        drift = np.linalg.norm(np.array([pose(b, "Scene_0")[0] for b in blocks]) - b0, axis=1)
        gate("6_blocks_still_10s", drift.max() <= 5e-4, max_mm=1e3 * drift.max(), per_block_mm=(1e3 * drift).round(3).tolist())

        # ---- motion targets (4)
        if 4 not in skip:
            for r in PSMS:
                goal = home[r] + [0.05, -0.05, 0.01, 0.2, 0.2, 0.1]
                target(r, joints=goal)
                spin(3.0)
                err = np.abs(q(r) - goal)
                gate(f"4_{r}_joints", err.max() <= 0.005, max_err=float(err.max()), err=err.round(5).tolist())
                target(r, joints=home[r])
                spin(2.0)
                p0, r0 = fk[r][0], R.from_matrix(np.asarray(fk[r][1]))
                goal_p, goal_r = p0 + [0.01, 0.0, -0.01], r0 * R.from_euler("x", 20, degrees=True)
                move(r, goal_p, goal_r)
                spin(2.0)
                p, rot = tip(r)
                dp, da = np.linalg.norm(p - goal_p), np.degrees((rot * goal_r.inv()).magnitude())
                gate(f"4_{r}_pose", dp <= 5e-4 and da <= 1.0, err_mm=1e3 * dp, err_deg=da)
                target(r, joints=home[r])
                spin(2.0)
            before = log_text().count("refused:")
            target("ecm", position=[0.0, 0.0, -0.1], rotation=R.identity())
            spin(1.0)
            dropped = log_text().count("refused:") == before
            goal = home["ecm"] + [0.05, 0.05, 0.005, 0.05]
            target("ecm", joints=goal)
            spin(3.0)
            err = np.abs(q("ecm") - goal)
            gate("4_ecm_pose_dropped_joints_followed", dropped and err.max() <= 0.005, pose_dropped=dropped,
                 max_err=float(err.max()))
            target("ecm", joints=home["ecm"])
            spin(2.0)

        # ---- jaws and grasp (5)
        if 5 not in skip:
            for r in PSMS:
                target(r, joints=home[r], width=2.0)
                spin(1.5)
                opened = float(q(r, ["psm_tool_gripper2_joint"])[0])
                target(r, joints=home[r], width=0.0)
                spin(1.5)
                closed = float(q(r, ["psm_tool_gripper2_joint"])[0])
                gate(f"5_{r}_jaws", opened >= 0.5 and abs(closed) <= 0.05, open_rad=opened, closed_rad=closed)
            for r in PSMS:
                grasp(r)

        # ---- randomize, success, recording (7)
        if 7 not in skip:
            for r in models:
                target(r, joints=home[r], width=2 * np.radians(12) if r in PSMS else float("nan"))
            spin(1.0)
            ok = io.call(Randomize, "/Sim_0/Randomize", Randomize.Request(id=0)).success
            spin(1.0)
            p, rot = pose(f"{S}/PegTransfer/Board", "Scene_0")
            yaw = rot.as_euler("xyz", degrees=True)[2]
            moved = abs(p[0]) <= 0.0101 and abs(p[1] - 0.72) <= 0.0101 and 84.9 <= yaw <= 95.1 and abs(yaw - 90) > 1e-3
            gate("7_randomize_board", ok and moved, success=ok, board=p.round(4).tolist(), yaw_deg=yaw)
            spin(1.0)
            # each block on a peg: its centre within 1 mm of a peg's axis
            pegs = np.array([pose(f"{S}/PegTransfer/Board/peg_{i:02d}", "Scene_0")[0] for i in range(12)])
            off = [float(np.min(np.linalg.norm(pegs[:, :2] - pose(b, "Scene_0")[0][:2], axis=1))) for b in blocks]
            gate("7_blocks_on_pegs", max(off) <= 1e-3, off_mm=(1e3 * np.array(off)).round(3).tolist())
            done = io.call(CheckSuccess, "/Sim_0/IsSuccess", CheckSuccess.Request(id=0)).success
            errors_after = [line for line in log_text().splitlines()[-400:] if "rror" in line and "is_success" in line.lower()]
            gate("7_is_success_false", done is False and not errors_after, got=done, errors=errors_after[:3])
            root = out.parent / "dataset"
            started = io.call(StartRecording, "/Sim_0/start_recording", StartRecording.Request(id=0, path=str(root)))
            spin(2.0)
            stopped = io.call(StopRecording, "/Sim_0/stop_recording", StopRecording.Request(id=0, save_episode=True))
            final = io.call(FinalizeRecording, "/Sim_0/finalize_recording", FinalizeRecording.Request(id=0), timeout=600)
            infos = sorted(root.rglob("meta/info.json"), key=lambda p: p.stat().st_mtime)
            info = json.loads(infos[-1].read_text()) if infos else {}
            feats = info.get("features", {})
            want = [next(iter(o)) for o in CONFIG["dataset"]["observations"]]
            got = (feats.get("observation.state", {}).get("names") or [None])
            got = got if isinstance(got, list) and got and isinstance(got[0], str) else got
            want_a = [next(iter(o)) for o in CONFIG["dataset"]["action"]]
            got_a = feats.get("action", {}).get("names")
            images = {k: v.get("shape") for k, v in feats.items() if k.startswith("observation.images")}
            gate("7_recording", started.success and stopped.success and final.success and got == want and got_a == want_a
                 and sorted(map(tuple, images.values())) == [(480, 720, 3)] * 2 and info.get("fps") == 20,
                 messages=[started.message, stopped.message, final.message], dataset=str(infos[-1].parent.parent) if infos else None,
                 state_names_match=got == want, action_names_match=got_a == want_a, images=images, fps=info.get("fps"),
                 frames=info.get("total_frames"))
            if infos:
                frame(infos[-1].parent.parent, out.parent / "left.png")
    except Exception as e:  # noqa: BLE001 -- the report says where it stopped
        import traceback
        gate("ran_to_the_end", False, error=f"{type(e).__name__}: {e}", trace=traceback.format_exc()[-1500:])
    finally:
        out.write_text(json.dumps({"failed": failed, "gates": report}, indent=1, default=str))
        rclpy.shutdown()
    return 1 if failed else 0


def frame(dataset: Path, png: Path) -> None:
    """The first frame of the left camera's video, saved for the by-eye image check."""
    import av
    from PIL import Image
    video = sorted(dataset.rglob("*left*/*.mp4")) or sorted(dataset.rglob("*.mp4"))
    with av.open(str(video[0])) as container:
        Image.fromarray(next(container.decode(video=0)).to_ndarray(format="rgb24")).save(png)


if __name__ == "__main__":
    sys.exit(main())
