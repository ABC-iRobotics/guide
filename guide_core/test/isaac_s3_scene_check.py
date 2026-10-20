"""S3 v2 Plan B: GUIDE core scene features, checked in a running GUIDE (Isaac 6.0.1 + ROS 2). By hand:

    ~/ros2_ws/.venv/bin/python guide_core/test/isaac_s3_scene_check.py make <dir>
    ros2 launch guide_core bringup.launch.py        # GUIDE up, ROS sourced
    ~/ros2_ws/.venv/bin/python guide_core/test/isaac_s3_scene_check.py check <dir> --out <report.json>

make writes two synthetic tasks from S1's converted LIBERO assets (~/libero_assets/usd), one per robot
profile: <dir>/s3_fr3 and <dir>/s3_panda. Each is a RecipeScene whose config/scene.yaml creates a wooden
cabinet, a flat stove and a plate on a stage USD (table, lights, a 512 x 512 agentview camera, a mark and
a region for the plate), whose robot comes from its profile at LIBERO's base pose, and whose
randomize.yaml places everything per episode (modify.pose, guide.fixture_pose, guide.joint_state).
config/arrangements/ holds a, b, c and bad.

check registers both (the second registration stops and replays the timeline) and gates, per scene and
arrangement, after 0.25 s of sim time (spec 4 phase 6):
- the plate within 5 mm / 3 deg;
- each fixture within 1 mm / 0.1 deg;
- the drawer within 1 mm and the knob within 0.01 rad of the arrangement;
- IsSuccess false / true / false for a / b / c;
- the robot base at its mount (1 mm / 0.1 deg);
- a 512 x 512 frame from each camera, saved as PNG beside the report.
It also gates that scene 0's cabinet stays at arrangement a's pose across scene 1's registration, that a
free draw succeeds, and that 'bad' (an unknown distribution) and 'missing' (no file) fail. It writes
--out JSON and exits 1 on any failed gate. Not collected by pytest (no test_ prefix).
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import yaml
from scipy.spatial.transform import Rotation as R

USD = Path.home() / "libero_assets" / "usd"
CAB = "/objects/wooden_cabinet_1_Xform/wooden_cabinet_1/Geometry/Body"
STOVE = "/objects/flat_stove_1_Xform/flat_stove_1/Geometry/Body"
PLATE = "/objects/plate_1_Xform/plate_1/Geometry/Body"
DRAWER = f"{CAB}/object/base/cabinet_top/top_level"
KNOB = f"{STOVE}/object/base/button/button"
MOUNT = [-0.66, 0.0, 0.012]  # LIBERO's robot0_base in the table-top frame
A = {"plate_position": [0.05, 0.20, 0.0033], "plate_rotation": [0.0, 0.0, 90.0],
     "cabinet_position": [0.02, -0.28, 0.005], "cabinet_rotation": [0.0, 0.0, 152.0],
     "stove_position": [-0.41, -0.14, 0.021], "stove_rotation": [0.0, 0.0, 0.0],
     "cabinet_top_state": 0.0, "stove_knob_state": 0.0}
B = {"plate_position": [0.055, 0.205, 0.0033], "plate_rotation": [0.0, 0.0, 95.0],
     "cabinet_position": [0.04, -0.26, 0.005], "cabinet_rotation": [0.0, 0.0, 155.0],
     "stove_position": [-0.40, -0.15, 0.021], "stove_rotation": [0.0, 0.0, 2.0],
     "cabinet_top_state": -0.12, "stove_knob_state": 1.0}
C = {**B, "plate_position": [-0.05, 0.12, 0.0033], "plate_rotation": [0.0, 0.0, 90.0]}
ARRANGEMENTS = {"a": A, "b": B, "c": C, "bad": {"no_such_distribution": 1.0}}
SUCCEEDS = {"a": False, "b": True, "c": False}  # a: drawer shut, knob off; c: the plate off its mark

INIT = """usd_path: '/assets/stage.usda'
publish_camera_topics: true
robots:
  franka:
    profile: @ROBOT@
    mount: {position: [-0.66, 0.0, 0.012], orientation: {euler: [z, 0.0]}}
cameras:
  agentview: {path: '/agentview', width: 512, height: 512, topic: '/cam_agentview'}
  wrist: {path: '/@ROBOT@/@ROBOT@_hand/wrist_camera', width: 512, height: 512, topic: '/cam_wrist'}
limits:
  - [-1.0, 1.0]
  - [-1.0, 1.0]
  - [-1.0, 1.0]
origin: [0.0, 0.0, 0.0]
"""
SCENE = """# S1's converted fixtures and the plate (its S1 fixes in ../assets/plate.usda).
cabinet:
  create.from_usd:
    usd: @CAB_USD@
    name: wooden_cabinet_1
    parent: /objects
stove:
  create.from_usd:
    usd: @STOVE_USD@
    name: flat_stove_1
    parent: /objects
plate:
  create.from_usd:
    usd: {property: file_path, value: ../assets/plate.usda}
    name: plate_1
    parent: /objects
"""
RANDOMIZE = """place:
  randomizer.register:
    place:
      plate:
        get.prims: {path_pattern: '@PLATE@$'}
      with.plate:
        modify.pose:
          write_to_usd: true
          position:
            distribution.uniform: {lower: [-0.10, 0.10, 0.0033], upper: [0.10, 0.25, 0.0033], name: plate_position}
          rotation:
            distribution.uniform: {lower: [0, 0, 0], upper: [0, 0, 180], name: plate_rotation}
      still:
        get.prims: {path_pattern: '@PLATE@(/object)?$'}
      with.still:
        modify.attribute: {name: 'physics:velocity', attribute_type: vector3f, value: [0, 0, 0]}

episode:
  trigger.on_custom_event: {event_name: guide_randomize}

with_episode:
  with.episode:
    randomizer.place: null
    cabinet:
      guide.fixture_pose:
        path_pattern: '/objects/wooden_cabinet_1_Xform/wooden_cabinet_1$'
        position:
          distribution.uniform: {lower: [0.0, -0.30, 0.005], upper: [0.04, -0.26, 0.005], name: cabinet_position}
        rotation:
          distribution.uniform: {lower: [0, 0, 150], upper: [0, 0, 156], name: cabinet_rotation}
    stove:
      guide.fixture_pose:
        path_pattern: '/objects/flat_stove_1_Xform/flat_stove_1$'
        position:
          distribution.uniform: {lower: [-0.42, -0.15, 0.021], upper: [-0.40, -0.13, 0.021], name: stove_position}
        rotation:
          distribution.uniform: {lower: [0, 0, 0], upper: [0, 0, 2], name: stove_rotation}

joints:
  trigger.on_custom_event: {event_name: guide_randomize_joints}

with_joints:
  with.joints:
    drawer:
      guide.joint_state:
        path_pattern: '@DRAWER@$'
        distribution:
          distribution.uniform: {lower: -0.16, upper: 0.0, name: cabinet_top_state}
    knob:
      guide.joint_state:
        path_pattern: '@KNOB@$'
        distribution:
          distribution.uniform: {lower: 0.0, upper: 2.0, name: stove_knob_state}
"""
SUCCESS = """instructions:
  - {cmd: 'on', kwargs: {a: '@PLATE@', b: '/marks/plate_home'}}   # quoted: YAML 1.1 reads a bare on as true
  - {cmd: in_region, kwargs: {obj: '@PLATE@', region: '/marks/plate_region'}}
  - {cmd: joint_below, kwargs: {joint: '@DRAWER@', value: -0.10}}
  - {cmd: joint_above, kwargs: {joint: '@KNOB@', value: 0.5}}
"""
TASK = """language:
  task: put the plate on its mark, open the top drawer and turn on the stove
  subtasks: [open the top drawer, turn on the stove]
target: '@PLATE@'
goal:
  - {'on': ['@PLATE@', /marks/plate_home]}
  - {open: '@DRAWER@'}
  - {turn_on: '@KNOB@'}
"""
STAGE = """#usda 1.0
(
    defaultPrim = "World"
    metersPerUnit = 1
    upAxis = "Z"
)

def Xform "World"
{
    def Cube "table" (
        prepend apiSchemas = ["PhysicsCollisionAPI"]
    )
    {
        double size = 1
        float3[] extent = [(-0.5, -0.5, -0.5), (0.5, 0.5, 0.5)]
        color3f[] primvars:displayColor = [(0.55, 0.45, 0.35)]
        double3 xformOp:translate = (0, 0, -0.025)
        float3 xformOp:scale = (1, 1.2, 0.05)
        uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:scale"]
    }

    def DomeLight "dome"
    {
        float inputs:intensity = 300
    }

    def DistantLight "sun"
    {
        float inputs:intensity = 1000
    }

    def Camera "agentview"
    {
        float2 clippingRange = (0.01, 100)
        float focalLength = 25.295
        float horizontalAperture = 20.955
        float verticalAperture = 20.955
        double3 xformOp:translate = (0.6586, 0, 0.7104)
        quatd xformOp:orient = (0.638018, 0.30485, 0.30485, 0.638018)
        uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:orient"]
    }

    def Xform "marks"
    {
        def Xform "plate_home"
        {
            double3 xformOp:translate = (0.05, 0.2, 0)
            uniform token[] xformOpOrder = ["xformOp:translate"]
        }

        def Cube "plate_region"
        {
            double size = 1
            float3[] extent = [(-0.5, -0.5, -0.5), (0.5, 0.5, 0.5)]
            uniform token purpose = "guide"
            double3 xformOp:translate = (0.05, 0.2, 0.03)
            float3 xformOp:scale = (0.06, 0.06, 0.06)
            uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:scale"]
        }
    }
}
"""
PLATE_USDA = """#usda 1.0
(
    defaultPrim = "plate"
    metersPerUnit = 1
    upAxis = "Z"
)

def Xform "plate" (
    prepend references = @@PLATE_USD@@
)
{
    over "Geometry"
    {
        over "Body" (
            prepend apiSchemas = ["PhysicsMassAPI"]
        )
        {
            float physics:mass = 0.0001
            float3 physics:diagonalInertia = (1e-9, 1e-9, 1e-9)

            over "PhysicsFixedJoint"
            {
                bool physics:jointEnabled = 0
            }
        }
    }
}
"""


def make(root: Path) -> None:
    """Write <root>/s3_fr3 and <root>/s3_panda (the source-package layout SceneManager registers by path)."""
    subs = {"@CAB_USD@": str(USD / "wooden_cabinet" / "wooden_cabinet.usda"),
            "@STOVE_USD@": str(USD / "flat_stove" / "flat_stove.usda"),
            "@PLATE_USD@": str(USD / "plate" / "plate.usda"),
            "@PLATE@": PLATE, "@DRAWER@": DRAWER, "@KNOB@": KNOB}
    for robot in ("fr3", "panda"):
        task = root / f"s3_{robot}"
        files = {f"s3_{robot}/scene.py": "from guide_core.scene.recipe_scene import RecipeScene as Scene\n",
                 "config/init.yaml": INIT, "config/scene.yaml": SCENE, "config/randomize.yaml": RANDOMIZE,
                 "config/reset.yaml": "instructions: []\n", "config/success.yaml": SUCCESS,
                 "config/task.yaml": TASK, "assets/stage.usda": STAGE, "assets/plate.usda": PLATE_USDA,
                 **{f"config/arrangements/{name}.yaml": yaml.safe_dump(values) for name, values in ARRANGEMENTS.items()}}
        for rel, text in files.items():
            for key, value in {**subs, "@ROBOT@": robot}.items():
                text = text.replace(key, value)
            (task / rel).parent.mkdir(parents=True, exist_ok=True)
            (task / rel).write_text(text)
        print(task)


class Io:
    """One rclpy node: GUIDE's services, /clock and the camera topics."""

    def __init__(self, rclpy):
        from rclpy.qos import qos_profile_sensor_data
        from rosgraph_msgs.msg import Clock

        self.rclpy, self.node, self.clients, self.now = rclpy, rclpy.create_node("s3_scene_check"), {}, None
        self.node.create_subscription(Clock, "/clock", self._tick, qos_profile_sensor_data)

    def _tick(self, msg):
        self.now = msg.clock.sec + msg.clock.nanosec * 1e-9

    def call(self, kind, name, request, timeout=900.0):
        if name not in self.clients:
            self.clients[name] = self.node.create_client(kind, name)
        client = self.clients[name]
        if not client.wait_for_service(timeout_sec=timeout):
            raise RuntimeError(f"{name} is not up after {timeout} s")
        future = client.call_async(request)
        self.rclpy.spin_until_future_complete(self.node, future, timeout_sec=timeout)
        if future.result() is None:
            raise RuntimeError(f"{name}: no reply in {timeout} s")
        return future.result()

    def register(self, path) -> int:
        from guide_msgs.srv import RegisterScene

        reply = self.call(RegisterScene, "/Sim_0/Register", RegisterScene.Request(path=str(path)))
        if not reply.success:
            raise RuntimeError(f"Register {path}: {reply.message}")
        return int(reply.id)

    def randomize(self, i: int, arrangement: str) -> bool:
        from guide_msgs.srv import Randomize

        return bool(self.call(Randomize, "/Sim_0/Randomize", Randomize.Request(id=i, arrangement=arrangement)).success)

    def settle(self, seconds: float) -> None:
        """Spin until the sim clock has run ``seconds`` past now."""
        deadline, start = time.monotonic() + 120.0 + 20.0 * seconds, None
        while start is None or self.now < start + seconds:
            if time.monotonic() > deadline:
                raise RuntimeError(f"/clock did not advance {seconds} s")
            self.rclpy.spin_once(self.node, timeout_sec=0.05)
            if start is None and self.now is not None:
                start = self.now

    def image(self, i: int, camera: str, png: Path):
        """One frame of ``camera`` (the topic ending cam_<camera>, Scene_<i>'s if the names say so):
        ((height, width), pixel spread); the frame saved to ``png``."""
        from PIL import Image as Picture
        from rclpy.qos import qos_profile_sensor_data
        from sensor_msgs.msg import Image

        deadline, topics = time.monotonic() + 60.0, []
        while not topics and time.monotonic() < deadline:
            self.rclpy.spin_once(self.node, timeout_sec=0.1)
            names = [n for n, _ in self.node.get_topic_names_and_types() if n.endswith(f"cam_{camera}")]
            topics = [n for n in names if f"Scene_{i}" in n] or names
        if not topics:
            raise RuntimeError(f"no topic ends in cam_{camera}")
        frames = []
        subscription = self.node.create_subscription(Image, topics[0], frames.append, qos_profile_sensor_data)
        while not frames and time.monotonic() < deadline + 60.0:
            self.rclpy.spin_once(self.node, timeout_sec=0.1)
        self.node.destroy_subscription(subscription)
        if not frames:
            raise RuntimeError(f"no image on {topics[0]}")
        msg = frames[0]
        pixels = np.frombuffer(bytes(msg.data), np.uint8).reshape(msg.height, msg.width, -1)
        Picture.fromarray(np.ascontiguousarray(pixels[..., :3])).save(png)
        return (int(msg.height), int(msg.width)), int(np.ptp(pixels))


def check(root: Path, out: Path) -> int:
    import rclpy
    from guide_msgs.srv import CheckSuccess
    from guide_msgs.srv import Pose as PoseSrv

    rclpy.init()
    io = Io(rclpy)
    report, failed = {}, []

    def gate(name, ok, **data):
        report[name] = {"ok": bool(ok), **data}
        print(f"{'PASS' if ok else 'FAIL'} {name} {data}", flush=True)
        if not ok:
            failed.append(name)

    def pose(path, frame=""):
        reply = io.call(PoseSrv, "/Sim_0/PoseRequest", PoseSrv.Request(path=path, frame_id=frame))
        if not reply.success:
            raise RuntimeError(f"PoseRequest {path}: {reply.message}")
        p, o = reply.pose.position, reply.pose.orientation
        return np.array([p.x, p.y, p.z]), R.from_quat([o.x, o.y, o.z, o.w])

    def error(i, path, position, euler_deg):
        """(m, deg) between the prim's pose in Scene_<i>'s frame and the expected one."""
        p, r = pose(f"/Scene_{i}{path}", f"Scene_{i}")
        wanted = R.from_euler("xyz", euler_deg, degrees=True)
        return float(np.linalg.norm(p - np.asarray(position))), float(np.degrees((r * wanted.inv()).magnitude()))

    def drawer(i):
        """The top drawer's travel along the cabinet base's y (spike J's measure)."""
        pb, rb = pose(f"/Scene_{i}{CAB}/object/base")
        pt, _ = pose(f"/Scene_{i}{CAB}/object/base/cabinet_top")
        return float(rb.inv().apply(pt - pb)[1])

    def knob(i):
        """The knob body's rotation in the stove base's frame."""
        _, rb = pose(f"/Scene_{i}{STOVE}/object/base")
        _, rk = pose(f"/Scene_{i}{STOVE}/object/base/button")
        return rb.inv() * rk

    try:
        scenes = {"fr3": io.register(root / "s3_fr3")}
        gate("scene0_a_before_scene1", io.randomize(scenes["fr3"], "a"))
        io.settle(0.25)
        scenes["panda"] = io.register(root / "s3_panda")  # stops and replays the timeline
        io.settle(2.0)
        m, deg = error(scenes["fr3"], CAB, A["cabinet_position"], A["cabinet_rotation"])
        gate("scene0_cabinet_stays_through_a_stop_and_play", m <= 0.001 and deg <= 0.1, error_m=m, error_deg=deg)

        for robot, i in scenes.items():
            reference = None
            for name in ("a", "b", "c"):
                values = ARRANGEMENTS[name]
                gate(f"scene{i}_{name}_randomize", io.randomize(i, name))
                io.settle(0.25)
                m, deg = error(i, PLATE, values["plate_position"], values["plate_rotation"])
                gate(f"scene{i}_{name}_plate", m <= 0.005 and deg <= 3.0, error_m=m, error_deg=deg)
                for key, path in (("cabinet", CAB), ("stove", STOVE)):
                    m, deg = error(i, path, values[f"{key}_position"], values[f"{key}_rotation"])
                    gate(f"scene{i}_{name}_{key}", m <= 0.001 and deg <= 0.1, error_m=m, error_deg=deg)
                d = drawer(i)
                gate(f"scene{i}_{name}_drawer", abs(d - values["cabinet_top_state"]) <= 0.001, value=d)
                k = knob(i)
                if reference is None:  # a: the knob at 0, the asset's own state
                    reference = k
                else:
                    q = float((k * reference.inv()).as_rotvec()[2]) + A["stove_knob_state"]
                    gate(f"scene{i}_{name}_knob", abs(q - values["stove_knob_state"]) <= 0.01, value=q)
                done = io.call(CheckSuccess, "/Sim_0/IsSuccess", CheckSuccess.Request(id=i)).success
                gate(f"scene{i}_{name}_is_success", done == SUCCEEDS[name], got=done)
            m, deg = error(i, f"/{robot}/{robot}_link0", MOUNT, [0.0, 0.0, 0.0])
            gate(f"scene{i}_robot_base_at_its_mount", m <= 0.001 and deg <= 0.1, error_m=m, error_deg=deg)
            for camera in ("agentview", "wrist"):
                shape, spread = io.image(i, camera, out.parent / f"scene{i}_{camera}.png")
                gate(f"scene{i}_{camera}_512", shape == (512, 512) and spread > 0, shape=list(shape), spread=spread)

        gate("free_draw", io.randomize(scenes["fr3"], ""))
        gate("bad_arrangement_fails", not io.randomize(scenes["fr3"], "bad"))
        gate("missing_arrangement_fails", not io.randomize(scenes["fr3"], "missing"))
    except Exception as e:  # noqa: BLE001 -- the report says where it stopped
        gate("ran_to_the_end", False, error=f"{type(e).__name__}: {e}")
    finally:
        out.write_text(json.dumps({"failed": failed, "gates": report}, indent=1))
        rclpy.shutdown()
    return 1 if failed else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("make").add_argument("dir", type=Path)
    run = commands.add_parser("check")
    run.add_argument("dir", type=Path)
    run.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "make":
        make(args.dir.expanduser())
        return 0
    return check(args.dir.expanduser(), args.out.expanduser())


if __name__ == "__main__":
    sys.exit(main())
