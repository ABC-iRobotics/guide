"""GUIDE container runner: one simulator, driven by a plan or by a master over ROS 2.

Container glue only -- GUIDE itself runs unchanged as `GUIDE --id N`. Environment:
  GUIDE_SIM_ID      simulator id -> /Sim_<id> (default 0)
  GUIDE_PLAN        plan file or s3:// URL; unset = slave mode (a master drives GUIDE)
  GUIDE_OUTPUT      directory or s3://bucket/prefix for finished datasets (overrides the plan's)
  GUIDE_MAX_SCENES  scenes a plan may use (default: one per job)
  GUIDE_MASTER      the master's name or address on guide-net: the DDS peer besides us
  GUIDE_DDS_NET     guide-net's subnet, e.g. 10.42.0.0/24 (unset: DDS on localhost only)
  GUIDE_SCRATCH     recording directory (default /scratch)
"""

from __future__ import annotations

import ipaddress
import itertools
import json
import os
import queue
import shlex
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
from collections import Counter
from pathlib import Path

import yaml

from guide_core.ros.task_bringup import s3_client, split_s3


def read_text(location: str, s3=None) -> str:
    if not location.startswith("s3://"):
        return Path(location).read_text()
    bucket, key = split_s3(location)
    return (s3 or s3_client()).get_object(Bucket=bucket, Key=key)["Body"].read().decode()


def load_plan(text: str) -> dict:
    plan = yaml.safe_load(text) or {}
    jobs = plan.get("jobs")
    if not isinstance(jobs, list) or not jobs:
        raise ValueError("plan: 'jobs' must be a non-empty list")
    return {"output": plan.get("output"), "jobs": [_job(i, job) for i, job in enumerate(jobs)]}


def _job(i: int, job: dict) -> dict:
    """Demonstration.srv's rules: [] = free draws and [-1] = every zone, one count each;
    otherwise one count per distinct zone >= 0."""
    where = f"plan: job {i}"
    task, zones, counts = job.get("task"), job.get("zones", []), job.get("counts")
    if not isinstance(task, str) or not task:
        raise ValueError(f"{where}: 'task' must name a package, a directory or an s3:// bundle")
    if not isinstance(counts, list) or not counts or not all(
        isinstance(n, int) and n > 0 for n in counts
    ):
        raise ValueError(f"{where}: 'counts' must be positive integers")
    if not isinstance(zones, list) or not all(isinstance(z, int) for z in zones):
        raise ValueError(f"{where}: 'zones' must be a list of integers")
    if zones in ([], [-1]):
        if len(counts) != 1:
            raise ValueError(f"{where}: zones {zones} take exactly one count")
    elif len(zones) != len(counts) or len(set(zones)) != len(zones) or min(zones) < 0:
        raise ValueError(f"{where}: one count per distinct zone >= 0")
    return {"task": task, "zones": zones, "counts": counts}


def work(job: dict, num_zones: int) -> list:
    """(zone, count) pairs; zone None = free draws. [-1] becomes every zone of the task's grid."""
    zones, counts = job["zones"], job["counts"]
    if not zones or (zones == [-1] and num_zones <= 1):
        return [(None, counts[0])]
    if zones == [-1]:
        return [(z, counts[0]) for z in range(num_zones)]
    return list(zip(zones, counts))


def capacity(items: list) -> int:
    """How many scenes a job's work can be spread over."""
    return items[0][1] if items[0][0] is None else len(items)


def allocate(loads: list, caps: list, scenes: int) -> list:
    """Scenes per job: one each, then every spare scene to the job with the most episodes per scene."""
    k = [1] * len(loads)
    for _ in range(scenes - len(loads)):
        open_jobs = [i for i in range(len(loads)) if k[i] < caps[i]]
        if not open_jobs:
            break
        i = max(open_jobs, key=lambda j: loads[j] / k[j])
        k[i] += 1
    return k


def deal(items: list, k: int) -> list:
    """A job's work cut into k scenes: free draws split by count, zones dealt whole, heaviest first."""
    if items[0][0] is None:
        n = items[0][1]
        return [{"zones": [], "counts": [n // k + (i < n % k)]} for i in range(k)]
    parts = [{"zones": [], "counts": []} for _ in range(k)]
    for zone, count in sorted(items, key=lambda zc: -zc[1]):
        part = min(parts, key=lambda p: sum(p["counts"]))
        part["zones"].append(zone)
        part["counts"].append(count)
    return parts


def overlay_ip(subnet: str, ip_json: str | None = None) -> str:
    net = ipaddress.ip_network(subnet)
    if ip_json is None:
        ip_json = subprocess.run(
            ["ip", "-j", "-4", "addr"], capture_output=True, text=True, check=True
        ).stdout
    for iface in json.loads(ip_json):
        for addr in iface.get("addr_info", []):
            if ipaddress.ip_address(addr["local"]) in net:
                return addr["local"]
    raise RuntimeError(f"no interface on {subnet}: is the container attached to guide-net?")


def dds_config(ip: str, peers: list) -> str:
    peer_xml = "".join(f'<Peer address="{p}"/>' for p in [ip, *peers])
    return f"""<?xml version="1.0" encoding="UTF-8" ?>
<!-- Written by docker/runner/guide_container.py: DDS on the guide-net interface only, unicast
     discovery (overlay networks carry no multicast) of this container and the master. -->
<CycloneDDS xmlns="https://cdds.io/config">
  <Domain id="any">
    <General>
      <Interfaces><NetworkInterface address="{ip}"/></Interfaces>
      <AllowMulticast>false</AllowMulticast>
    </General>
    <Discovery>
      <ParticipantIndex>auto</ParticipantIndex>
      <MaxAutoParticipantIndex>60</MaxAutoParticipantIndex>
      <Peers>{peer_xml}</Peers>
    </Discovery>
  </Domain>
</CycloneDDS>
"""


def zone_counts(dataset: Path) -> Counter:
    meta = dataset / "meta" / "guide_episodes.jsonl"
    lines = meta.read_text().splitlines() if meta.is_file() else []
    return Counter(json.loads(line).get("zone") for line in lines if line.strip())


def complete(scene: dict, counts: Counter) -> bool:
    if not scene["zones"]:
        return sum(counts.values()) == scene["counts"][0]
    return all(counts[z] == n for z, n in zip(scene["zones"], scene["counts"]))


def deliver(dataset: Path, output: str, ns: str, s3=None) -> str:
    """Move or upload one finished dataset; the scratch copy goes only once all of it is out."""
    if output.startswith("s3://"):
        bucket, prefix = split_s3(output)
        base = "/".join(p for p in (prefix.strip("/"), ns, dataset.name) if p)
        client = s3 or s3_client()
        for f in sorted(p for p in dataset.rglob("*") if p.is_file()):
            client.upload_file(str(f), bucket, f"{base}/{f.relative_to(dataset).as_posix()}")
        shutil.rmtree(dataset)
        return f"s3://{bucket}/{base}"
    target = Path(output) / ns / dataset.name
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(dataset), str(target))
    owner = Path(output).stat()  # the container runs as root: hand the files to the folder's owner
    for p in [target.parent, target, *target.rglob("*")]:
        os.chown(p, owner.st_uid, owner.st_gid)
    return str(target)


def guide_cmd(sim_id: int, env) -> list:
    if env.get("GUIDE_CMD"):  # the mock image's stand-in simulator; tests
        return shlex.split(env["GUIDE_CMD"]) + ["--id", str(sim_id)]
    from ament_index_python.packages import get_package_prefix

    exe = Path(get_package_prefix("guide_core")) / "lib" / "guide_core" / "GUIDE"
    return [env.get("ISAACSIM_PYTHON", sys.executable), str(exe), "--id", str(sim_id)]


def dds_uri(net, master) -> str:
    if not net:  # sealed: the localhost config GUIDE ships
        from ament_index_python.packages import get_package_share_directory

        return f"file://{get_package_share_directory('guide_core')}/config/cyclonedds_localhost.xml"
    path = Path(tempfile.gettempdir()) / "cyclonedds.xml"
    path.write_text(dds_config(overlay_ip(net), [master] if master else []))
    return f"file://{path}"


def wait_for_name(host: str) -> None:
    """Cyclone resolves peer names once, at start: wait until the master's name resolves."""
    for attempt in itertools.count():
        try:
            socket.gethostbyname(host)
            return
        except OSError:
            if attempt % 30 == 0:
                print(f"[container] waiting for {host} to resolve...", flush=True)
            time.sleep(1.0)


def wait(client, guide, timeout: float = 1800.0) -> None:
    # A cold start compiles shaders (minutes); a solver waits for MoveIt and joint states.
    deadline = time.monotonic() + timeout
    while not client.wait_for_service(timeout_sec=1.0):
        if guide.poll() is not None:
            raise RuntimeError("GUIDE exited")
        if time.monotonic() > deadline:
            raise TimeoutError(f"{client.srv_name} did not come up in {timeout:.0f} s")


def call(client, request, guide, timeout: float):
    done = threading.Event()
    future = client.call_async(request)
    future.add_done_callback(lambda _: done.set())
    deadline = time.monotonic() + timeout
    while not done.wait(1.0):
        if guide.poll() is not None:
            raise RuntimeError("GUIDE exited")
        if time.monotonic() > deadline:
            raise TimeoutError(f"{client.srv_name} did not answer in {timeout:.0f} s")
    return future.result()


def zone_count(package: str) -> int:
    """The task's zone grid, read the way its solver reads it (block_bin solve_task.scene_num_zones)."""
    from ament_index_python.packages import get_package_share_directory

    from guide_core.ros.task_bringup import TaskBringup
    from guide_core.types.randomization.replicator_guide import zone_grid

    TaskBringup(0).activate()  # tasks GUIDE fetched live in its overlay
    grid = zone_grid(str(Path(get_package_share_directory(package)) / "config" / "randomize.yaml"))
    return grid.num_zones if grid is not None else 1


def run_plan(node, plan, cap, scratch, finalized, handle, guide) -> bool:
    from guide_msgs.srv import Demonstration, RegisterScene
    from std_srvs.srv import Trigger

    register = node.create_client(RegisterScene, "Register")
    wait(register, guide)

    def add(task):
        # Register fetches and builds an unknown task first: allow for a long build.
        reply = call(register, RegisterScene.Request(path=task, bringup=True), guide, 3600)
        if not reply.success:
            raise RuntimeError(f"Register {task!r}: {reply.message}")
        return reply.id, reply.package

    jobs = plan["jobs"]
    firsts = [add(job["task"]) for job in jobs]
    items = [work(job, zone_count(pkg) if job["zones"] == [-1] else 1)
             for job, (_, pkg) in zip(jobs, firsts)]
    shares = allocate([sum(n for _, n in it) for it in items], [capacity(it) for it in items], cap)
    scenes = {}
    for job, (first, _), it, k in zip(jobs, firsts, items, shares):
        parts = deal(it, k)
        scenes[first] = parts[0]
        for part in parts[1:]:
            scenes[add(job["task"])[0]] = part
    for sid, part in scenes.items():
        client = node.create_client(Demonstration, f"Scene_{sid}/generate_demonstration")
        wait(client, guide)
        request = Demonstration.Request(
            path=str(scratch / f"scene_{sid}"), zones=part["zones"], counts=part["counts"])
        reply = call(client, request, guide, 60)
        if not reply.success:
            raise RuntimeError(f"scene {sid}: {reply.message}")
    results = {}
    while len(results) < len(scenes):
        if guide.poll() is not None:
            raise RuntimeError("GUIDE exited before the plan finished")
        try:
            event = finalized.get(timeout=1.0)
        except queue.Empty:
            continue
        results[event["scene"]] = handle(event, scenes.get(event["scene"]))
    # The same exit a master uses.
    call(node.create_client(Trigger, "shutdown"), Trigger.Request(), guide, 60)
    return all(results.values())


def main(env=None) -> int:
    import rclpy
    from rclpy.executors import MultiThreadedExecutor
    from rclpy.qos import DurabilityPolicy, QoSProfile
    from rclpy.signals import SignalHandlerOptions
    from std_msgs.msg import String

    env = os.environ if env is None else env
    sim_id = int(env.get("GUIDE_SIM_ID") or 0)
    ns = f"Sim_{sim_id}"
    try:
        plan = load_plan(read_text(env["GUIDE_PLAN"])) if env.get("GUIDE_PLAN") else None
        cap = int(env.get("GUIDE_MAX_SCENES") or (len(plan["jobs"]) if plan else 0))
        if plan and len(plan["jobs"]) > cap:
            raise ValueError(f"plan: {len(plan['jobs'])} jobs need as many scenes; GUIDE_MAX_SCENES is {cap}")
    except (OSError, ValueError) as e:
        print(f"[container] {e}", flush=True)
        return 1  # before Isaac starts
    output = env.get("GUIDE_OUTPUT") or (plan or {}).get("output")
    scratch = Path(env.get("GUIDE_SCRATCH") or "/scratch") / ns
    scratch.mkdir(parents=True, exist_ok=True)
    master = env.get("GUIDE_MASTER")
    if master:
        wait_for_name(master)
    os.environ["CYCLONEDDS_URI"] = dds_uri(env.get("GUIDE_DDS_NET"), master)  # GUIDE, its launches, us

    guide = subprocess.Popen(guide_cmd(sim_id, env))
    # docker stop / service rm: hand SIGTERM to GUIDE, which shuts down as /shutdown does; we
    # keep delivering what it finalizes until it is gone.
    previous = {s: signal.signal(s, lambda *_: guide.send_signal(signal.SIGTERM))
                for s in (signal.SIGTERM, signal.SIGINT)}
    rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
    node = rclpy.create_node("guide_container", namespace=ns)
    latched = QoSProfile(depth=100, durability=DurabilityPolicy.TRANSIENT_LOCAL)
    finalized: queue.Queue = queue.Queue()
    node.create_subscription(String, "dataset_finalized",
                             lambda m: finalized.put(json.loads(m.data)), latched)
    announce = node.create_publisher(String, "dataset_delivered", latched)
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    threading.Thread(target=executor.spin, daemon=True).start()

    def handle(event: dict, scene: dict | None = None) -> bool:
        if not event["path"]:
            print(f"[container] scene {event['scene']} recorded nothing", flush=True)
            return False
        dataset = Path(event["path"])
        ok = complete(scene, zone_counts(dataset)) if scene else True
        try:
            target = deliver(dataset, output, ns) if output else str(dataset)
        except Exception as e:  # keep policy: the dataset stays in scratch
            print(f"[container] delivering {dataset} failed, kept in scratch: {e}", flush=True)
            target, ok = None, False
        announce.publish(String(data=json.dumps(
            {"dataset": dataset.name, "target": target, "complete": ok})))
        return ok

    try:
        ok = run_plan(node, plan, cap, scratch, finalized, handle, guide) if plan else True
    except Exception as e:
        print(f"[container] {e}", flush=True)
        ok = False
        if guide.poll() is None:
            guide.send_signal(signal.SIGTERM)
    # Slave mode serves until GUIDE exits (a master's /shutdown, or SIGTERM); a finished plan has
    # asked it to shut down already. Deliver whatever it finalizes on the way out.
    quiet_since = None
    while True:
        try:
            handle(finalized.get(timeout=1.0))
            quiet_since = None
        except queue.Empty:
            if guide.poll() is None:
                continue
            quiet_since = quiet_since or time.monotonic()
            if time.monotonic() - quiet_since > 3.0:  # the last announcements are in
                break
    for s, h in previous.items():
        signal.signal(s, h)
    rclpy.try_shutdown()
    return 0 if ok and guide.returncode == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
