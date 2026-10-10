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
import json
import os
import shutil
import subprocess
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
