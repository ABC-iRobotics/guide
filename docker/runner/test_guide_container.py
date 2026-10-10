"""The container runner's pure parts: plan, splitting, DDS config, completeness, delivery."""

import json
import os
import random
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

import guide_container as c

PLAN = """
output: s3://bucket/guide
jobs:
  - {task: block_bin, zones: [-1], counts: [5]}
  - {task: s3://tasks/cube_stack.tar.gz, counts: [30]}
"""


def test_a_plan_is_read_with_its_defaults():
    plan = c.load_plan(PLAN)
    assert plan["output"] == "s3://bucket/guide"
    assert plan["jobs"][1] == {"task": "s3://tasks/cube_stack.tar.gz", "zones": [], "counts": [30]}


@pytest.mark.parametrize("jobs, reason", [
    ("[]", "non-empty"),
    ("[{zones: [1], counts: [2]}]", "'task'"),
    ("[{task: t, zones: [1, 2], counts: [3]}]", "one count per distinct zone"),
    ("[{task: t, zones: [2, 2], counts: [1, 1]}]", "one count per distinct zone"),
    ("[{task: t, zones: [-2], counts: [1]}]", "one count per distinct zone"),
    ("[{task: t, zones: [-1], counts: [1, 2]}]", "exactly one count"),
    ("[{task: t, counts: [0]}]", "positive"),
])
def test_bad_plans_are_rejected(jobs, reason):
    with pytest.raises(ValueError, match=reason):
        c.load_plan(f"jobs: {jobs}")


def test_work_expands_every_zone_and_keeps_free_draws_whole():
    assert c.work({"zones": [-1], "counts": [5]}, 3) == [(0, 5), (1, 5), (2, 5)]
    assert c.work({"zones": [-1], "counts": [5]}, 1) == [(None, 5)]
    assert c.work({"zones": [], "counts": [30]}, 20) == [(None, 30)]
    assert c.work({"zones": [2, 16], "counts": [4, 10]}, 20) == [(2, 4), (16, 10)]


def test_spare_scenes_go_to_the_heaviest_job():
    assert c.allocate([100, 30], [20, 30], 4) == [3, 1]
    assert c.allocate([100, 30], [2, 30], 4) == [2, 2]  # a zoned job can't use more scenes than zones
    assert c.allocate([5], [5], 1) == [1]


def test_a_job_is_dealt_evenly():
    parts = c.deal(c.work({"zones": [-1], "counts": [5]}, 20), 3)
    assert sorted(sum(p["counts"]) for p in parts) == [30, 35, 35]
    assert sorted(z for p in parts for z in p["zones"]) == list(range(20))
    assert c.deal([(None, 10)], 3) == [
        {"zones": [], "counts": [4]}, {"zones": [], "counts": [3]}, {"zones": [], "counts": [3]},
    ]


IP_JSON = json.dumps([
    {"ifname": "lo", "addr_info": [{"local": "127.0.0.1"}]},
    {"ifname": "eth0", "addr_info": [{"local": "10.42.0.7"}]},
    {"ifname": "eth1", "addr_info": [{"local": "10.0.9.3"}]},
])


def test_the_overlay_address_is_picked_by_subnet():
    assert c.overlay_ip("10.42.0.0/24", IP_JSON) == "10.42.0.7"
    with pytest.raises(RuntimeError, match="guide-net"):
        c.overlay_ip("10.99.0.0/24", IP_JSON)


def test_dds_stays_on_the_overlay_and_peers_with_the_master():
    ns = {"c": "https://cdds.io/config"}
    root = ET.fromstring(c.dds_config("10.42.0.7", ["guide-master"]))
    assert [i.get("address") for i in root.iterfind(".//c:NetworkInterface", ns)] == ["10.42.0.7"]
    assert root.find(".//c:AllowMulticast", ns).text == "false"
    assert [p.get("address") for p in root.iterfind(".//c:Peer", ns)] == ["10.42.0.7", "guide-master"]


def dataset(tmp_path, zones):
    d = tmp_path / "scratch" / "Sim_0" / "scene_0" / "dataset_0_0_x"
    (d / "meta").mkdir(parents=True)
    (d / "meta" / "guide_episodes.jsonl").write_text(
        "".join(json.dumps({"episode_index": i, "zone": z}) + "\n" for i, z in enumerate(zones)))
    (d / "data").mkdir()
    (d / "data" / "file-000.parquet").write_bytes(b"x")
    return d


def test_completeness_per_scene(tmp_path):
    counts = c.zone_counts(dataset(tmp_path, [2, 2, 16]))
    assert c.complete({"zones": [2, 16], "counts": [2, 1]}, counts)
    assert not c.complete({"zones": [2, 16], "counts": [2, 2]}, counts)
    assert c.complete({"zones": [], "counts": [3]}, counts)


def test_delivery_to_a_folder_moves_it_under_the_simulator(tmp_path):
    d = dataset(tmp_path, [0])
    out = tmp_path / "out"
    out.mkdir()
    assert c.deliver(d, str(out), "Sim_3") == str(out / "Sim_3" / "dataset_0_0_x")
    assert (out / "Sim_3" / "dataset_0_0_x" / "data" / "file-000.parquet").stat().st_uid == out.stat().st_uid
    assert not d.exists()


class FakeS3:
    def __init__(self, fail=False):
        self.keys, self.fail = [], fail

    def upload_file(self, path, bucket, key):
        if self.fail:
            raise ConnectionError("endpoint unreachable")
        self.keys.append((bucket, key))


def test_delivery_to_s3_uploads_every_file_then_frees_scratch(tmp_path):
    d = dataset(tmp_path, [0])
    s3 = FakeS3()
    assert c.deliver(d, "s3://bucket/guide", "Sim_3", s3=s3) == "s3://bucket/guide/Sim_3/dataset_0_0_x"
    assert sorted(s3.keys) == [
        ("bucket", "guide/Sim_3/dataset_0_0_x/data/file-000.parquet"),
        ("bucket", "guide/Sim_3/dataset_0_0_x/meta/guide_episodes.jsonl"),
    ]
    assert not d.exists()


def test_a_failed_upload_keeps_the_dataset(tmp_path):
    d = dataset(tmp_path, [0])
    with pytest.raises(ConnectionError):
        c.deliver(d, "s3://bucket/guide", "Sim_3", s3=FakeS3(fail=True))
    assert (d / "meta" / "guide_episodes.jsonl").is_file()


def test_a_plan_is_read_from_a_file_or_s3(tmp_path):
    f = tmp_path / "plan.yaml"
    f.write_text(PLAN)
    assert c.read_text(str(f)) == PLAN

    class Body:
        def read(self):
            return PLAN.encode()

    class S3:
        def get_object(self, Bucket, Key):
            assert (Bucket, Key) == ("plans", "a/plan.yaml")
            return {"Body": Body()}

    assert c.read_text("s3://plans/a/plan.yaml", s3=S3()) == PLAN


MOCK = Path(__file__).resolve().parents[1] / "mock" / "mock_sim.py"

FAKE_GUIDE = """
import json, pathlib, sys, time
import rclpy
from rclpy.qos import DurabilityPolicy, QoSProfile
from std_msgs.msg import String
d = pathlib.Path(sys.argv[1]); (d / "meta").mkdir(parents=True)
(d / "meta" / "info.json").write_text("{}")
rclpy.init()
node = rclpy.create_node("GUIDE", namespace="Sim_7")
pub = node.create_publisher(String, "dataset_finalized",
                            QoSProfile(depth=100, durability=DurabilityPolicy.TRANSIENT_LOCAL))
pub.publish(String(data=json.dumps({"scene": 0, "path": str(d)})))
time.sleep(5)  # long enough for the runner to discover us and read the latched message
"""


def env_for(tmp_path, monkeypatch, **extra):
    monkeypatch.setenv("CYCLONEDDS_URI", "")  # main() sets it; monkeypatch restores it
    monkeypatch.setenv("ROS_DOMAIN_ID", str(random.randint(100, 200)))  # private: never meet a live GUIDE
    return {**os.environ, "GUIDE_SCRATCH": str(tmp_path / "scratch"), **extra}


def test_slave_mode_delivers_what_guide_finalizes(tmp_path, monkeypatch):
    fake = tmp_path / "fake_guide.py"
    fake.write_text(FAKE_GUIDE)
    made = tmp_path / "scratch" / "Sim_7" / "dataset_7_0_x"
    out = tmp_path / "out"
    out.mkdir()

    code = c.main(env_for(tmp_path, monkeypatch, GUIDE_SIM_ID="7", GUIDE_OUTPUT=str(out),
                          GUIDE_CMD=f"{sys.executable} {fake} {made}"))

    assert code == 0
    assert (out / "Sim_7" / "dataset_7_0_x" / "meta" / "info.json").is_file()


def test_an_unreachable_bucket_keeps_the_dataset(tmp_path, monkeypatch):
    fake = tmp_path / "fake_guide.py"
    fake.write_text(FAKE_GUIDE)
    made = tmp_path / "scratch" / "Sim_7" / "dataset_7_0_x"
    # boto3 reads the process environment, not the runner's env argument.
    for key, value in {"AWS_ENDPOINT_URL": "http://127.0.0.1:9", "AWS_ACCESS_KEY_ID": "x",
                       "AWS_SECRET_ACCESS_KEY": "y", "AWS_MAX_ATTEMPTS": "1"}.items():
        monkeypatch.setenv(key, value)

    code = c.main(env_for(tmp_path, monkeypatch, GUIDE_SIM_ID="7", GUIDE_OUTPUT="s3://guide/out",
                          GUIDE_CMD=f"{sys.executable} {fake} {made}"))

    assert code == 0  # GUIDE was fine; the dataset just could not leave
    assert (made / "meta" / "info.json").is_file()


def test_a_plan_runs_to_the_end_on_the_mock(tmp_path, monkeypatch):
    plan = tmp_path / "plan.yaml"
    plan.write_text("jobs:\n  - {task: block_bin, zones: [1, 2], counts: [2, 1]}\n"
                    "  - {task: s3://t/cube_stack.tar.gz, counts: [2]}\n")
    out = tmp_path / "out"
    out.mkdir()

    started = time.monotonic()
    code = c.main(env_for(tmp_path, monkeypatch, GUIDE_PLAN=str(plan), GUIDE_OUTPUT=str(out),
                          GUIDE_MAX_SCENES="3", GUIDE_CMD=f"{sys.executable} {MOCK}"))

    assert code == 0 and time.monotonic() - started < 120
    assert len(list((out / "Sim_0").iterdir())) == 3  # block_bin split in two, cube_stack whole


def test_a_bad_plan_never_starts_guide(tmp_path, monkeypatch):
    plan = tmp_path / "plan.yaml"
    plan.write_text("jobs:\n  - {task: block_bin, zones: [1, 1], counts: [2, 2]}\n")
    marker = tmp_path / "started"

    code = c.main(env_for(tmp_path, monkeypatch, GUIDE_PLAN=str(plan),
                          GUIDE_CMD=f"touch {marker}"))

    assert code == 1 and not marker.exists()
