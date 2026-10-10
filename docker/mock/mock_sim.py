#!/usr/bin/env python3
"""A stand-in GUIDE for communication tests: the same namespace, services and topics, no Isaac.

Register adds a scene and serves its Scene_<i>/generate_demonstration (in the real system the
task's solver does). A request writes a LeRobot-shaped dataset (meta/guide_episodes.jsonl, one
line per episode) and announces it on /Sim_<id>/dataset_finalized, as GUIDE does.
/Sim_<id>/shutdown exits. /Sim_<id>/clock runs at 10x real time, so two mocks' clocks differ.
"""

import argparse
import json
import os
import threading
import time
from datetime import datetime
from pathlib import Path

import rclpy
from guide_msgs.srv import Demonstration, RegisterScene
from rclpy.executors import ExternalShutdownException, MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile
from rosgraph_msgs.msg import Clock
from std_msgs.msg import String
from std_srvs.srv import Trigger


def episodes(zones: list, counts: list) -> list:
    if not zones:
        return [None] * counts[0]
    if zones == [-1]:
        return [z for z in (0, 1) for _ in range(counts[0])]  # the mock's grid: two zones
    return [z for z, n in zip(zones, counts) for _ in range(n)]


class MockSim(Node):
    def __init__(self, sim_id: int):
        super().__init__("GUIDE", namespace=f"Sim_{sim_id}")
        self.sim_id, self.scenes, self.t0 = sim_id, 0, time.monotonic()
        self.finalized = self.create_publisher(
            String, "dataset_finalized",
            QoSProfile(depth=100, durability=DurabilityPolicy.TRANSIENT_LOCAL))
        self.clock = self.create_publisher(Clock, "clock", 10)
        self.create_timer(0.01, self.tick)
        self.create_service(RegisterScene, "Register", self.register)
        self.create_service(Trigger, "shutdown", self.shutdown)

    def tick(self):
        t = (time.monotonic() - self.t0) * 10.0
        msg = Clock()
        msg.clock.sec, msg.clock.nanosec = int(t), int(t % 1 * 1e9)
        self.clock.publish(msg)

    def register(self, request, response):
        i = self.scenes
        self.scenes += 1
        self.create_service(Demonstration, f"Scene_{i}/generate_demonstration",
                            lambda req, res: self.generate(i, req, res))
        response.id, response.offset, response.success = i, [0.0, 2.0 * i, 0.0], True
        response.package = Path(request.path).name.removesuffix(".tar.gz")
        return response

    def generate(self, scene: int, request, response):
        threading.Thread(target=self.record, args=(scene, request), daemon=True).start()
        response.success, response.message = True, "Started generating."
        return response

    def record(self, scene: int, request):
        time.sleep(1.0)
        stamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        root = Path(request.path or "~/dataset").expanduser() / f"dataset_{self.sim_id}_{scene}_{stamp}"
        (root / "meta").mkdir(parents=True)
        eps = episodes(list(request.zones), list(request.counts))
        (root / "meta" / "info.json").write_text(json.dumps({"total_episodes": len(eps)}))
        (root / "meta" / "guide_episodes.jsonl").write_text("".join(
            json.dumps({"episode_index": k, "zone": z}) + "\n" for k, z in enumerate(eps)))
        self.finalized.publish(String(data=json.dumps({"scene": scene, "path": str(root)})))

    def shutdown(self, request, response):
        threading.Timer(1.0, lambda: os._exit(0)).start()  # answer first
        response.success, response.message = True, "Shutting down."
        return response


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--id", type=int, default=0)
    args, _ = parser.parse_known_args()
    rclpy.init()
    executor = MultiThreadedExecutor()
    executor.add_node(MockSim(args.id))
    try:
        executor.spin()
    except (KeyboardInterrupt, ExternalShutdownException):
        pass


if __name__ == "__main__":
    main()
