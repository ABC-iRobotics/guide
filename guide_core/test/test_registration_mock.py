# flake8: noqa: E402
import multiprocessing

multiprocessing.set_start_method("fork", force=True)

import sys
import time
from unittest.mock import MagicMock

# 1. MOCK ALL ISAAC SIM AND ROS MODULES
sys.modules["isaacsim"] = MagicMock()
sys.modules["omni"] = MagicMock()
sys.modules["omni.replicator"] = MagicMock()
sys.modules["omni.replicator.core"] = MagicMock()
sys.modules["omni.isaac"] = MagicMock()
sys.modules["omni.isaac.core"] = MagicMock()
sys.modules["omni.isaac.core.utils"] = MagicMock()
sys.modules["omni.isaac.core.utils.stage"] = MagicMock()
sys.modules["omni.isaac.core.world"] = MagicMock()
sys.modules["rclpy"] = MagicMock()

import ament_index_python.packages

import guide_core.core.runtime as rt

rt.get_package_share_directory = (
    lambda x: ament_index_python.packages.get_package_share_directory("guide_core") + "/dummy_scene"
)

from guide_core.core.runtime import IsaacSimRuntime

IsaacSimRuntime._create_world = lambda self: setattr(self, "_world", MagicMock())


def mock_initialize(self, config=None):
    self.state = 5  # RUNNING
    self._dt = 1.0 / 60.0
    self._cmd_q = __import__("queue").Queue()
    self._create_world()


IsaacSimRuntime.initialize = mock_initialize

import guide_core.core.isaac_manager as im
from guide_core.core.guide_simulator import GUIDESimulator
from guide_core.core.recorder_manager import RecorderServer


def test_registration():
    print("1. Starting Recorder Server...")
    RecorderServer.start_server(address=("127.0.0.1", 50051), authkey=b"isaac_sim_recorder")

    print("2. Starting GUIDESimulator...")
    sim = GUIDESimulator(sim_id=0, namespace="GUIDE")

    print("3. Initializing runtime (starts isaac manager process)...")
    im.start_isaac_process = lambda config, debug, address, authkey: __start_isaac_process_fork(
        config, debug, address, authkey
    )

    def __start_isaac_process_fork(config, debug, address, authkey):
        ctx = multiprocessing.get_context("fork")
        p = ctx.Process(
            target=im.run_isaac_server, args=(config, debug, address, authkey), daemon=True
        )
        p.start()

        class IsaacBaseManager(im.BaseManager):
            pass

        IsaacBaseManager.register("get_runtime")
        IsaacBaseManager.register("get_scene_manager")

        manager = IsaacBaseManager(address=address, authkey=authkey)
        for _ in range(300):
            try:
                manager.connect()
                return manager, p
            except ConnectionRefusedError:
                time.sleep(0.1)
        raise RuntimeError("Failed to connect")

    sim.init_runtime(config={}, debug=True, logger=MagicMock())

    sim._runtime.call = MagicMock(return_value=True)

    print("4. Initializing scene manager...")
    sim.init_scene_manager()

    print("5. Registering scene...")
    try:
        from guide_core.core.recorder_manager import RecorderServer as RS

        RS.get_client = classmethod(
            lambda cls: RecorderServer.get_client(
                address=("127.0.0.1", 50051), authkey=b"isaac_sim_recorder"
            )
        )

        import ament_index_python.packages

        id, offset = sim.register_scene(
            ament_index_python.packages.get_package_share_directory("guide_core") + "/dummy_scene"
        )
        print(f"SUCCESS! Registered scene. ID: {id}, Offset: {offset}")
    except Exception as e:
        import traceback

        print(f"FAILED to register scene: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    test_registration()
