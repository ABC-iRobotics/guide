from __future__ import annotations

import asyncio
import time
from concurrent.futures import ThreadPoolExecutor
from threading import Thread
from typing import Any, Optional, Tuple

from guide_core.core.runtime import IsaacSimRuntime
from guide_core.scene.scene_manager import SceneManager
from guide_core.types.geometry import Point, Pose


class GUIDESimulator:

    _sim_id: int
    _sim_path: str

    _runtime: IsaacSimRuntime
    _scene_manager: SceneManager

    def __init__(self, sim_id: int = 0, namespace: Optional[str] = None):
        self.loop = asyncio.get_event_loop()
        self._sim_id = sim_id
        if namespace is not None:
            self._sim_path = f"/{namespace}"
        else:
            self._sim_path = f"/Sim_{sim_id}"

    # -------------------
    # Runtime functions
    # -------------------
    def init_runtime(self, config: Optional[dict] = None, debug: bool = False, logger: Any = None):
        self._logger = logger

        # 1. Start Recorder Server in a separate process before simulation starts
        from guide_core.core.recorder_manager import RecorderServer

        RecorderServer.start_server()

        # 2. Start Isaac Sim in this process natively
        self._runtime = IsaacSimRuntime(config=config, debug=debug, logger=self._logger)
        self._runtime._simulator = self

    def run_runtime_loop(self):
        self._runtime.run_loop()

    def call(self, name: str, timeout: Optional[float] = None, *args: Any, **kwargs: Any) -> Any:
        return self._runtime.call(name, timeout, *args, **kwargs)

    def _add_scene_to_runtime(
        self,
        scene_id: int,
        offset: Tuple[float, float, float] = (0, 0, 0),
        config: Optional[dict] = None,
    ):

        scene_path = f"/Scene_{scene_id}"
        pose: Pose = Pose(position=Point(offset))

        self._runtime.call("add_scene", stage_config=config, root=scene_path)
        self._runtime.call("set_world_poses", prim_path=scene_path, pose=pose)

    # --------------------------
    # Scene manager functions
    # --------------------------
    def init_scene_manager(self):
        from guide_core.scene.scene_manager import SceneManager

        self._scene_manager = SceneManager(sim_id=self._sim_id, logger=self._logger)

        if self._runtime._world:
            self._runtime._world.add_physics_callback(
                "scene_manager_step", self._scene_manager.step(self._runtime)
            )

    def register_scene(self, package_name: str) -> Tuple[int, Tuple[float, float, float]]:
        return self.call("register_scene", package_name=package_name)

    def _add_scene_to_manager(self, package_name: str):
        assert package_name is not None

        return self._scene_manager.add_scene(package_name)

    def _create_batches(self, instructions):
        batches = []
        batch = []
        for instruction in instructions:
            batch.append(instruction)
            if not instruction.get("parallel", False):
                batches.append(batch)
                batch = []
        if batch:
            batches.append(batch)

        return batches

    def _execute_batch(self, batch):
        with ThreadPoolExecutor() as executor:
            result = list(executor.map(self._proccess_instruction, batch))

        if any(
            instruction.get("reset_on", None) == result[i] for i, instruction in enumerate(batch)
        ):
            raise Exception("Reset condition is met during instruction execution.")

        return result

    def _proccess_instruction(self, instruction):
        cmd_name = instruction.get("cmd", None)
        args = instruction.get("args", [])
        kwargs = instruction.get("kwargs", {})

        assert cmd_name is not None

        if cmd_name == "sync":
            return

        self._logger.debug(
            f"[TRACE] _proccess_instruction: calling runtime.call('{cmd_name}', args={args}, kwargs={kwargs})"
        )

        result = self._runtime.call(cmd_name, *args, **kwargs)

        self._logger.debug(
            f"[TRACE] _proccess_instruction: runtime.call('{cmd_name}') returned: {result}"
        )
        return result

    def _execute_instructions(self, instructions):
        batches = self._create_batches(instructions)
        success = False
        while not success:
            result = []
            for batch in batches:
                try:
                    batch_result = self._execute_batch(batch)
                    result.extend(batch_result)
                    success = True
                except Exception as e:
                    success = False
                    print(f"Batch execution failed with error: {e}. Restarting batch execution.")
                    exit()
                    # break
        return result

    def reset_scene(self, scene_id: int) -> bool:
        return self.call("reset_scene", scene_id=scene_id)

    def randomize_scene(self, scene_id: int) -> bool:
        return self.call("randomize_scene", scene_id=scene_id)

    def is_success(self, scene_id: int) -> bool:
        return self.call("is_success", scene_id=scene_id)

    def pause(self):
        if self._runtime.is_running():
            self.call("pause")

    def play(self):
        self.call("start")

    def stop(self):
        if self._runtime.is_running():
            self.call("stop")


def main(sim: GUIDESimulator):
    for _ in range(2):
        try:
            sim.register_scene("/home/user/ros2_ws/src/block_bin_task/src/block_bin_task")
        except Exception as e:
            print(f"Exception: {e}")
            sim.call("shutdown")
            return
    # time.sleep(10)
    sim.call("start")
    time.sleep(10)
    print("Resetting scene")
    sim.reset_scene(0)
    time.sleep(10)
    for _ in range(20):
        print("Randomizing scene")
        sim.randomize_scene(0)
        time.sleep(1)
    time.sleep(270)
    sim.call("shutdown")


if __name__ == "__main__":
    sim = GUIDESimulator()
    sim.init_runtime(debug=True)
    sim.init_scene_manager()

    main_t = Thread(target=main, args=(sim,))
    main_t.start()

    sim.run_runtime_loop()

    main_t.join()
