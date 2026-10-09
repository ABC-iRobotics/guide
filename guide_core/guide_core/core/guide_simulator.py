from __future__ import annotations

from typing import Any, Optional, Tuple

from guide_core.core.runtime import IsaacSimRuntime
from guide_core.scene.scene_manager import SceneManager


class GUIDESimulator:

    _sim_id: int
    _sim_path: str

    _runtime: IsaacSimRuntime
    _scene_manager: SceneManager

    def __init__(self, sim_id: int = 0, namespace: Optional[str] = None):
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

    # --------------------------
    # Scene manager functions
    # --------------------------
    def init_scene_manager(self):
        self._scene_manager = SceneManager(sim_id=self._sim_id, logger=self._logger)

        if self._runtime._world:
            self._runtime._world.add_physics_callback(
                "scene_manager_step", self._scene_manager.step(self._runtime)
            )
        elif self._logger is not None:
            self._logger.warning(
                "Runtime world is None; scene_manager_step physics callback not added."
            )

    def register_scene(self, package_name: str) -> Tuple[int, Tuple[float, float, float]]:
        return self.call("register_scene", package_name=package_name)

    def _add_scene_to_manager(self, package_name: str):
        assert package_name is not None

        return self._scene_manager.add_scene(package_name)

    def reset_scene(self, scene_id: int) -> bool:
        return self.call("reset_scene", scene_id=scene_id)

    def randomize_scene(
        self, scene_id: int, use_zone: bool = False, zone: int = 0, seed: int | None = None
    ) -> bool:
        # seed=None keeps the scene's own (scene_id, episode_index) stream, which is what
        # demonstration generation wants: every episode a fresh layout. A caller that
        # passes one is asking for a REPEATABLE layout -- see Randomize.srv.
        return self.call(
            "randomize_scene", scene_id=scene_id, use_zone=use_zone, zone=zone, seed=seed
        )

    def is_success(self, scene_id: int) -> bool:
        return self.call("is_success", scene_id=scene_id)

    def play(self):
        self.call("start")

    def stop(self):
        if self._runtime.is_running():
            self.call("stop")

