from __future__ import annotations

import importlib
import inspect
import json
import pkgutil
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Tuple

from guide_core.core.runtime import IsaacSimRuntime
from guide_core.scene.scene_orchestrator import SceneOrchestrator
from guide_core.types.scene_state import SceneState


class SceneManager:
    _scenes: List[SceneOrchestrator]
    _locks: Dict[int, Lock]

    def __init__(self, sim_id: int = 0, logger: Any = None):
        self._scenes = []
        self._locks = {}
        self._sim_id = sim_id
        self._logger = logger

    def get_scene_usd_path(self, scene_id: int):
        scene = self._scenes[scene_id]
        if hasattr(scene, "_usd_path"):
            return str(scene._usd_path)
        return None

    def get_scene_robot_graphs(self, scene_id: int):
        return self._scenes[scene_id].create_robot_graphs()

    def get_scene_camera_graphs(self, scene_id: int):
        return self._scenes[scene_id].create_camera_graphs()

    def wait_start_recording_event(self, scene_id: int, timeout=None):
        return self._scenes[scene_id].recorder.wait_start_recording(timeout)

    def wait_stop_recording_event(self, scene_id: int, timeout=None):
        return self._scenes[scene_id].recorder.wait_stop_recording(timeout)

    def wait_idle_event(self, scene_id: int, timeout=None):
        return self._scenes[scene_id].recorder.is_idle()

    def clear_idle_event(self, scene_id: int):
        # We don't need to clear idle_event if we are checking is_idle directly, but if needed:
        pass

    def add_scene(self, package_name: str) -> Tuple[int, Tuple[float, float, float], Dict]:
        try:
            self._logger.info(f"[SceneManager] add_scene started for package: {package_name}")

            scene_class = None
            scene_path = package_name

            path_obj = Path(package_name)
            if path_obj.is_absolute() or path_obj.exists():
                # 1. Filesystem Path
                if path_obj.is_dir() and not (path_obj / "scene.py").exists():
                    found_scenes = list(path_obj.rglob("scene.py"))
                    if found_scenes:
                        scene_file = found_scenes[0]
                        scene_class = self.import_class_from_path(
                            str(scene_file.parent), "scene.py", "Scene"
                        )
                        scene_path = str(scene_file.parent)
                    else:
                        raise FileNotFoundError(f"Could not find scene.py in {package_name}")
                else:
                    scene_class = self.import_class_from_path(package_name, "scene.py", "Scene")
            else:
                # Try to resolve as a Python or ROS package
                spec = importlib.util.find_spec(package_name)
                if spec is not None:
                    # 2. Python package
                    try:
                        module = importlib.import_module(f"{package_name}.scene")
                        scene_class = getattr(module, "Scene")
                        if hasattr(module, "__file__") and module.__file__:
                            scene_path = str(Path(module.__file__).parent)
                        elif spec.submodule_search_locations:
                            scene_path = spec.submodule_search_locations[0]
                        else:
                            scene_path = str(Path(spec.origin).parent)
                    except ImportError:
                        pass

                if scene_class is None:
                    try:
                        # 3. ROS Package
                        from ament_index_python.packages import (
                            get_package_share_directory,
                        )

                        share_dir = get_package_share_directory(package_name)
                        module = importlib.import_module(f"{package_name}.scene")
                        scene_class = getattr(module, "Scene")
                        scene_path = share_dir
                    except Exception:
                        raise ImportError(
                            f"Failed to load Scene class for {package_name}. It is not a valid path, python package, or ROS package."
                        )

            if scene_class is None:
                raise ImportError(f"Failed to load Scene class for {package_name}")

            id = len(self._scenes)
            scene = scene_class(
                scene_id=id, sim_id=self._sim_id, path=scene_path, logger=self._logger
            )
            self._scenes.append(scene)
            self._locks[id] = Lock()

            offset = self._calculate_offset(id)
            scene.set_offset(offset)

            self._logger.info(
                f"[SceneManager] add_scene finished successfully. ID: {id}, Offset: {offset}"
            )

            clean_config = {}
            try:
                clean_config = json.loads(json.dumps(self._scenes[id]._config))
            except Exception as e:
                self._logger.error(f"[SceneManager] Failed to JSON serialize config! Error: {e}")
                clean_config = {}

            return (id, offset, clean_config)
        except Exception as e:
            import traceback

            err_msg = traceback.format_exc()
            self._logger.error(f"[SceneManager] add_scene FAILED with exception:\n{err_msg}")
            # Return failure tuple safely over IPC instead of raising, to avoid lock pickling issues in traceback context
            return (-1, (0.0, 0.0, 0.0), {"error": err_msg})

    def import_class_from_path(self, package_path: str, module_file: str, class_name: str):
        package_path = Path(package_path)
        module_path = package_path / module_file

        if not module_path.exists():
            raise FileNotFoundError(f"Module file not found: {module_path}")

        spec = importlib.util.spec_from_file_location(
            module_path.stem,
            module_path,
        )
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load spec for {module_path}")

        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        cls = getattr(module, class_name, None)
        if not inspect.isclass(cls):
            raise ImportError(f"{class_name} not found in {module_path}")
        return cls

    def _verify_package(self, package_name: str, class_name: str) -> bool:
        assert package_name is not None
        assert class_name is not None

        try:
            package = importlib.import_module(package_name)
        except ImportError:
            print(f"{package_name} module is not found!")
            return False

        # top-level ellenőrzés
        if inspect.isclass(getattr(package, class_name, None)):
            return True

        # almodulok bejárása
        if hasattr(package, "__path__"):
            for _, modname, _ in pkgutil.walk_packages(package.__path__, package.__name__ + "."):
                try:
                    module = importlib.import_module(modname)
                except Exception:
                    continue

                if inspect.isclass(getattr(module, class_name, None)):
                    return True

        print(f"{class_name} class is not found!")
        return False

    def _import_class(self, module_name: str, class_name: str):
        module = importlib.import_module(module_name)
        try:
            cls = getattr(module, class_name)
        except AttributeError:
            raise ImportError(f"Module '{module_name}' does not define '{class_name}'")
        return cls

    def _calculate_offset(self, scene_id: int) -> Tuple[float, float, float]:
        offset = [0.0, 0.0, 0.0]
        for idx in range(scene_id):
            offset[1] = (
                offset[1]
                + self._scenes[idx].bounding_box["yp"]
                + self._scenes[idx].bounding_box["yn"]
            )

        offset[1] = (
            offset[1]
            - self._scenes[0].bounding_box["yn"]
            + self._scenes[scene_id].bounding_box["yp"]
        )

        offset[0] = offset[0] - self._scenes[scene_id].origin[0]
        offset[1] = offset[1] - self._scenes[scene_id].origin[1]
        offset[2] = offset[2] - self._scenes[scene_id].origin[2]

        return tuple(offset)

    def reset_preprocess(self, scene_id: int):
        instructions = self._scenes[scene_id].reset_instructions

        try:
            instructions = self._scenes[scene_id].reset_preprocess(instructions)
        except NotImplementedError:
            pass

        return instructions

    def reset_postprocess(self, scene_id: int, result):
        try:
            result = self._scenes[scene_id].reset_postprocess(result)
        except NotImplementedError:
            pass

        return result

    def randomize_preprocess(self, scene_id: int):
        instructions = self._scenes[scene_id].randomize_instructions

        try:
            instructions = self._scenes[scene_id].randomize_preprocess(instructions)
        except NotImplementedError:
            pass

        return instructions

    def randomize_postprocess(self, scene_id: int, result):
        try:
            result = self._scenes[scene_id].randomize_postprocess(result)
        except NotImplementedError:
            pass

        return result

    def is_success_preprocess(self, scene_id: int):
        instructions = self._scenes[scene_id].success_instructions
        try:
            instructions = self._scenes[scene_id].is_success_preprocess(instructions)
        except NotImplementedError:
            pass

        return instructions

    def is_success_postprocess(self, scene_id: int, result):
        try:
            result = self._scenes[scene_id].is_success_postprocess(result)
        except NotImplementedError:
            pass

        return result

    def step(self, runtime: IsaacSimRuntime):
        def step_task(step_size: float):
            current_step = runtime._world.current_time_step_index

            for scene_id, scene in enumerate(self._scenes):
                state = scene.state

                if state == SceneState.PREPARATION:
                    import omni.replicator.core as rep

                    if not getattr(scene, "render_products_ready", False):
                        # Ensure render products are created securely on sim thread only if requested
                        if (
                            hasattr(scene, "_config")
                            and "cameras" in scene._config
                            and scene._config["cameras"]
                        ):
                            scene.create_render_products(rep)
                        scene.render_products_ready = True
                        scene.warmup_frames = 0
                    else:
                        scene.warmup_frames += 1
                        if scene.warmup_frames > 10:
                            # check_warmup must run on sim thread natively
                            if scene.check_warmup():
                                scene.state = SceneState.RECORDING
                                scene.recorder.set_start_recording()

                elif state == SceneState.RECORDING:
                    f_sim = 120  # Hardware/Sim dependent
                    f_record = getattr(scene, "record_frequency", 10)
                    interval = max(1, f_sim // f_record)

                    if current_step % interval == 0:
                        try:
                            # record_step must run natively and return a frame dict
                            data = scene.record_step(current_step)
                            if data:
                                import queue

                                try:
                                    scene.recorder.put_record_data(data)
                                except queue.Full:
                                    pass  # Drop frame to not block physics
                        except Exception as e:
                            print(f"Error in record_step: {e}")

                elif state == SceneState.FINALIZING:
                    if scene.recorder.is_idle():
                        scene.state = SceneState.IDLE

        return step_task

    def start_recording(self, scene_id: int):
        with self._locks[scene_id]:
            self._scenes[scene_id].state = SceneState.PREPARATION
            self._scenes[scene_id].recorder.clear_start_recording()
            if hasattr(self._scenes[scene_id], "clear_recording_history"):
                self._scenes[scene_id].clear_recording_history()

    def stop_recording(self, scene_id: int, save_episode: bool = True):
        with self._locks[scene_id]:
            self._scenes[scene_id].state = SceneState.FINALIZING
            self._scenes[scene_id].recorder.clear_stop_recording()
            if save_episode:
                self._scenes[scene_id].recorder.put_record_data("FINALIZE_EPISODE")
            else:
                self._scenes[scene_id].recorder.put_record_data("DISCARD_EPISODE")

    def finalize_recording(self, scene_id: int):
        with self._locks[scene_id]:
            self._scenes[scene_id].state = SceneState.FINALIZING
            self._scenes[scene_id].recorder.clear_stop_recording()
            self._scenes[scene_id].recorder.put_record_data("FINALIZE")
            self._scenes[scene_id].recorder.set_start_recording()

    def finalize_all_recordings(self):
        for scene_id in range(len(self._scenes)):
            with self._locks[scene_id]:
                self._scenes[scene_id].state = SceneState.FINALIZING
                self._scenes[scene_id].recorder.clear_stop_recording()
                self._scenes[scene_id].recorder.put_record_data("SHUTDOWN")
                self._scenes[scene_id].recorder.set_start_recording()

        for scene_id in range(len(self._scenes)):
            self._scenes[scene_id].recorder.wait_shutdown(15.0)

    def get_scene_state(self, scene_id: int) -> SceneState:
        return self._scenes[scene_id].state

    def check_warmup(self, scene_id: int):
        try:
            return self._scenes[scene_id].check_warmup()
        except NotImplementedError:
            return []

    def record_step(self, scene_id: int):
        try:
            return self._scenes[scene_id].record_step()
        except NotImplementedError:
            return []
