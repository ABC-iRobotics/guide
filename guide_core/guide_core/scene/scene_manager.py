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
                # 1. Filesystem path. Two layouts, both with config/ and assets/ beside
                #    the returned scene_path: the source package (<dir>/<dir>/scene.py,
                #    the same shape as the installed share dir) and the flat one
                #    (<dir>/scene.py, like guide_core/dummy_scene).
                pkg_scene = path_obj / path_obj.name / "scene.py"
                if path_obj.is_dir() and pkg_scene.exists():
                    scene_class = self.import_class_from_path(
                        str(pkg_scene.parent), "scene.py", "Scene"
                    )
                    scene_path = str(path_obj)
                elif path_obj.is_dir() and not (path_obj / "scene.py").exists():
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
                # Setup ROS 2 Flag
                ros2_enabled = False
                try:
                    from ament_index_python.packages import get_package_share_directory

                    ros2_enabled = True
                except ImportError:
                    pass

                # 2. ROS Package (Primary if ROS 2 is enabled)
                if ros2_enabled:
                    try:
                        share_dir = get_package_share_directory(package_name)
                        # According to standard ROS 2 python package install structure (e.g. block_bin),
                        # the python files are copied/symlinked into share_dir / package_name
                        ros_pkg_path = Path(share_dir) / package_name
                        scene_file_path = ros_pkg_path / "scene.py"

                        if scene_file_path.exists():
                            scene_class = self.import_class_from_path(
                                str(ros_pkg_path), "scene.py", "Scene"
                            )
                            scene_path = share_dir
                    except Exception:
                        pass

                # 3. Python package (Fallback)
                if scene_class is None:
                    spec = importlib.util.find_spec(package_name)
                    if spec is not None:
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
                    raise ImportError(
                        f"Failed to load Scene class for {package_name}. It is not a valid path, ROS package, or python package."
                    )

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

    def randomize_preprocess(self, scene_id: int, seed=None, inject=None, zone=None):
        # Drawing now happens inside the scene's randomize() lifecycle (seeded +
        # captured). It mutates randomize_instructions in place with concrete,
        # drawn poses and records every value; we just return the instructions.
        # `zone` (>=0) places the scene's zone target in that grid cell.
        self._scenes[scene_id].randomize(seed=seed, inject=inject, zone=zone)
        return self._scenes[scene_id].randomize_instructions

    def get_last_record_json(self, scene_id: int) -> str:
        ctx = getattr(self._scenes[scene_id], "_last_context", None)
        if ctx is None or ctx.record is None:
            return ""
        return ctx.to_json()

    def randomize_postprocess(self, scene_id: int, result):
        try:
            result = self._scenes[scene_id].randomize_postprocess(result)
        except NotImplementedError:
            pass

        # Capture this episode's reproduction metadata for the dataset sidecar:
        # seed + drawn values from the record, task/target/goal from the result.
        self._capture_episode_meta(scene_id, result)

        return result

    def _capture_episode_meta(self, scene_id: int, result):
        """Assemble per-episode sidecar metadata and hand it to the scene's recorder.

        Stays in the core layer: reads the seeded ``_last_context`` and the scene's
        ``task``, and pulls ``target``/``goal`` from the randomize_postprocess result
        when it is a JSON object exposing them (generic — skipped otherwise).
        """
        scene = self._scenes[scene_id]
        recorder = getattr(scene, "recorder", None)
        if recorder is None:
            return
        try:
            meta: Dict[str, Any] = {"scene_id": int(scene_id), "task": getattr(scene, "task", "")}
            ctx = getattr(scene, "_last_context", None)
            if ctx is not None:
                meta["draw_index"] = int(ctx.episode_index)
                if ctx.record is not None:
                    meta["seed"] = int(ctx.record.seed)
                    meta["values"] = ctx.record.to_dict().get("values", {})
                # Zone metadata: the requested zone (or None) + its cell bounds.
                grid = getattr(scene, "_grid", None)
                if grid is not None:
                    meta["zone"] = None if ctx.zone is None else int(ctx.zone)
                    if ctx.zone is not None and ctx.zone >= 0:
                        cl, ch = grid.cell_bounds(int(ctx.zone))
                        meta["zone_cell"] = {
                            "low": [float(x) for x in cl],
                            "high": [float(x) for x in ch],
                        }
            if isinstance(result, str):
                try:
                    parsed = json.loads(result)
                    if isinstance(parsed, dict):
                        for k in ("target", "goal"):
                            if k in parsed:
                                meta[k] = parsed[k]
                except (ValueError, TypeError):
                    pass
            # Per-robot starting joint configuration (varies per episode since the
            # robot is not reset to a fixed home).
            try:
                start_state = scene.get_start_state()
                if start_state:
                    meta["start_state"] = start_state
            except Exception:
                pass
            # Main randomized object's pose: the manipulated target, pulled from the
            # recorded draw values (keyed by prim path) via the target path.
            target = meta.get("target")
            if target is not None:
                meta["main_object"] = {
                    "prim": target,
                    "pose": self._main_object_pose(meta.get("values") or {}, target),
                }
            recorder.set_pending_episode_meta(meta)
        except Exception as e:
            self._logger.debug(f"Could not capture episode meta: {e}")

    @staticmethod
    def _main_object_pose(values: dict, target: str):
        """Pose of the main randomized object (the target) from the recorded draw
        values. Value keys are scene-prefixed prim paths, possibly wildcards
        (e.g. ``/Scene_0/blocks/*``); match on the target's parent-path suffix."""
        if not isinstance(values, dict) or not target:
            return None
        tparent = str(target).rsplit("/", 1)[0]  # "/blocks" ("" for a top-level prim)
        for k, v in values.items():
            if not isinstance(v, (list, tuple)):
                continue
            base = str(k).split("*", 1)[0].rstrip("/")  # e.g. "/Scene_0/blocks"
            # Exact prim (key "/Scene_0/bin_0" vs target "/bin_0") or wildcard
            # child (key "/Scene_0/blocks/*" vs target "/blocks/red_block").
            if base.endswith(str(target)) or (tparent and base.endswith(tparent)):
                return list(v)
        return None

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

    def start_recording(self, scene_id: int, path: str = ""):
        with self._locks[scene_id]:
            self._scenes[scene_id].state = SceneState.PREPARATION
            # Forward the requested dataset base dir to the recorder (empty => ~/dataset).
            self._scenes[scene_id].recorder.set_output_path(path)
            # self._scenes[scene_id].recorder.clear_start_recording()
            if hasattr(self._scenes[scene_id], "clear_recording_history"):
                self._scenes[scene_id].clear_recording_history()

    def stop_recording(self, scene_id: int, save_episode: bool = True):
        with self._locks[scene_id]:
            self._scenes[scene_id].recorder.clear_start_recording()
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
