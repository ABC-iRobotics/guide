from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from importlib import resources
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import yaml
from scipy.spatial.transform import Rotation as R

from guide_core.scene.scene_recorder import SceneRecorder
from guide_core.types.geometry import Point, Pose, Rotation
from guide_core.types.scene_state import SceneState

logger = logging.getLogger("SceneOrchestrator")


class SceneOrchestrator(ABC):

    scene_id: int

    _offset: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    _pkg_name: str

    _config: dict
    _usd_path: str
    bounding_box: dict
    origin: list

    reset_instructions: list[dict] = []
    randomize_instructions: list[dict] = []

    state: SceneState
    recorder: SceneRecorder

    def __init__(
        self,
        scene_id: int,
        sim_id: int = 0,
        path: Optional[str] = None,
        config_path: str = "/config/init.yaml",
        reset_path: str = "/config/reset.yaml",
        randomize_path: str = "/config/randomize.yaml",
        success_path: str = "/config/success.yaml",
        logger: Any = None,
    ):
        self._scene_id = scene_id
        self._sim_id = sim_id
        if logger is None:
            import logging

            self._logger = logging.getLogger("SceneOrchestrator")
        else:
            self._logger = logger

        self.task = ""

        # Getting init.yaml
        package_name = self.__class__.__module__.split(".")[0]

        if path is None:
            self._path = resources.files(package_name)
            config_file = resources.files(package_name).joinpath(config_path)
        else:
            self._path = Path(path)
            config_file = Path(f"{path}/{config_path}")

        # with resources.files(package_name).joinpath(config_path).open('r') as f:
        with config_file.open("r") as f:
            self._config = yaml.safe_load(f)

        self._get_usd_params(package_name)

        self._get_limits()
        self._get_origin()

        # Getting reset.yaml
        self.reset_instructions = self.parse_instruction(Path(f"{path}/{reset_path}"))

        # Getting randomize.yaml
        self.randomize_instructions = self.parse_instruction(Path(f"{path}/{randomize_path}"))

        # Getting success.yaml
        self.success_instructions = self.parse_instruction(Path(f"{path}/{success_path}"))

        self.state = SceneState.IDLE

        # Start separate recorder process
        dataset_name = f"dataset_{self._sim_id}_{self._scene_id}"
        from guide_core.core.recorder_manager import RecorderServer

        try:
            client = RecorderServer.get_client()
            self.recorder = client.get_recorder(package_name, dataset_name, self._config)
        except ConnectionRefusedError:
            self._logger.error(
                "RecorderServer not running! Ensure it was started in the Main Process."
            )
            self.recorder = None

    def _get_usd_params(self, package_name):
        self._usd_path = self._path.joinpath(self._config["usd_path"].lstrip("/"))

    def _get_limits(self):
        limits: Optional[dict] = self._config.get("limits", None)

        assert limits is not None

        # Creating bounding box
        self.bounding_box = {
            "xp": limits.get("xp", 0.0),
            "xn": limits.get("xn", 0.0),
            "yp": limits.get("yp", 0.0),
            "yn": limits.get("yn", 0.0),
            "zp": limits.get("zp", 0.0),
            "zn": limits.get("zn", 0.0),
        }

    def _get_origin(self):
        self.origin = self._config.get("origin", [0.0, 0.0, 0.0])

        assert self.origin is not None

    def set_offset(self, offset: Tuple[float, float, float]):
        self._offset = offset

    def create_robot_graphs(self):
        robot_list: List[Dict] = []
        for name, data in self._config.get("robots", {}).items():
            path = data["path"]
            assert path is not None
            default_joint_states = data["default_joint_states"]
            assert default_joint_states is not None
            robot_list.append(
                {
                    "namespace": f"/{name}",
                    "articulation_root": f"{path}",
                    "path": f"/{name}",
                    "default_joint_states": default_joint_states,
                }
            )
        return robot_list

    def create_camera_graphs(self):
        camera_list: List[Dict] = []
        for name, data in self._config.get("cameras", {}).items():
            path = data["path"]
            assert path is not None
            width = data["width"]
            assert width is not None
            height = data["height"]
            assert height is not None
            topic = data["topic"]
            assert topic is not None

            camera_list.append(
                {
                    "camera_path": f"{path}",
                    "path": f"/{name}",
                    "width": width,
                    "height": height,
                    "frame": f"{name}",
                    "topic": f"{topic}",
                }
            )
        return camera_list

    def parse_instruction(self, path: Path):

        relative_path_keywords = ["prim_path", "articulation_root", "scope"]

        with path.open("r") as f:
            file = yaml.safe_load(f)
        instructions = file.get("instructions", None)
        instruction_list = []
        for instruction in instructions:
            for key, value in instruction.get("kwargs", {}).items():
                if key in relative_path_keywords:
                    if isinstance(value, list):
                        instruction["kwargs"][key] = [f"/Scene_{self._scene_id}{v}" for v in value]
                    else:
                        instruction["kwargs"][key] = f"/Scene_{self._scene_id}{value}"
                if key == "pose":
                    # Position
                    position_dict = instruction["kwargs"]["pose"].get("position", None)
                    if position_dict is None:
                        position_dict = {"value": [0.0, 0.0, 0.0]}
                    position_base = np.array(position_dict.get("value", [0.0, 0.0, 0.0]))

                    randomize = position_dict.get("random", None)
                    if randomize is None:
                        randomize_low = None
                        randomize_high = None
                    else:
                        randomize_low = np.array(randomize.get("low", [0.0, 0.0, 0.0]))
                        randomize_high = np.array(randomize.get("high", [0.0, 0.0, 0.0]))

                    position = Point(
                        coordinates=position_base,
                        random_low=randomize_low,
                        random_high=randomize_high,
                    )

                    # Orientation
                    orientation_dict = instruction["kwargs"]["pose"].get("orientation", None)
                    if orientation_dict is None:
                        orientation_dict = {"value": [0.0, 0.0, 0.0]}
                    orientation_base = np.array(orientation_dict.get("value", [0.0, 0.0, 0.0]))

                    randomize = orientation_dict.get("random", None)
                    if randomize is None:
                        randomize_axis = None
                        randomize_angle = None
                    else:
                        randomize_axis = np.array(randomize.get("axis", [0.0, 0.0, 0.0]))
                        randomize_angle = np.array(randomize.get("angle", 0.0))

                    orientation = Rotation(
                        R.from_euler("xyz", orientation_base, degrees=True),
                        random_axis=randomize_axis,
                        random_max_angle=randomize_angle,
                    )

                    instruction["kwargs"][key] = Pose(position, orientation)
            instruction_list.append(instruction)

        assert instruction_list is not None

        return instruction_list

    @abstractmethod
    def reset_preprocess(self, instructions):
        raise NotImplementedError("Reset preprocess function is not implemented for this scene.")

    @abstractmethod
    def reset_postprocess(self, result):
        raise NotImplementedError("Reset postprocess function is not implemented for this scene.")

    @abstractmethod
    def randomize_preprocess(self, instructions):
        raise NotImplementedError(
            "Randomize preprocess function is not implemented for this scene."
        )

    @abstractmethod
    def randomize_postprocess(self, result):
        raise NotImplementedError(
            "Randomize postprocess function is not implemented for this scene."
        )

    @abstractmethod
    def is_success_preprocess(self, instructions):
        raise NotImplementedError("Success preprocess function is not implemented for this scene.")

    @abstractmethod
    def is_success_postprocess(self, result):
        raise NotImplementedError("Success postprocess function is not implemented for this scene.")

    @abstractmethod
    def check_warmup(self):
        raise NotImplementedError("Check warmup function is not implemented for this scene.")

    def create_render_products(self, rep):
        dataset_cfg = self._config.get("dataset", {})
        images_cfg = dataset_cfg.get("images", [])

        self.rgb_annotators = {}

        for img_item in images_cfg:
            for ds_key, cam_name in img_item.items():
                cam_cfg = self._config.get("cameras", {}).get(cam_name)
                if cam_cfg:
                    cam_path = cam_cfg["path"]
                    res = (cam_cfg["width"], cam_cfg["height"])
                    rp = rep.create.render_product(f"/Scene_{self._scene_id}{cam_path}", res)
                    annotator = rep.AnnotatorRegistry.get_annotator("rgb")
                    annotator.attach([rp])
                    self.rgb_annotators[ds_key] = annotator

        self.setup_dataset()

    def setup_dataset(self):
        try:
            from isaacsim.core.prims import SingleArticulation
        except ImportError:
            try:
                from omni.isaac.core.articulations import SingleArticulation
            except ImportError:
                from omni.isaac.core.articulations import (
                    ArticulationView as SingleArticulation,
                )

        self.robots_views = {}
        self.ee_views = {}
        self.obs_masks = {}  # robot_name -> attr -> { 'indices': [], 'keys': [] }
        self.act_masks = {}  # robot_name -> attr -> { 'indices': [], 'keys': [] }

        dataset_cfg = self._config.get("dataset", {})
        obs_cfg = dataset_cfg.get("observations", [])
        act_cfg = dataset_cfg.get("action", [])

        def parse_mapping(cfg_list):
            mapping = {}
            cartesian_keys = {"x", "y", "z", "wx", "wy", "wz"}
            for item in cfg_list:
                for ds_key, target in item.items():
                    if ds_key in cartesian_keys:
                        continue
                    parts = target.split(".")
                    if len(parts) >= 3:
                        r_name = parts[0]
                        attr = parts[-1]
                        j_name = ".".join(parts[1:-1])
                        mapping[ds_key] = (r_name, j_name, attr)
            return mapping

        obs_mapping = parse_mapping(obs_cfg)
        act_mapping = parse_mapping(act_cfg)

        required_robots = {r for r, _, _ in obs_mapping.values()}
        # Create views for robots strictly found in config using SingleArticulation
        for r_name in required_robots:
            if r_name in self._config.get("robots", {}):
                robot_cfg = self._config["robots"][r_name]
                robot_path = robot_cfg.get("path", f"/{r_name}")
                prim_path = f"/Scene_{self._scene_id}{robot_path}"
                try:
                    view = SingleArticulation(
                        prim_path=prim_path, name=f"{r_name}_view_{self._scene_id}"
                    )
                except TypeError:
                    view = SingleArticulation(
                        prim_paths_expr=prim_path, name=f"{r_name}_view_{self._scene_id}"
                    )
                if not view.handles_initialized:
                    view.initialize()
                self.robots_views[r_name] = view
                self._logger.info(f"Robot view '{r_name}': dof_names={view.dof_names}")

                # Setup end-effector view if configured
                ee_name = robot_cfg.get("end_effector_name")
                if ee_name:
                    import omni.usd
                    from omni.isaac.core.prims import XFormPrimView

                    stage = omni.usd.get_context().get_stage()

                    # Try direct path first
                    ee_path = f"{prim_path}/{ee_name}"
                    actual_ee_path = None

                    if stage.GetPrimAtPath(ee_path).IsValid():
                        actual_ee_path = ee_path
                    else:
                        # Fallback: search the stage for a prim with this name under the robot root
                        robot_prim = stage.GetPrimAtPath(prim_path)
                        if robot_prim.IsValid():
                            from pxr import Usd

                            for prim in Usd.PrimRange(robot_prim):
                                if prim.GetName() == ee_name:
                                    actual_ee_path = str(prim.GetPath())
                                    break

                    if actual_ee_path:
                        try:
                            ee_view = XFormPrimView(
                                prim_paths_expr=actual_ee_path,
                                name=f"{r_name}_ee_view_{self._scene_id}",
                            )

                            # Safely initialize without assuming handles_initialized exists
                            if hasattr(ee_view, "handles_initialized"):
                                if not ee_view.handles_initialized:
                                    ee_view.initialize()
                            else:
                                ee_view.initialize()

                            self.ee_views[r_name] = ee_view
                            self._logger.info(
                                f"Created XFormPrimView for end_effector: {actual_ee_path}"
                            )
                        except Exception as e:
                            self._logger.warning(
                                f"Failed to create ee_view for {actual_ee_path}: {e}"
                            )
                    else:
                        self._logger.warning(
                            f"Could not find any prim matching end_effector_name '{ee_name}' under '{prim_path}'"
                        )

        def find_dof_index(dof_names, j_name):
            """Find DOF index by exact match or suffix match (e.g. 'joint1' matches 'fr3_joint1')."""
            if j_name in dof_names:
                return dof_names.index(j_name)
            for idx, dof in enumerate(dof_names):
                if dof.endswith(j_name):
                    return idx
            return None

        def build_mask(mapping, mask_dict):
            for ds_key, (r_name, j_name, attr) in mapping.items():
                target_robot = r_name
                if target_robot not in self.robots_views:
                    for rv_name, rv in self.robots_views.items():
                        if find_dof_index(rv.dof_names, j_name) is not None:
                            target_robot = rv_name
                            break

                if target_robot in self.robots_views:
                    dof_names = self.robots_views[target_robot].dof_names
                    dof_idx = find_dof_index(dof_names, j_name)
                    if dof_idx is not None:
                        if target_robot not in mask_dict:
                            mask_dict[target_robot] = {}
                        if attr not in mask_dict[target_robot]:
                            mask_dict[target_robot][attr] = {"indices": [], "keys": []}

                        mask_dict[target_robot][attr]["indices"].append(dof_idx)
                        mask_dict[target_robot][attr]["keys"].append(ds_key)
                    else:
                        self._logger.warning(
                            f"Joint '{j_name}' not found in dof_names of robot '{target_robot}'. Available: {dof_names}"
                        )
                else:
                    self._logger.warning(
                        f"Robot '{target_robot}' not found in robots_views for ds_key='{ds_key}'"
                    )

        build_mask(obs_mapping, self.obs_masks)
        build_mask(act_mapping, self.act_masks)
        self._logger.info(f"obs_masks: {self.obs_masks}")
        self._logger.info(f"act_masks: {self.act_masks}")

    def record_step(self, current_step: int):
        observation = {"x": 0.0, "y": 0.0, "z": 0.0, "wx": 0.0, "wy": 0.0, "wz": 0.0}
        action = {"x": 0.0, "y": 0.0, "z": 0.0, "wx": 0.0, "wy": 0.0, "wz": 0.0}

        # ~~~~~~~~~~~~~~ Observations ~~~~~~~~~~~~~ #
        # Images
        if hasattr(self, "rgb_annotators"):
            for ds_key, annotator in self.rgb_annotators.items():
                data = annotator.get_data()
                if data is not None:
                    # Isaac Sim annotators return RGBA (4 channels), strip alpha for RGB
                    if data.ndim == 3 and data.shape[2] == 4:
                        data = data[:, :, :3]
                    observation[ds_key] = data

        # Joint states
        if hasattr(self, "obs_masks"):
            for r_name, attr_masks in self.obs_masks.items():
                view = self.robots_views[r_name]
                for attr, mask in attr_masks.items():
                    joint_indices = mask["indices"]
                    if attr == "pos":
                        vals = view.get_joint_positions(joint_indices=joint_indices)
                    elif attr == "vel":
                        vals = view.get_joint_velocities(joint_indices=joint_indices)
                    elif attr == "acc":
                        vals = view.get_joint_accelerations(joint_indices=joint_indices)
                    elif attr == "eff":
                        vals = view.get_joint_efforts(joint_indices=joint_indices)
                    else:
                        continue

                    if vals is not None:
                        if hasattr(vals, "ndim") and vals.ndim > 1:
                            vals = vals[0]
                        elif not hasattr(vals, "__iter__"):
                            vals = [vals]
                        for val, key in zip(vals, mask["keys"]):
                            observation[key] = float(val)

        # ~~~~~~~~~~~~~~~~ Actions ~~~~~~~~~~~~~~~~ #
        if hasattr(self, "act_masks"):
            for r_name, attr_masks in self.act_masks.items():
                view = self.robots_views[r_name]
                for attr, mask in attr_masks.items():
                    joint_indices = mask["indices"]

                    vals = None
                    if hasattr(view, "get_applied_action"):
                        action_obj = view.get_applied_action()
                        if action_obj is not None:
                            if attr == "pos":
                                vals = action_obj.joint_positions
                            elif attr == "vel":
                                vals = action_obj.joint_velocities
                            elif attr == "eff":
                                vals = action_obj.joint_efforts

                            if vals is not None:
                                vals = np.asarray(vals)
                                if joint_indices is not None and len(joint_indices) > 0:
                                    vals = vals[joint_indices]

                    if vals is not None:
                        if hasattr(vals, "ndim") and vals.ndim > 1:
                            vals = vals[0]
                        elif not hasattr(vals, "__iter__"):
                            vals = [vals]
                        for val, key in zip(vals, mask["keys"]):
                            action[key] = float(val)

        # Cartesian Pose Delta from End Effector View
        if hasattr(self, "ee_views") and self.ee_views:
            dataset_cfg = self._config.get("dataset", {})
            cartesian_robot = dataset_cfg.get("cartesian_velocity_robot")
            if cartesian_robot is None:
                cartesian_robot = list(self.ee_views.keys())[0]

            if cartesian_robot in self.ee_views:
                ee_view = self.ee_views[cartesian_robot]

                # Get world pose of the end effector directly
                curr_pos, curr_rot = ee_view.get_world_poses()

                if curr_pos is not None:
                    if curr_pos.ndim > 1:
                        curr_pos = curr_pos[0]
                        curr_rot = curr_rot[0]

                    # Isaac Sim quaternions are usually [w, x, y, z]
                    r_curr = R.from_quat([curr_rot[1], curr_rot[2], curr_rot[3], curr_rot[0]])
                    abs_rotvec = r_curr.as_rotvec()

                    # Calculate deltas based on last recorded data
                    if (
                        not hasattr(self, "_last_recorded_obs_pose")
                        or self._last_recorded_obs_pose is None
                    ):
                        # For the first frame, delta is zero
                        obs_delta_pos = np.zeros(3)
                        obs_delta_rotvec = np.zeros(3)
                    else:
                        last_curr_pos, last_curr_rot = self._last_recorded_obs_pose

                        # Observation Delta (current - last_current)
                        obs_delta_pos = curr_pos - last_curr_pos
                        r_last_curr = R.from_quat(
                            [last_curr_rot[1], last_curr_rot[2], last_curr_rot[3], last_curr_rot[0]]
                        )
                        obs_delta_r = r_curr * r_last_curr.inv()
                        obs_delta_rotvec = obs_delta_r.as_rotvec()

                    # Store current for the next step
                    self._last_recorded_obs_pose = (curr_pos.copy(), curr_rot.copy())

                    # Add absolute pose to observation
                    observation["x"] = float(curr_pos[0])
                    observation["y"] = float(curr_pos[1])
                    observation["z"] = float(curr_pos[2])

                    observation["wx"] = float(abs_rotvec[0])
                    observation["wy"] = float(abs_rotvec[1])
                    observation["wz"] = float(abs_rotvec[2])

                    # Without kinematics solver, we don't have the target pose.
                    # We copy the actual execution delta to the action.
                    action["x"] = float(obs_delta_pos[0])
                    action["y"] = float(obs_delta_pos[1])
                    action["z"] = float(obs_delta_pos[2])

                    action["wx"] = float(obs_delta_rotvec[0])
                    action["wy"] = float(obs_delta_rotvec[1])
                    action["wz"] = float(obs_delta_rotvec[2])

        return {
            "timestamp": current_step,
            "observation": observation,
            "action": action,
            "task": self.task,
        }

    def clear_recording_history(self):
        self._last_recorded_obs_pose = None

    @abstractmethod
    def reset_lightweight(self):
        raise NotImplementedError("Lightweight reset is not implemented for this scene.")

    def finalize(self):
        self.recorder.put_record_data("FINALIZE")
        self.recorder.set_start_recording()
