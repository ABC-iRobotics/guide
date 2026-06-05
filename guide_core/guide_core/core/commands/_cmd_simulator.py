from typing import Any, Tuple

import numpy as np

from guide_core.types.geometry import Point, Pose


def _execute_instruction_directly(self, instruction: dict) -> Any:
    cmd_name = instruction.get("cmd", None)
    args = instruction.get("args", [])
    kwargs = instruction.get("kwargs", {})
    if cmd_name == "sync":
        return None
    handler = getattr(self, f"_cmd_{cmd_name}")
    return handler(*args, **kwargs)


def _execute_instructions_directly(self, instructions: list) -> list:
    results = []
    for instruction in instructions:
        results.append(self._execute_instruction_directly(instruction))
    return results


def _cmd_randomize_scene(self, scene_id: int) -> bool:
    simulator = getattr(self, "_simulator", None)
    if simulator is None:
        raise RuntimeError("Simulator reference not set on IsaacSimRuntime!")
    instructions = simulator._scene_manager.randomize_preprocess(scene_id)
    results = self._execute_instructions_directly(instructions)
    return simulator._scene_manager.randomize_postprocess(scene_id, results)


def _cmd_reset_scene(self, scene_id: int) -> bool:
    simulator = getattr(self, "_simulator", None)
    if simulator is None:
        raise RuntimeError("Simulator reference not set on IsaacSimRuntime!")
    instructions = simulator._scene_manager.reset_preprocess(scene_id)
    results = self._execute_instructions_directly(instructions)
    return simulator._scene_manager.reset_postprocess(scene_id, results)


def _cmd_is_success(self, scene_id: int) -> bool:
    simulator = getattr(self, "_simulator", None)
    if simulator is None:
        raise RuntimeError("Simulator reference not set on IsaacSimRuntime!")
    instructions = simulator._scene_manager.is_success_preprocess(scene_id)
    results = self._execute_instructions_directly(instructions)
    return simulator._scene_manager.is_success_postprocess(scene_id, results)


def _cmd_register_scene(self, package_name: str) -> Tuple[int, Tuple[float, float, float]]:
    simulator = getattr(self, "_simulator", None)
    if simulator is None:
        raise RuntimeError("Simulator reference not set on IsaacSimRuntime!")

    id, offset, config = simulator._add_scene_to_manager(package_name)
    if id == -1:
        raise RuntimeError(
            f"SceneManager failed to add scene:\n{config.get('error', 'Unknown error')}"
        )

    config.update({"package": package_name})

    usd_path = simulator._scene_manager.get_scene_usd_path(id)
    if usd_path is not None:
        config["usd_path_absolute"] = usd_path

    # Directly execute add_scene and set_world_poses instead of calling self.call
    scene_path = f"/Scene_{id}"
    pose: Pose = Pose(position=Point(offset))

    self._cmd_add_scene(stage_config=config, root=scene_path)
    self._cmd_set_world_poses(prim_path=scene_path, pose=pose)

    robots = simulator._scene_manager.get_scene_robot_graphs(id)
    cameras = simulator._scene_manager.get_scene_camera_graphs(id)

    if robots is not None:
        for robot in robots:
            self._cmd_create_robot_control(
                namespace=f'{simulator._sim_path}{scene_path}{robot.get("namespace", "/robot")}',
                articulation_root=f'{scene_path}{robot.get("articulation_root", "/robot")}',
                path=f'{scene_path}/Graph{robot.get("path", "/robot")}_control_graph',
                default_joint_states=np.array(robot.get("default_joint_states", [])),
            )
            self._cmd_create_tf_graph(
                namespace=f'{simulator._sim_path}{scene_path}{robot.get("namespace", "/robot")}',
                prim=f"{scene_path}",
                parent_prim="/",
                path=f'{scene_path}/Graph{robot.get("path", "/robot")}_tf_graph_world',
            )
            self._cmd_create_tf_graph(
                namespace=f'{simulator._sim_path}{scene_path}{robot.get("namespace", "/robot")}',
                prim=f'{scene_path}{robot.get("articulation_root", "/robot")}',
                parent_prim=f'{scene_path}{robot.get("articulation_root", "/robot")}'.rsplit(
                    "/", 1
                )[0],
                path=f'{scene_path}/Graph{robot.get("path", "/robot")}_tf_graph',
            )

    if cameras is not None:
        for camera in cameras:
            self._cmd_create_camera(
                camera_path=f'{scene_path}{camera.get("camera_path", "/cam")}',
                path=f'{scene_path}/Graph{camera.get("path", "/cam")}_camera_graph',
                width=camera.get("width", 640),
                height=camera.get("height", 480),
                frame=camera.get("frame", "cam"),
                namespace=f"{simulator._sim_path}{scene_path}",
                topic=camera.get("topic", "/rgb"),
            )
            self._cmd_create_tf_graph(
                namespace=f"{simulator._sim_path}{scene_path}",
                prim=f'{scene_path}{camera.get("camera_path", "/cam")}',
                parent_prim=f"{scene_path}",
                path=f'{scene_path}/Graph{camera.get("path", "/cam")}_tf_graph',
            )

    return id, offset
