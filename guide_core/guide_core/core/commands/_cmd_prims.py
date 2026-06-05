import logging
from typing import Any

import isaacsim.core.utils.prims as prims_utils
import numpy as np
from isaacsim.core.prims import RigidPrim, XFormPrim
from scipy.spatial.transform import Rotation as R

from guide_core.types.geometry import Point, Pose, Rotation
from guide_core.types.isaac_state import IsaacState

UNINITIALIZED = IsaacState.UNINITIALIZED
INITIALIZING = IsaacState.INITIALIZING
STOPPED = IsaacState.STOPPED
LOADING = IsaacState.LOADING
READY = IsaacState.READY
RUNNING = IsaacState.RUNNING
PAUSED = IsaacState.PAUSED
ERROR = IsaacState.ERROR
SHUTTING_DOWN = IsaacState.SHUTTING_DOWN


class PrimDoesNotExistError(Exception):
    pass


def __check_valid_prim(self, prim_path: str) -> None:
    if not prims_utils.is_prim_path_valid(prim_path):
        raise PrimDoesNotExistError("Invalid prim path.")


def _cmd_create_prim(
    self,
    prim_path: str,
    prim_type: str = "Xform",
    position: np.ndarray | None = None,
    translation: np.ndarray | None = None,
    orientation: np.ndarray | None = None,
    scale: np.ndarray | None = None,
    usd_path: str | None = None,
    semantic_label: str | None = None,
    semantic_type: str = "class",
    attributes: dict | None = None,
) -> None:
    if self._stage is None:
        logging.error("No stage is currently open. Cannot create prim.")
        return

    if position is not None and translation is not None:
        logging.error("Position and translation cannot be defined simultaneously.")
        return

    try:
        new_prim = prims_utils.create_prim(
            prim_path,
            prim_type,
            position,
            translation,
            orientation,
            scale,
            usd_path,
            semantic_label,
            semantic_type,
            attributes,
        )
        if not new_prim:
            logging.error(f"Failed to create prim at {prim_path} of type {prim_type}.")
        else:
            logging.info(f"Successfully created prim at {prim_path} of type {prim_type}.")
    except Exception as e:
        logging.error(f"Error creating prim at {prim_path} of type {prim_type}: {e}")


def _cmd_delete_prim(self, prim_path: str) -> None:
    if not prims_utils.is_prim_path_valid(prim_path):
        logging.error("Invalid prim path. Cannot delete prim.")
        return PrimDoesNotExistError("Invalid prim path. Cannot delete prim.")

    try:
        prims_utils.delete_prim(prim_path)
        logging.info(f"Successfully deleted prim at {prim_path}.")
    except Exception as e:
        logging.error(f"Error deleting prim at {prim_path}: {e}")


def _cmd_get_prim_attribute_names(self, prim_path: str) -> Any:
    try:
        self.__check_valid_prim(prim_path)
    except Exception as e:
        return e

    try:
        value = prims_utils.get_prim_attribute_names(prim_path)
        return value
    except Exception:
        return


def _cmd_get_prim_attribute_value(self, prim_path: str, attribute_name: str) -> Any:
    try:
        self.__check_valid_prim(prim_path)
    except Exception as e:
        return e

    try:
        value = prims_utils.get_prim_attribute_value(prim_path, attribute_name)
        logging.info(f"Successfully got {attribute_name} for prim {prim_path}.")
        return value
    except Exception:
        return Exception(f"Error getting{attribute_name} for prim {prim_path}.")


def _cmd_set_prim_attribute_value(self, prim_path: str, attribute_name: str, value: Any) -> None:
    try:
        self.__check_valid_prim(prim_path)
    except Exception as e:
        return e

    try:
        prims_utils.set_prim_attribute_value(prim_path, attribute_name, value)
    except Exception as e:
        logging.error(f"Error setting property '{attribute_name}' of prim at {prim_path}: {e}")
        return Exception(f"Error setting property '{attribute_name}' of prim at {prim_path}: {e}")


def _cmd_set_prim_visibility(self, prim_path: str, visibility: bool) -> bool:
    try:
        self.__check_valid_prim(prim_path)
    except Exception as e:
        return e

    prim = prims_utils.get_prim_at_path(prim_path)

    try:
        prims_utils.set_prim_visibility(prim, visibility)
    except Exception as e:
        logging.error(f"Error setting visibility of prim at {prim_path}: {e}")
        return Exception(f"Error setting visibility of prim at {prim_path}: {e}")


# XFormPrim


def __size_to_num_of_prims(
    self, prims: XFormPrim, singleton: list | np.ndarray
) -> list | np.ndarray:
    if (
        isinstance(singleton, int)
        or isinstance(singleton, float)
        or isinstance(singleton, bool)
        or (isinstance(singleton, list) and len(singleton) == 1)
        or (isinstance(singleton, np.ndarray) and singleton.shape[0] == 1)
    ) and prims.count > 1:
        # singleton = np.vstack([singleton] * prims.count)
        if isinstance(singleton, list) or isinstance(singleton, np.ndarray):
            singleton = singleton[0]
        singleton = [singleton for _ in range(prims.count)]

    return singleton


def __get_xform(self, prim_path: str | list[str]):
    try:
        # self.__pre_initialize_xform_properties(prim_path)
        return XFormPrim(prim_paths_expr=prim_path, reset_xform_properties=True)
    except BaseException as e:
        self._logger.error(f"Error in __get_xform: {e}")
        raise PrimDoesNotExistError("Invalid prim path.")


def _cmd_get_visibilities(self, prim_path: str | list[str]) -> list[bool]:
    try:
        prims = self.__get_xform(prim_path)
    except Exception as e:
        self._logger.error(e)
        return

    return prims.get_visibilities()


def _cmd_set_visibilities(
    self, prim_path: str | list[str], visibilities: bool | list[bool]
) -> None:
    try:
        prims = self.__get_xform(prim_path)
    except Exception as e:
        self._logger.error(e)
        return

    if isinstance(visibilities, bool) and prims.count() > 1:
        visibilities = [visibilities] * prims.count()

    prims.set_visibilities(visibilities)


def _cmd_get_world_poses(self, prim_path: str | list[str]) -> Pose | list[Pose]:
    try:
        prims = self.__get_xform(prim_path)
    except Exception as e:
        self._logger.error(e)
        return

    positions: np.ndarray
    orientations: np.ndarray
    positions, orientations = prims.get_world_poses()

    if positions.shape[0] == 1:
        # _cmd_create_prim(self, prim_path='/' + prims.prim_paths[0].split('/')[-1]+'_marker', prim_type="Xform", position=positions[0], orientation=orientations[0])
        return Pose(Point(positions), Rotation(R.from_quat(np.roll(orientations, -1))))
    return [
        Pose(Point(p), Rotation(R.from_quat(np.roll(o, -1))))
        for p, o in zip(positions, orientations)
    ]


def _cmd_set_world_poses(self, prim_path: str | list[str], pose: Pose | list[Pose]) -> None:
    try:
        prims = self.__get_xform(prim_path)
    except Exception as e:
        self._logger.error(e)
        return

    if isinstance(pose, Pose):
        pose = [pose]

    pose = self.__size_to_num_of_prims(prims, pose)

    positions = np.asarray([p.position.to_numpy() for p in pose])
    orientations = np.vstack([p.orientation.to_numpy_quat() for p in pose])

    prims.set_world_poses(positions, orientations)


def _cmd_get_local_poses(self, prim_path: str | list[str]) -> Pose | list[Pose]:
    try:
        prims = self.__get_xform(prim_path)
    except Exception as e:
        self._logger.error(e)
        return

    positions: np.ndarray
    orientations: np.ndarray
    positions, orientations = prims.get_local_poses()

    if positions.shape[0] == 1:
        return Pose(Point(positions), Rotation(R.from_quat(np.roll(orientations, -1))))
    return [
        Pose(Point(p), Rotation(R.from_quat(np.roll(o, -1))))
        for p, o in zip(positions, orientations)
    ]


def _cmd_set_local_poses(self, prim_path: str | list[str], pose: Pose | list[Pose]) -> None:
    try:
        prims = self.__get_xform(prim_path)
    except Exception as e:
        self._logger.error(e)
        return

    if isinstance(pose, Pose):
        pose = [pose]

    import numpy as np

    pose = self.__size_to_num_of_prims(prims, pose)
    positions = np.asarray([p.position.to_numpy() for p in pose])
    orientations = np.vstack([p.orientation.to_numpy_quat() for p in pose])

    prims.set_local_poses(positions, orientations)


def _cmd_get_world_scales(self, prim_path: str | list[str]) -> np.ndarray:
    try:
        prims = self.__get_xform(prim_path)
    except Exception as e:
        self._logger.error(e)
        return

    return prims.get_world_scales()


def _cmd_get_local_scales(self, prim_path: str | list[str]) -> np.ndarray:
    try:
        prims = self.__get_xform(prim_path)
    except Exception as e:
        self._logger.error(e)
        return

    return prims.get_local_scales()


def _cmd_set_local_scales(self, prim_path: str | list[str], scales: list | np.ndarray) -> None:
    try:
        prims = self.__get_xform(prim_path)
    except Exception as e:
        self._logger.error(e)
        return

    scales = self.__size_to_num_of_prims(prims, scales)

    prims.set_local_scales(scales)


# RigidPrim


def __get_rigid(self, prim_path: str | list[str]):
    try:
        # self.__pre_initialize_xform_properties(prim_path)
        return RigidPrim(prim_paths_expr=prim_path, usd=False)
    except BaseException:
        raise PrimDoesNotExistError("Invalid prim path.")


def _cmd_get_linear_velocities(self, prim_path: str | list[str]) -> np.ndarray:
    try:
        prims = self.__get_rigid(prim_path)
    except Exception as e:
        self._logger.error(e)
        return

    return prims.get_linear_velocities()


def _cmd_set_linear_velocities(
    self, prim_path: str | list[str], velocities: list | np.ndarray
) -> None:
    try:
        prims = self.__get_rigid(prim_path)
    except Exception as e:
        self._logger.error(e)
        return

    velocities = self.__size_to_num_of_prims(prims, velocities)

    prims.set_linear_velocities(velocities)


def _cmd_get_angular_velocities(self, prim_path: str | list[str]) -> np.ndarray:
    try:
        prims = self.__get_rigid(prim_path)
    except Exception as e:
        self._logger.error(e)
        return

    return prims.get_angular_velocities()


def _cmd_set_angular_velocities(
    self, prim_path: str | list[str], velocities: list | np.ndarray
) -> None:
    try:
        prims = self.__get_rigid(prim_path)
    except Exception as e:
        self._logger.error(e)
        return

    velocities = self.__size_to_num_of_prims(prims, velocities)

    prims.set_angular_velocities(velocities)


def _cmd_apply_force(self, prim_path: str | list[str], forces: np.ndarray, is_global: bool) -> None:

    try:
        prims = self.__get_rigid(prim_path)
    except Exception as e:
        self._logger.error(e)
        return

    forces = self.__size_to_num_of_prims(prims, forces)

    prims.apply_forces(forces, is_global=is_global)


def _cmd_get_masses(self, prim_path: str | list[str]) -> float | np.ndarray:
    try:
        prims = self.__get_rigid(prim_path)
    except Exception as e:
        self._logger.error(e)
        return

    return prims.get_masses()


def _cmd_set_masses(self, prim_path: str | list[str], masses: float | np.ndarray) -> None:
    try:
        prims = self.__get_rigid(prim_path)
    except Exception as e:
        self._logger.error(e)
        return

    masses = self.__size_to_num_of_prims(prims, masses)

    return prims.set_masses(masses)


def _cmd_enable_rigid_body_physics(self, prim_path: str | list[str]) -> None:
    try:
        prims = self.__get_rigid(prim_path)
    except Exception as e:
        self._logger.error(e)
        return

    prims.enable_rigid_body_physics()


def _cmd_disable_rigid_body_physics(self, prim_path: str | list[str]) -> None:
    try:
        prims = self.__get_rigid(prim_path)
    except Exception as e:
        self._logger.error(e)
        return

    prims.disable_rigid_body_physics()


def _cmd_enable_gravities(self, prim_path: str | list[str]) -> None:
    try:
        prims = self.__get_rigid(prim_path)
    except Exception as e:
        self._logger.error(e)
        return

    prims.enable_gravities()


def _cmd_disable_gravities(self, prim_path: str | list[str]) -> None:
    try:
        prims = self.__get_rigid(prim_path)
    except Exception as e:
        self._logger.error(e)
        return

    prims.disable_gravities()
