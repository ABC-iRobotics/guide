import isaacsim.core.utils.prims as prims_utils
import numpy as np
from isaacsim.util.clash_detection import ClashDetector
from omni.physx import get_physx_interface

try:
    import isaacsim.core.utils.bounds as bounds_utils
except ImportError:
    try:
        import omni.isaac.core.utils.bounds as bounds_utils
    except ImportError:
        bounds_utils = None

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


def __init_clash_detector(self, tolerance: float = 0.0):

    assert self.state not in [UNINITIALIZED, INITIALIZING, ERROR, SHUTTING_DOWN]

    self._cd = ClashDetector(
        self._stage, tolerance=tolerance, logging=self._debug, clash_data_layer=False
    )


def _cmd_get_scope(self) -> str:
    if getattr(self, "_cd", None) is None:
        self.__init_clash_detector()

    return self._cd.get_scope()


def _cmd_set_scope(self, scope: str) -> None:
    if getattr(self, "_cd", None) is None:
        self.__init_clash_detector()

    self._cd.set_scope(scope)


def _cmd_check_bounding_box_collision(
    self, prim_path: str, target_scope: str, tol: float = 0.01, check_containment: bool = False
) -> bool:
    try:
        get_physx_interface().update_transformations(False, True)
    except Exception as e:
        self._logger.debug(f"[CLASH_DEBUG] Failed to update_transformations: {e}")

    if bounds_utils is None or not target_scope:
        return False

    bbox_cache = bounds_utils.create_bbox_cache()

    try:
        centroid_target, axes_target, half_extent_target = bounds_utils.compute_obb(
            bbox_cache, prim_path
        )
        centroid_scope, axes_scope, half_extent_scope = bounds_utils.compute_obb(
            bbox_cache, target_scope
        )

        self._logger.debug(
            f"[CLASH_DEBUG]   Target OBB: Centroid {centroid_target}, Axes {axes_target}, Extents {half_extent_target}"
        )
        self._logger.debug(
            f"[CLASH_DEBUG]   Scope OBB: Centroid {centroid_scope}, Axes {axes_scope}, Extents {half_extent_scope}"
        )

        if centroid_target is not None and centroid_scope is not None:
            A = np.array(axes_target, dtype=float)
            norms_A = np.linalg.norm(A, axis=1)
            norms_A[norms_A == 0] = 1.0
            A = A / norms_A[:, np.newaxis]

            B = np.array(axes_scope, dtype=float)
            norms_B = np.linalg.norm(B, axis=1)
            norms_B[norms_B == 0] = 1.0
            B = B / norms_B[:, np.newaxis]

            a = np.array(half_extent_target, dtype=float)
            b = np.array(half_extent_scope, dtype=float)

            T = np.array(centroid_scope, dtype=float) - np.array(centroid_target, dtype=float)

            if check_containment:
                # For target A to be fully contained in scope B:
                # The projection of A's radius plus distance between centers on each of B's local axes
                # must be <= B's half-extent (plus tolerance).
                contained = True
                for i in range(3):
                    L = B[i]
                    dist = abs(np.dot(T, L))
                    rA = (
                        a[0] * abs(np.dot(A[0], L))
                        + a[1] * abs(np.dot(A[1], L))
                        + a[2] * abs(np.dot(A[2], L))
                    )

                    if dist + rA > b[i] + tol:
                        contained = False
                        break

                self._logger.debug(
                    f"[CLASH_DEBUG]   OBB Containment check (tolerance={tol}) result: {contained}"
                )
                return contained

            else:
                overlap = True

                # 15 Separating Axes
                # 3 from A
                for i in range(3):
                    L = A[i]
                    rA = a[i]
                    rB = (
                        b[0] * abs(np.dot(B[0], L))
                        + b[1] * abs(np.dot(B[1], L))
                        + b[2] * abs(np.dot(B[2], L))
                    )
                    if abs(np.dot(T, L)) > rA + rB + tol:
                        overlap = False
                        break

                # 3 from B
                if overlap:
                    for i in range(3):
                        L = B[i]
                        rA = (
                            a[0] * abs(np.dot(A[0], L))
                            + a[1] * abs(np.dot(A[1], L))
                            + a[2] * abs(np.dot(A[2], L))
                        )
                        rB = b[i]
                        if abs(np.dot(T, L)) > rA + rB + tol:
                            overlap = False
                            break

                # 9 cross products
                if overlap:
                    for i in range(3):
                        if not overlap:
                            break
                        for j in range(3):
                            L = np.cross(A[i], B[j])
                            mag = np.linalg.norm(L)
                            if mag < 1e-5:
                                continue
                            L = L / mag
                            rA = (
                                a[0] * abs(np.dot(A[0], L))
                                + a[1] * abs(np.dot(A[1], L))
                                + a[2] * abs(np.dot(A[2], L))
                            )
                            rB = (
                                b[0] * abs(np.dot(B[0], L))
                                + b[1] * abs(np.dot(B[1], L))
                                + b[2] * abs(np.dot(B[2], L))
                            )
                            if abs(np.dot(T, L)) > rA + rB + tol:
                                overlap = False
                                break

                self._logger.debug(
                    f"[CLASH_DEBUG]   OBB SAT Overlap check (tolerance={tol}) result: {overlap}"
                )
                return overlap

    except Exception as e:
        self._logger.debug(
            f"[CLASH_DEBUG]   Error computing OBB or SAT: {e}. Falling back to AABB..."
        )

        # Fallback to AABB
        aabb_target = bounds_utils.compute_aabb(bbox_cache, prim_path, include_children=True)
        aabb_scope = bounds_utils.compute_aabb(bbox_cache, target_scope, include_children=True)

        self._logger.debug(f"[CLASH_DEBUG]   Target AABB: {aabb_target}")
        self._logger.debug(f"[CLASH_DEBUG]   Scope AABB: {aabb_scope}")

        if aabb_target is not None and aabb_scope is not None:
            if check_containment:
                contained = True
                for i in range(3):
                    min_a, max_a = aabb_target[i], aabb_target[i + 3]
                    min_b, max_b = aabb_scope[i], aabb_scope[i + 3]
                    # Target A must be completely inside Scope B
                    if min_a < min_b - tol or max_a > max_b + tol:
                        contained = False
                        break
                self._logger.debug(
                    f"[CLASH_DEBUG]   AABB Containment check (tolerance={tol}) result: {contained}"
                )
                return contained
            else:
                overlap = True
                for i in range(3):
                    min_a, max_a = aabb_target[i], aabb_target[i + 3]
                    min_b, max_b = aabb_scope[i], aabb_scope[i + 3]
                    if max_a < min_b - tol or min_a > max_b + tol:
                        overlap = False
                        break

                self._logger.debug(
                    f"[CLASH_DEBUG]   AABB Overlap check (tolerance={tol}) result: {overlap}"
                )
                return overlap

    return False


def _cmd_is_prim_clashing(
    self, prim_path: str, scope: str | None = None, tolerance: float | None = None
) -> bool:
    # Force PhysX simulated transforms to write back to USD stage so ClashDetector sees current poses
    try:
        get_physx_interface().update_transformations(False, True)
    except Exception as e:
        self._logger.debug(f"[CLASH_DEBUG] Failed to update_transformations: {e}")

    tol = tolerance if tolerance is not None else 0.0
    if getattr(self, "_cd", None) is None or tolerance is not None:
        self.__init_clash_detector(tolerance=tol)

    assert self.state in [RUNNING]

    if isinstance(scope, str):
        self._cmd_set_scope(scope)

    prim = prims_utils.get_prim_at_path(prim_path)

    # Detailed clash query logging
    self._logger.debug(
        f"[CLASH_DEBUG] Querying clash for prim: {prim_path} (Type: {prim.GetTypeName() if prim else 'None'}, Tolerance: {tol})"
    )
    if prim:
        # Log children
        children = [
            f"{child.GetPath().pathString} ({child.GetTypeName()})" for child in prim.GetChildren()
        ]
        self._logger.debug(f"[CLASH_DEBUG]   Children of {prim_path}: {children}")
        # Log translation attribute if exists
        try:
            translate = prim.GetAttribute("xformOp:translate").Get()
            self._logger.debug(f"[CLASH_DEBUG]   USD xformOp:translate: {translate}")
        except Exception as e:
            self._logger.debug(f"[CLASH_DEBUG]   No xformOp:translate attribute: {e}")

    res = self._cd.is_prim_clashing(prim)
    self._logger.debug(f"[CLASH_DEBUG]   is_prim_clashing returned: {res}")

    # Fallback to Bounding Box (OBB/AABB) overlap check within tolerance if mesh clash returned False
    if not res:
        target_scope = scope if scope is not None else getattr(self, "_scope", None)
        if target_scope is None:
            try:
                target_scope = self._cd.get_scope()
            except Exception:
                target_scope = ""

        res = self._check_bounding_box_collision(
            prim_path, target_scope, tol, check_containment=False
        )

    return res


def _cmd_is_prim_contained(
    self, prim_path: str, scope: str | None = None, tolerance: float | None = None
) -> bool:
    # Force PhysX simulated transforms to write back to USD stage to get current poses
    try:
        get_physx_interface().update_transformations(False, True)
    except Exception as e:
        self._logger.debug(f"[CLASH_DEBUG] Failed to update_transformations: {e}")

    tol = tolerance if tolerance is not None else 0.0

    assert self.state in [RUNNING]

    target_scope = scope if scope is not None else getattr(self, "_scope", None)
    if target_scope is None:
        try:
            target_scope = self._cd.get_scope()
        except Exception:
            target_scope = ""

    # Detailed clash query logging
    self._logger.debug(
        f"[CLASH_DEBUG] Querying containment for prim: {prim_path} in scope: {target_scope} (Tolerance: {tol})"
    )

    # Only rely on bounding box (OBB/AABB) for complete containment check
    return self._cmd_check_bounding_box_collision(
        prim_path, target_scope, tol, check_containment=True
    )
