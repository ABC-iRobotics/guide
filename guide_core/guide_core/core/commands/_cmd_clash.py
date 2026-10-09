import isaacsim.core.utils.prims as prims_utils
from omni.physx import get_physx_interface

# Mesh-level clash detection comes from the `isaacsim.util.clash_detection` extension,
# which IsaacSimRuntime._import_isaac_extensions enables before the commands are
# imported. Isaac Sim 5.x had dropped it; 6.0 ships it again. Without it ClashDetector
# is None and only the bounding-box (OBB/AABB) overlap check below is used.
try:
    from isaacsim.util.clash_detection import ClashDetector
except ImportError:
    ClashDetector = None

# Optional: only present when the Isaac Sim core extension is loaded.
try:
    import isaacsim.core.utils.bounds as bounds_utils
except ImportError:
    bounds_utils = None

from guide_core.types.bounding import AABB, OBB, BoundingVolumeOps
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

    if ClashDetector is None:
        # No mesh-level clash detector (extension not loaded): keep _cd None
        # and rely on the bounding-box fallback. Scope is tracked on self._scope.
        self._cd = None
        return

    self._cd = ClashDetector(
        self._stage, tolerance=tolerance, logging=self._debug, clash_data_layer=False
    )


def _cmd_get_scope(self) -> str:
    if getattr(self, "_cd", None) is None:
        self.__init_clash_detector()

    if self._cd is None:
        return getattr(self, "_scope", "") or ""

    return self._cd.get_scope()


def _cmd_set_scope(self, scope: str) -> None:
    if getattr(self, "_cd", None) is None:
        self.__init_clash_detector()

    # Track scope locally so the bounding-box fallback works without ClashDetector.
    self._scope = scope

    if self._cd is not None:
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

    # Build bounding volumes for the target prim and the scope, preferring
    # oriented boxes (OBB) and falling back to axis-aligned boxes (AABB) when the
    # OBB computation is unavailable. Overlap/containment are delegated to FCL
    # (via BoundingVolumeOps) instead of a hand-rolled separating-axis test.
    try:
        target = OBB.from_isaac(*bounds_utils.compute_obb(bbox_cache, prim_path))
        scope = OBB.from_isaac(*bounds_utils.compute_obb(bbox_cache, target_scope))
        kind = "OBB"
    except Exception as e:
        self._logger.debug(f"[CLASH_DEBUG]   Error computing OBB: {e}. Falling back to AABB...")
        try:
            target = AABB.from_isaac(
                bounds_utils.compute_aabb(bbox_cache, prim_path, include_children=True)
            )
            scope = AABB.from_isaac(
                bounds_utils.compute_aabb(bbox_cache, target_scope, include_children=True)
            )
            kind = "AABB"
        except Exception as e2:
            self._logger.debug(f"[CLASH_DEBUG]   Error computing AABB: {e2}.")
            return False

    ops = BoundingVolumeOps(tolerance=tol)
    if check_containment:
        # Is the target fully contained within the scope?
        result = ops.contains(scope, target)
        self._logger.debug(
            f"[CLASH_DEBUG]   {kind} containment check (tolerance={tol}) result: {result}"
        )
    else:
        result = ops.intersects(target, scope)
        self._logger.debug(
            f"[CLASH_DEBUG]   {kind} overlap check (tolerance={tol}) result: {result}"
        )
    return result


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

    if self._cd is not None:
        res = self._cd.is_prim_clashing(prim)
        self._logger.debug(f"[CLASH_DEBUG]   is_prim_clashing returned: {res}")
    else:
        # No mesh-level detector (extension not loaded): defer entirely to the
        # bounding-box overlap check below.
        res = False

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
