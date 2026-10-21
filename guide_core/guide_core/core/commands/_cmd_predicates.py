"""Success predicates over the live scene: the commands config/success.yaml names (guide_core.scene.predicates).

Each reads the poses or the joint value it needs and returns {success, reason}; RecipeScene ANDs them.
Prim paths arrive scene-prefixed (SceneOrchestrator.parse_instruction).
"""
import numpy as np

from guide_core.scene import predicates

TOUCH_TOL = 0.005  # m: two meshes this close touch (PhysX rests bodies a contact offset apart)


def __pose(self, path: str):
    """(position, quat w, x, y, z, scale) of ``path`` in the world, as XFormPrim reads it (physics included)."""
    from isaacsim.core.prims import XFormPrim

    if not self._stage.GetPrimAtPath(path).IsValid():
        raise ValueError(f"success predicate: no prim {path}")
    view = XFormPrim(prim_paths_expr=path, reset_xform_properties=False)
    position, orientation = view.get_world_poses()
    return np.asarray(position)[0], np.asarray(orientation)[0], np.asarray(view.get_world_scales())[0]


def __region_bound(self, region: str):
    """(position, quat w, x, y, z, scale, local min, local max) of ``region``'s own bound: the box it stands for
    (visible or not)."""
    from pxr import Usd, UsdGeom

    position, orientation, scale = self.__pose(region)
    purposes = [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy, UsdGeom.Tokens.guide]
    box = UsdGeom.BBoxCache(Usd.TimeCode.Default(), purposes, False, True).ComputeUntransformedBound(
        self._stage.GetPrimAtPath(region)).ComputeAlignedRange()
    if box.IsEmpty():
        raise ValueError(f"success predicate: region {region} has no extent")
    return position, orientation, scale, box.GetMin(), box.GetMax()


def __region_box(self, region: str):
    """World (centre, half-extent vector) of ``region``'s own bound."""
    position, orientation, scale, lo, hi = self.__region_bound(region)
    return predicates.obb_world(lo, hi, position, orientation, scale)


def __joint_position(self, joint: str) -> float:
    """``joint``'s position in SI units (m, rad), from its articulation."""
    from isaacsim.core.prims import SingleArticulation

    from guide_core.types.randomization import fixtures

    prim = self._stage.GetPrimAtPath(joint)
    root = fixtures.articulation_root(prim) if prim.IsValid() else None
    if root is None:
        raise ValueError(f"success predicate: {joint} is not a joint of an articulation")
    view = SingleArticulation(prim_path=str(root.GetPath()), name="success_predicate_view")
    view.initialize()
    names = list(view.dof_names)
    if prim.GetName() not in names:
        raise ValueError(f"success predicate: {joint} is not one of {root.GetPath()}'s DOFs {names}")
    return float(np.ravel(view.get_joint_positions())[names.index(prim.GetName())])


def __touching(self, a: str, b: str):
    """Whether ``a`` and ``b`` (prim paths, subtrees) touch: a mesh clash within TOUCH_TOL, LIBERO's check_contact
    (any geom contact). Needs the clash detector (Isaac Sim 6.0 ships it); the box fallback would call any two
    boxes that overlap touching, so there is none."""
    return lambda: self._cmd_is_prim_clashing(a, b, TOUCH_TOL, mesh_only=True)


def _cmd_on(self, a: str, b: str, xy_tol: float = predicates.XY_TOL, contact: bool = False) -> dict:
    return predicates.on(self.__pose(a)[0], self.__pose(b)[0], xy_tol, f"on({a}, {b})",
                         self.__touching(a, b) if contact else None)


def _cmd_on_region(self, obj: str, region: str, touch: str | None = None) -> dict:
    """``obj`` on ``region`` (a site prim, or a table zone's hidden box; ``touch``: the site's parent object, which
    ``obj`` must touch)."""
    position, orientation, scale, lo, hi = self.__region_bound(region)
    centre, half = predicates.box_world(lo, hi, position, orientation, scale)
    return predicates.on_region(self.__pose(obj)[0], centre, orientation, half, f"on_region({obj}, {region})",
                                self.__touching(obj, touch) if touch else None)


def _cmd_in_region(self, obj: str, region: str) -> dict:
    return predicates.in_box(self.__pose(obj)[0], *self.__region_box(region), label=f"in_region({obj}, {region})")


def _cmd_joint_above(self, joint: str, value: float, inclusive: bool = False) -> dict:
    return predicates.joint_above(self.__joint_position(joint), value, joint, inclusive)


def _cmd_joint_below(self, joint: str, value: float) -> dict:
    return predicates.joint_below(self.__joint_position(joint), value, joint)
