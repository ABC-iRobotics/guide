"""Success predicates with LIBERO's thresholds (S3 v2 spec 3.4; LIBERO's base_predicates.py and
base_object_states.py, quoted in ~/libero_assets/s3_inputs/benchmarks/LIBERO.md section 6).

Pure NumPy over body ORIGINS, as LIBERO tests them; core/commands/_cmd_predicates.py reads the live
poses and joint values. Each returns {"success": bool, "reason": str}. Joint values are SI (m, rad).
"""
import numpy as np
from scipy.spatial.transform import Rotation

XY_TOL = 0.03  # On(obj, obj): the origins closer than this in xy (strict)
IN_Z_MARGIN = 0.01  # In(obj, region): the box's lower z face drops by this much

#: The success instructions and their prim-path kwargs, which GUIDE scene-prefixes like prim_path.
PREDICATE_PATHS = {"on": ("a", "b"), "in_region": ("obj", "region"), "joint_above": ("joint",), "joint_below": ("joint",)}


def on(a, b, xy_tol: float = XY_TOL, label: str = "on") -> dict:
    """``a`` rests on ``b`` (LIBERO's ObjectState.check_ontop): b's origin no higher than a's, origins
    within ``xy_tol`` in xy (strict).
    ponytail: no contact test (LIBERO also asks that the two touch); S4 adds it with a contact report."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    xy, dz = float(np.hypot(*(a[:2] - b[:2]))), float(a[2] - b[2])
    return {"success": bool(dz >= 0.0 and xy < xy_tol),
            "reason": f"{label}: xy {xy:.4f} m (< {xy_tol}), a {dz:+.4f} m above b (>= 0)"}


def obb_world(local_min, local_max, position, quat_wxyz, scale):
    """World (centre, half-extent vector R @ (scale * half)) of a prim's own box [local_min, local_max]
    under its world pose and scale: LIBERO's (site_pos, site_xmat @ size)."""
    lo, hi = np.asarray(local_min, float), np.asarray(local_max, float)
    rotation = Rotation.from_quat(np.asarray(quat_wxyz, float), scalar_first=True)
    s = np.asarray(scale, float)
    return np.asarray(position, float) + rotation.apply(s * (lo + hi) / 2), rotation.apply(s * (hi - lo) / 2)


def in_box(point, center, half_vec, z_margin: float = IN_Z_MARGIN, label: str = "in_region") -> dict:
    """LIBERO's In (SiteObject.in_box), reproduced deliberately: the region's world-axis box of |R| . half
    (half = |half_vec|, the abs of the rotated half-extent vector), so exact only for a region rotated by
    0 or 90 degree multiples; the lower z face dropped by ``z_margin``, the point strictly inside."""
    p, c = np.asarray(point, float), np.asarray(center, float)
    half = np.abs(np.asarray(half_vec, float))
    lower, upper = c - half, c + half
    lower[2] -= z_margin
    inside = bool(np.all(p > lower) and np.all(p < upper))
    return {"success": inside, "reason": f"{label}: {np.round(p, 4).tolist()} {'inside' if inside else 'outside'} "
                                         f"{np.round(lower, 4).tolist()}..{np.round(upper, 4).tolist()}"}


def joint_above(q: float, value: float, joint: str = "joint") -> dict:
    """``q > value`` (strict; LIBERO's close of a negative-open drawer is qpos > 0.0)."""
    return {"success": bool(q > value), "reason": f"{joint} = {q:.4f} (> {value})"}


def joint_below(q: float, value: float, joint: str = "joint") -> dict:
    """``q < value`` (strict; LIBERO's open of a cabinet drawer is qpos < -0.14)."""
    return {"success": bool(q < value), "reason": f"{joint} = {q:.4f} (< {value})"}
