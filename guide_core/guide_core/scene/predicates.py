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
PREDICATE_PATHS = {"on": ("a", "b"), "on_region": ("obj", "region", "touch"), "in_region": ("obj", "region"),
                   "joint_above": ("joint",), "joint_below": ("joint",)}
ON_Z_LOW = -0.005  # On(obj, region): the origin's height over the region's top face, from this far below...
ON_Z_HIGH = 0.10  # ...to this far above it (SiteObject.under's other_height)


def _touching(ok: bool, touching, reason: str) -> dict:
    """The geometric test ``ok``, then, only if it holds, the contact LIBERO also asks for: ``touching``, a
    callable (a mesh clash query is not free) or None for no contact requirement (a site on the table has none)."""
    if ok and touching is not None and not touching():
        return {"success": False, "reason": f"{reason}, but not touching"}
    return {"success": bool(ok), "reason": reason}


def on(a, b, xy_tol: float = XY_TOL, label: str = "on", touching=None) -> dict:
    """``a`` rests on ``b`` (LIBERO's ObjectState.check_ontop): b's origin no higher than a's, origins
    within ``xy_tol`` in xy (strict), and, when ``touching`` is given, the two touch."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    xy, dz = float(np.hypot(*(a[:2] - b[:2]))), float(a[2] - b[2])
    return _touching(dz >= 0.0 and xy < xy_tol, touching, f"{label}: xy {xy:.4f} m (< {xy_tol}), a {dz:+.4f} m above b (>= 0)")


def box_world(local_min, local_max, position, quat_wxyz, scale):
    """World (centre, half-extent vector in the box's own axes = LIBERO's site size) of a prim's own box
    [local_min, local_max] under its world pose and scale."""
    lo, hi = np.asarray(local_min, float), np.asarray(local_max, float)
    s = np.asarray(scale, float)
    centre = np.asarray(position, float) + Rotation.from_quat(np.asarray(quat_wxyz, float), scalar_first=True).apply(s * (lo + hi) / 2)
    return centre, np.abs(s * (hi - lo) / 2)


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


def on_region(point, centre, quat_wxyz, half, label: str = "on_region", touching=None) -> dict:
    """``point`` (an object's origin) is on a region (LIBERO's SiteObject.under): in the region's own frame the point
    is within |xy| < half_xy and ``ON_Z_LOW`` < z - half_z < ``ON_Z_HIGH`` (half_z - 0.005 < z < half_z + 0.10), plus
    contact with the region's parent object when ``touching`` is given.
    LIBERO multiplies by the site's matrix where it should multiply by its transpose, which is the same only for a
    region turned 0 or 180 degrees about z; this uses the transpose, so a tilted region (the wine rack's) is tested
    in its own frame."""
    d = Rotation.from_quat(np.asarray(quat_wxyz, float), scalar_first=True).inv().apply(np.asarray(point, float) - np.asarray(centre, float))
    half = np.asarray(half, float)
    dz = float(d[2] - half[2])
    ok = bool(ON_Z_LOW < dz < ON_Z_HIGH and np.all(np.abs(d[:2]) < half[:2]))
    return _touching(ok, touching, f"{label}: in the region frame xy {np.round(d[:2], 4).tolist()} (|.| < {np.round(half[:2], 4).tolist()}), "
                                   f"{dz:+.4f} m over its top ({ON_Z_LOW} .. {ON_Z_HIGH})")


def joint_above(q: float, value: float, joint: str = "joint", inclusive: bool = False) -> dict:
    """``q > value`` (strict; LIBERO's close of a negative-open drawer is qpos > 0.0), or ``q >= value`` when
    ``inclusive`` (LIBERO's turn-on of the stove: qpos >= 0.5)."""
    return {"success": bool(q >= value if inclusive else q > value), "reason": f"{joint} = {q:.4f} ({'>=' if inclusive else '>'} {value})"}


def joint_below(q: float, value: float, joint: str = "joint") -> dict:
    """``q < value`` (strict; LIBERO's open of a cabinet drawer is qpos < -0.14)."""
    return {"success": bool(q < value), "reason": f"{joint} = {q:.4f} (< {value})"}
