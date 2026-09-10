"""Quaternion ordering helpers.

Rotation *algebra* is delegated to SciPy (``scipy.spatial.transform.Rotation``),
matching the rest of ``guide_core`` (e.g. ``types/geometry.py``). The only thing
that lives here is the storage-order convention: SciPy uses scalar-last
``[x, y, z, w]`` quaternions, whereas the framework / Isaac Sim use scalar-first
``[w, x, y, z]``. These helpers convert between the two.
"""

from __future__ import annotations

import numpy as np

IDENTITY_QUAT: np.ndarray = np.array([1.0, 0.0, 0.0, 0.0])  # [w, x, y, z]


def as_vec(arr, n: int | None = None) -> np.ndarray:
    """Return ``arr`` as a 1-D float array, optionally asserting its length."""
    v = np.asarray(arr, dtype=float).reshape(-1)
    if n is not None and v.shape[0] != n:
        raise ValueError(f"expected length-{n} vector, got shape {v.shape}")
    return v


def as_range(matrix) -> tuple[np.ndarray, np.ndarray]:
    """Split a ``[[x_min, x_max], [y_min, y_max], [z_min, z_max]]`` matrix into
    ``(low, high)`` vectors. An empty/None matrix is the zero range."""
    m = np.asarray(matrix if matrix else np.zeros((3, 2)), dtype=float)
    if m.shape != (3, 2):
        raise ValueError(f"expected a 3x2 [[min, max], ...] matrix, got shape {m.shape}")
    return m[:, 0].copy(), m[:, 1].copy()


def wxyz_to_xyzw(q) -> np.ndarray:
    """Scalar-first ``[w, x, y, z]`` -> SciPy scalar-last ``[x, y, z, w]``."""
    w, x, y, z = as_vec(q, 4)
    return np.array([x, y, z, w])


def xyzw_to_wxyz(q) -> np.ndarray:
    """SciPy scalar-last ``[x, y, z, w]`` -> scalar-first ``[w, x, y, z]``."""
    x, y, z, w = as_vec(q, 4)
    return np.array([w, x, y, z])

# ----------------------------------------------------------------------
# Convention note: SciPy as_quat()/from_quat() use scalar-last [x,y,z,w];
# guide_core and Isaac Sim use scalar-first [w,x,y,z]. Keep all conversions
# routed through wxyz_to_xyzw / xyzw_to_wxyz so the boundary stays explicit.
