"""The FR3 test fixture: a description built Kit-free from the XRDF test's raw exporter outputs.

franka_description's FR3 stands in for the USD: the XRDF test measured its link frames
equal to the USD's, except the right finger, whose USD frame equals the left finger's at
q = 0. Its inertials stand in for PhysX's.
"""

import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pinocchio as pin

DATA = Path(__file__).parent / "data"
FD_URDF = (DATA / "fr3_franka_description.urdf").read_text()
ARM = [f"fr3_joint{i}" for i in range(1, 8)]


def fd_dynamics(q):
    """(M, g) of franka_description's FR3, fingers locked, by Pinocchio: the physical reference."""
    full = pin.buildModelFromXML(FD_URDF)
    model = pin.buildReducedModel(full, [full.getJointId(n) for n in full.names[1:] if n not in ARM], pin.neutral(full))
    data = model.createData()
    M = pin.crba(model, data, q)
    return np.triu(M) + np.triu(M, 1).T, pin.computeGeneralizedGravity(model, data, q).copy()
