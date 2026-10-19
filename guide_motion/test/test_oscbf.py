"""The OSCBF submodule (external/oscbf) runs on this venv (numpy 2.3.1, jax 0.4.30, cbfpy 0.1.0) and agrees with Pinocchio."""

import tempfile
import xml.etree.ElementTree as ET

import numpy as np

import guide_motion  # noqa: F401  (JAX settings before JAX is imported)
from fr3_fixture import ARM, FD_URDF, fd_dynamics


def test_oscbf_dynamics_match_pinocchio_on_franka_description():
    from oscbf.core.manipulator import Manipulator

    root = ET.fromstring(FD_URDF)
    for joint in root.iter("joint"):  # OSCBF wants every non-arm joint fixed
        if joint.get("type") != "fixed" and joint.get("name") not in ARM:
            joint.set("type", "fixed")
            for mimic in joint.findall("mimic"):
                joint.remove(mimic)
    for link in root.iter("link"):  # and loads every mesh it finds
        for tag in ("visual", "collision"):
            for element in link.findall(tag):
                link.remove(element)
    with tempfile.NamedTemporaryFile("w", suffix=".urdf") as f:
        f.write(ET.tostring(root, encoding="unicode"))
        f.flush()
        man = Manipulator.from_urdf(f.name)
    for q in np.random.default_rng(0).uniform(-1.5, 1.5, size=(3, 7)):
        M, g = fd_dynamics(q)
        np.testing.assert_allclose(np.asarray(man.mass_matrix(q)), M, atol=1e-12)
        np.testing.assert_allclose(np.asarray(man.gravity_vector(q)), g, atol=1e-12)
