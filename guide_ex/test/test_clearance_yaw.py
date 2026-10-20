"""ChooseClearanceYaw: the finger axis with more room to the other blocks, if the wrist can turn to it."""

import numpy as np
from scipy.spatial.transform import Rotation as R

from guide_core.types.geometry import Point, Pose, Rotation
from guide_ex.utility.rotation import ChooseClearanceYaw


def pose(x, y, yaw, flip=False):
    rot = R.from_euler("z", yaw) * (R.from_euler("x", np.pi) if flip else R.identity())
    return Pose(Point([x, y, 0.0]), Rotation.from_scipy(rot))


def choose(grasp, others):
    out = ChooseClearanceYaw().run(grasp, others).outputs["pose"]
    return np.arctan2(*out.orientation.to_scipy().as_matrix()[1::-1, 0])  # heading of the hand's x axis


def test_ep3_neighbour_5cm_away_turns_the_hand_90():
    green = pose(0.457, 0.0915, np.radians(-22.8), flip=True)
    yellow = pose(0.469, 0.143, np.radians(-46.3))
    assert np.isclose(choose(green, [green, yellow]), np.radians(-22.8 + 90))  # the target itself is ignored


def test_no_neighbours_keeps_the_yaw():
    green = pose(0.457, 0.0915, np.radians(-22.8), flip=True)
    assert np.isclose(choose(green, []), np.radians(-22.8))
    assert np.isclose(choose(green, [pose(0.2, -0.3, 0.0)]), np.radians(-22.8))  # far away: tie


def test_unreachable_turn_keeps_the_yaw():
    # base at the origin: a grasp behind it (q1 = -121 deg) puts q7 = 45 - 121 - 90 = -166 deg
    green = pose(-0.3, -0.5, 0.0, flip=True)
    yellow = pose(-0.3, -0.45, 0.0)  # in the way of the pad on the y-axis finger line
    assert np.isclose(choose(green, [yellow]), 0.0)
