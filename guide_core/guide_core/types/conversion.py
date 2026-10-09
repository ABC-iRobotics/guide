import numpy as np
from geometry_msgs.msg import Point as RosPoint
from geometry_msgs.msg import Quaternion as RosQuaternion
from geometry_msgs.msg import Vector3 as RosVector3
from scipy.spatial.transform import Rotation as R

# -----------------------------
# Rotation
# -----------------------------

# Quaternion


def ros_quat_to_scipy_rot(q: RosQuaternion) -> R:
    return R.from_quat([q.x, q.y, q.z, q.w])  # (x,y,z,w)


def scipy_rot_to_ros_quat(rot: R) -> RosQuaternion:
    q = rot.as_quat()  # (x,y,z,w)
    if len(q.shape) != 1:
        q = q[0, :]
    return RosQuaternion(x=q[0], y=q[1], z=q[2], w=q[3])


# -----------------------------
# Vector3 / Point
# -----------------------------


def vec3_to_ros_vec(v: np.ndarray) -> RosVector3:
    return RosVector3(x=v[0], y=v[1], z=v[2])


def vec3_to_ros_point(v: np.ndarray) -> RosPoint:
    return RosPoint(x=v[0], y=v[1], z=v[2])


def ros_to_vec3(v: RosVector3 | RosPoint) -> np.ndarray:
    return np.array([v.x, v.y, v.z], dtype=float)
