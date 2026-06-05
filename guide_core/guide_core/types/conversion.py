import numpy as np
from geometry_msgs.msg import Point as RosPoint
from geometry_msgs.msg import Quaternion as RosQuaternion
from geometry_msgs.msg import Vector3 as RosVector3
from scipy.spatial.transform import Rotation as R

# try:
#     import carb
#     from pxr import Gf
# except Exception as e:
#     pass


# -----------------------------
# Rotation
# -----------------------------

# Quaternion


# def scipy_rot_to_gf_quat(rot: R) -> Gf.Quatd:
#     q = rot.as_quat()  # (x,y,z,w)
#     return Gf.Quatd(float(q[3]), Gf.Vec3d(float(q[0]), float(q[1]), float(q[2])))


# def gf_quat_to_scipy_rot(q: Gf.Quatd) -> R:
#     im = q.GetImaginary()
#     w = q.GetReal()
#     return R.from_quat([im[0], im[1], im[2], w])


def ros_quat_to_scipy_rot(q: RosQuaternion) -> R:
    return R.from_quat([q.x, q.y, q.z, q.w])  # (x,y,z,w)


def scipy_rot_to_ros_quat(rot: R) -> RosQuaternion:
    q = rot.as_quat()  # (x,y,z,w)
    if len(q.shape) != 1:
        q = q[0, :]
    return RosQuaternion(x=q[0], y=q[1], z=q[2], w=q[3])


# Matrix


# def gf_matrix_to_scipy_rot(m: Gf.Matrix3d) -> R:
#     return R.from_matrix((m.T / scale_factor_from_gf_matrix(m)).T)


# def scipy_rot_to_gf_matrix(rot: R) -> Gf.Matrix3d:
#     return Gf.Matrix3d(rot.as_matrix().tolist())


# def scale_factor_from_gf_matrix(m: Gf.Matrix3d) -> float:
#     return np.absolute(np.linalg.eigvals(m))


# -----------------------------
# Vector3 / Point
# -----------------------------


# def vec3_to_gf(v: np.ndarray) -> Gf.Vec3d:
#     return Gf.Vec3d(float(v[0]), float(v[1]), float(v[2]))


# def gf_to_vec3(v: Gf.Vec3d) -> np.ndarray:
#     return np.array([v[0], v[1], v[2]], dtype=float)


def vec3_to_ros_vec(v: np.ndarray) -> RosVector3:
    return RosVector3(x=v[0], y=v[1], z=v[2])


def vec3_to_ros_point(v: np.ndarray) -> RosPoint:
    return RosPoint(x=v[0], y=v[1], z=v[2])


def ros_to_vec3(v: RosVector3 | RosPoint) -> np.ndarray:
    return np.array([v.x, v.y, v.z], dtype=float)


# def ros_point_to_gf(v: RosVector3 | RosPoint) -> Gf.Vec3d:
#     return vec3_to_gf(ros_to_vec3(v))


# -----------------------------
# Gf <-> Carb
# -----------------------------


# def gf_to_carb_vec3(v: Gf.Vec3d) -> carb.Float3:
#     return carb.Float3(v[0], v[1], v[2])


# def carb_to_gf_vec3(v: carb.Float3) -> Gf.Vec3d:
#     return Gf.Vec3d(v[0], v[1], v[2])


# def gf_quat_to_carb_quat(q: Gf.Quatd) -> carb.Float4:
#     im = q.GetImaginary()
#     return carb.Float4(im[0], im[1], im[2], q.GetReal())


# def carb_quat_to_gf_quat(q: carb.Float4) -> Gf.Quatd:
#     return Gf.Quatd(q[3], Gf.Vec3d(q[0], q[1], q[2]))
