import numpy as np
from scipy.spatial.transform import Rotation as R

from guide_core.types.geometry import Pose, Transform
from guide_ex.core.base_node import BaseNode
from guide_ex.core.states import DemoStatus, ExecutionResult, Layer
from guide_ex.recipe import node


@node()
class TransformPose(BaseNode):
    level = Layer.UTILITY

    def __init__(self, alias=None, dynamic_map=None, static_args=None, output_map=None):
        super().__init__("TransformPose", alias, dynamic_map, static_args, output_map)

    def run(self, l_pose: Pose | Transform, r_pose: Pose | Transform) -> ExecutionResult:
        """
        Composes two poses or transforms: ``l_pose * r_pose``.

        Args:
            l_pose (Pose | Transform): The left operand, e.g. a frame or an offset to apply.
            r_pose (Pose | Transform): The right operand, given in ``l_pose``'s frame.
        Returns:
            ExecutionResult: outputs `pose`, of ``l_pose``'s type.
        """
        pose = l_pose * r_pose

        return ExecutionResult(status=DemoStatus.PERFECT, outputs={"pose": pose})


@node()
class InvertPose(BaseNode):
    level = Layer.UTILITY

    def __init__(self, alias=None, dynamic_map=None, static_args=None, output_map=None):
        super().__init__("InvertPose", alias, dynamic_map, static_args, output_map)

    def run(self, pose: Pose | Transform) -> ExecutionResult:
        """
        Inverts the given pose or transform.

        Args:
            pose (Pose | Transform): The pose or transform to be inverted.
        Returns:
            ExecutionResult: The result containing the inverted pose.
        """
        inverted_pose = pose.inv()

        return ExecutionResult(status=DemoStatus.PERFECT, outputs={"pose": inverted_pose})


def is_at_offset(pose: Pose, reference: Pose, offset, tolerance) -> bool:
    """True if `pose` sits at `reference` + `offset`, within `tolerance` on each axis.

    Offset and tolerance are 3-vectors in the frame both poses are in (world: z up), so
    "rests on top of" is an offset of (0, 0, height) and "next in a row" (spacing, 0, 0).
    """
    d = pose.position.to_numpy() - reference.position.to_numpy() - np.asarray(offset, float)
    return bool(np.all(np.abs(d) <= np.asarray(tolerance, float)))


@node()
class ChainLength(BaseNode):
    """
    How many of `poses`, from the first, follow one another at a fixed offset.

    poses[i + 1] must sit at poses[i] + `offset` (see is_at_offset): a stack is
    (0, 0, height), a row (spacing, 0, 0). Counting stops at the first gap, so whatever
    lies beyond a gap counts as not in place, even if it rests on something.
    """

    level = Layer.UTILITY

    def __init__(self, alias=None, dynamic_map=None, static_args=None, output_map=None):
        super().__init__("ChainLength", alias, dynamic_map, static_args, output_map)

    def run(self, poses: list, offset, tolerance=(0.02, 0.02, 0.01)) -> ExecutionResult:
        """
        Args:
            poses: The chain, first element first.
            offset: Where each element sits relative to the one before it.
            tolerance: Largest deviation from `offset` per axis.
        Returns:
            ExecutionResult with `length` (elements in place, from the first) and
            `complete` (all of them).
        """
        length = min(1, len(poses))
        while length < len(poses) and is_at_offset(
            poses[length], poses[length - 1], offset, tolerance
        ):
            length += 1
        return ExecutionResult(
            status=DemoStatus.PERFECT,
            outputs={"length": length, "complete": length == len(poses)},
        )


@node()
class PoseDisplacement(BaseNode):
    """
    How far a moved body is from where it started: the joint value of a drawer (slide, `distance`) or
    a knob (hinge, `angle`), read from the body's pose instead of the joint, so a stop condition needs
    no joint service.
    """

    level = Layer.UTILITY

    def __init__(self, alias=None, dynamic_map=None, static_args=None, output_map=None):
        super().__init__("PoseDisplacement", alias, dynamic_map, static_args, output_map)

    def run(self, pose: Pose, reference: Pose) -> ExecutionResult:
        """
        Args:
            pose: The body now.
            reference: The same body before it moved (the same frame).
        Returns:
            ExecutionResult with `distance` (m, between the origins) and `angle` (rad, 0 to pi, of the
            turn between the orientations).
        """
        distance = float(np.linalg.norm(pose.position.to_numpy() - reference.position.to_numpy()))
        angle = float((reference.orientation.to_scipy().inv() * pose.orientation.to_scipy()).magnitude())
        return ExecutionResult(status=DemoStatus.PERFECT, outputs={"distance": distance, "angle": angle})


@node()
class RotateAbout(BaseNode):
    """A pose turned about an axis of another pose: one step of an arc around a knob."""

    level = Layer.UTILITY

    def __init__(self, alias=None, dynamic_map=None, static_args=None, output_map=None):
        super().__init__("RotateAbout", alias, dynamic_map, static_args, output_map)

    def run(self, pose: Pose, about: Pose, angle: float, direction: float = 1.0, axis: str = "z") -> ExecutionResult:
        """
        Args:
            pose: The pose to turn.
            about: The frame whose `axis` through its origin is the turning axis.
            angle: Size of the turn, radians.
            direction: +1 or -1; its sign picks the sense of the turn (a number, so a recipe takes it as a parameter).
            axis: The axis of `about`: x, y or z.
        Returns:
            ExecutionResult with `pose`, position and orientation turned by `direction * angle`.
        """
        about_rot = about.orientation.to_scipy()
        turn = R.from_rotvec(about_rot.apply(np.eye(3)["xyz".index(axis)]) * direction * angle)
        centre = about.position.to_numpy()
        moved = Pose.from_numpy(centre + turn.apply(pose.position.to_numpy() - centre),
                                turn * pose.orientation.to_scipy())
        return ExecutionResult(status=DemoStatus.PERFECT, outputs={"pose": moved})


@node()
class PushPoses(BaseNode):
    """The two hand poses of a push: behind the object, and where the hand ends with the object at the goal."""

    level = Layer.UTILITY

    def __init__(self, alias=None, dynamic_map=None, static_args=None, output_map=None):
        super().__init__("PushPoses", alias, dynamic_map, static_args, output_map)

    def run(self, object_pose: Pose, goal_pose: Pose, reach: float = 0.10, gap: float = 0.05,
            height: float = 0.015) -> ExecutionResult:
        """
        The push runs along the table, straight from the object to the goal, in the base xy plane. The
        hand points down, its x axis along the push, so the closed fingers lead with their width.

        Args:
            object_pose: The object now (its centre).
            goal_pose: Where its centre should end up (a region prim's centre).
            reach: Hand-to-centre distance at the end of the push: the object's radius plus the finger's.
            gap: Extra distance behind the object where the hand starts, so it lowers clear of it.
            height: Hand height over the object's centre, to catch its lower rim.
        Returns:
            ExecutionResult with `start_pose` and `end_pose` (hand poses, base frame).
        """
        obj, goal = object_pose.position.to_numpy(), goal_pose.position.to_numpy()
        along = goal[:2] - obj[:2]
        norm = np.linalg.norm(along)
        if norm < 1e-6:
            return ExecutionResult(status=DemoStatus.FAILURE, error_message="PushPoses: the object is already at the goal.")
        along /= norm
        rot = R.from_euler("z", np.arctan2(along[1], along[0])) * R.from_euler("x", np.pi)

        def hand(xy):
            return Pose.from_numpy(np.array([*xy, obj[2] + height]), rot)

        return ExecutionResult(status=DemoStatus.PERFECT, outputs={
            "start_pose": hand(obj[:2] - along * (reach + gap)), "end_pose": hand(goal[:2] - along * reach)})
