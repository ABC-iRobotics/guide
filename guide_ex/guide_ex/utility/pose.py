import numpy as np
from scipy.spatial.transform import Rotation as R

from guide_core.types.geometry import Point, Pose, Transform
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
            height: float = 0.015, yaw: float = 0.0, lean: float = 0.0) -> ExecutionResult:
        """
        The push runs along the table, straight from the object to the goal, in the base xy plane. The
        hand points down, its x axis along the push, so the closed fingers lead with their width.

        Args:
            object_pose: The object now (its centre).
            goal_pose: Where its centre should end up (a region prim's centre).
            reach: Hand-to-centre distance at the end of the push: the object's radius plus the finger's.
            gap: Extra distance behind the object where the hand starts, so it lowers clear of it.
            height: Hand height over the object's centre, to catch its lower rim.
            yaw: Turn of the hand about the vertical from that (rad), e.g. to lay its 20 cm body along a fixture's face.
            lean: Tilt of the wrist ahead along the push (rad), which takes the hand's body away from what is behind.
        Returns:
            ExecutionResult with `start_pose` and `end_pose` (hand poses, base frame).
        """
        obj, goal = object_pose.position.to_numpy(), goal_pose.position.to_numpy()
        along = goal[:2] - obj[:2]
        norm = np.linalg.norm(along)
        if norm < 1e-6:
            return ExecutionResult(status=DemoStatus.FAILURE, error_message="PushPoses: the object is already at the goal.")
        along /= norm
        rot = R.from_euler("z", np.arctan2(along[1], along[0]) + yaw) * R.from_euler("x", np.pi) * R.from_euler("y", -lean)

        def hand(xy):
            return Pose.from_numpy(np.array([*xy, obj[2] + height]), rot)

        return ExecutionResult(status=DemoStatus.PERFECT, outputs={
            "start_pose": hand(obj[:2] - along * (reach + gap)), "end_pose": hand(goal[:2] - along * reach)})


def miss_offset(rng, axes, r) -> np.ndarray:
    """An offset of length uniform in [r[0], r[1]] in the span of `axes` (x, y, z in any combination
    and order: "x" is +-x, "xy" a ring, "xyz" a shell), its direction uniform there."""
    if not axes or set(axes) - set("xyz") or len(set(axes)) != len(axes):
        raise ValueError(f"axes must be distinct letters of 'xyz', not {axes!r}")
    d = np.zeros(3)
    idx = ["xyz".index(a) for a in sorted(axes)]
    d[idx] = rng.standard_normal(len(idx))
    return rng.uniform(*r) * d / np.linalg.norm(d)


@node()
class MissPose(BaseNode):
    """
    An intentional miss for retry demonstrations: `pose` moved off its goal, so the move to it
    ends beside the goal and the gripper closes or opens there.

    `miss` = {p, axes, r}: with chance p (default 1), an offset of length uniform in r = [lo, hi] m,
    direction uniform in the span of `axes` -- "x", "y", "z", "xy", "xz", "yz", "xyz" in any order --
    of the pose's own frame (a top-down grasp: x the fingers' free axis, y the one they close
    along, z down). None passes the pose through. The node misses a goal once: a pose within
    `forget` of one it already missed comes back unchanged, so a retry (a recovery that resumes
    it, the next loop pass) reaches the goal. The node instance keeps what it missed and draws
    from (seed, its call count): built per episode with a per-episode seed, an episode's misses repeat.

    Planned misses: with `key` (what the goal belongs to, e.g. the cube picked or the support placed
    on) and miss = {at: {key: {axes, r}}}, the node misses exactly the listed keys, each once, and
    never p-draws. A goal that moved (a cube that fell) keeps its key, so its re-pick is not missed.
    """

    level = Layer.UTILITY

    def __init__(self, alias=None, dynamic_map=None, static_args=None, output_map=None):
        super().__init__("MissPose", alias, dynamic_map, static_args, output_map)
        self._missed, self._keys, self._calls = [], set(), 0

    def run(self, pose: Pose, miss: dict | None = None, seed: int = 0, forget: float = 0.02,
            key: str | None = None) -> ExecutionResult:
        """
        Args:
            pose (Pose): The goal.
            miss (dict): {p: chance, axes: e.g. "xy", r: [lo, hi] metres} or {at: {key: {axes, r}}};
                None: no miss.
            seed (int): The episode's seed.
            forget (float): Metres: a goal this close to one already missed is not missed again.
            key (str): What the goal belongs to, for planned misses (`at`).
        Returns:
            ExecutionResult: outputs `pose`, the goal or the miss.
        """
        self._calls += 1
        goal = pose.position.to_numpy()
        if miss is None:
            return ExecutionResult(status=DemoStatus.PERFECT, outputs={"pose": pose})
        if "at" in miss:
            if key is None or key not in miss["at"] or key in self._keys:
                return ExecutionResult(status=DemoStatus.PERFECT, outputs={"pose": pose})
            rng = np.random.default_rng([int(seed), self._calls])
            self._keys.add(key)
            miss = miss["at"][key]
        else:
            if any(np.linalg.norm(goal - g) < forget for g in self._missed):
                return ExecutionResult(status=DemoStatus.PERFECT, outputs={"pose": pose})
            rng = np.random.default_rng([int(seed), self._calls])
            if rng.random() >= miss.get("p", 1.0):
                return ExecutionResult(status=DemoStatus.PERFECT, outputs={"pose": pose})
            self._missed.append(goal)
        offset = pose.orientation.to_numpy() @ miss_offset(rng, miss["axes"], miss["r"])
        return ExecutionResult(status=DemoStatus.PERFECT, outputs={
            "pose": Pose(position=Point.from_numpy(goal + offset), orientation=pose.orientation)})


@node()
class FaceNarrowSide(BaseNode):
    """
    A top-down grasp near an obstacle, turned about the vertical (in steps of 2 pi / symmetry, the
    grasped object's symmetry) so the hand's narrow side faces the obstacle: the pose's own x axis
    (the fingers' free axis) points at it and the fingers close tangentially, the hand's 20 cm body
    along y kept clear of it. A cube that fell next to the tower is then re-picked without touching
    the tower. Farther than `near` (horizontally) the pose comes back unchanged.
    """

    level = Layer.UTILITY

    def __init__(self, alias=None, dynamic_map=None, static_args=None, output_map=None):
        super().__init__("FaceNarrowSide", alias, dynamic_map, static_args, output_map)

    def run(self, pose: Pose, obstacle: Pose, near: float = 0.15, symmetry: int = 4,
            q7_home: float = np.pi / 4, q7_limit: float = 2.8) -> ExecutionResult:
        """
        Args:
            pose (Pose): The grasp, in the robot base frame, same frame as `obstacle` (z up).
            obstacle (Pose): What to keep clear of (its centre).
            near (float): Metres: farther than this the grasp is not turned.
            symmetry (int): Turns allowed: multiples of 2 pi / symmetry.
            q7_home (float): Joint 7 at hand yaw 0 and q1 = 0 (FR3 home: pi/4).
            q7_limit (float): Largest |q7| allowed (FR3: 2.8973, less a margin).
        Returns:
            ExecutionResult: outputs `pose`.
        """
        p = pose.position.to_numpy()
        d = obstacle.position.to_numpy()[:2] - p[:2]
        dist = np.linalg.norm(d)
        if dist > near or dist < 1e-6:
            return ExecutionResult(status=DemoStatus.PERFECT, outputs={"pose": pose})

        def turned(k):
            return R.from_euler("z", 2 * np.pi * k / symmetry) * pose.orientation.rot

        def facing(k):  # |cos| between the hand's x (horizontal part) and the way to the obstacle
            x = turned(k).as_matrix()[:2, 0]
            return abs(x @ d) / (np.linalg.norm(x) * dist + 1e-9)

        def q7(k):  # joint 7 for a down-pointing hand: q7_home + q1 - yaw of the hand's x (pose in the base frame)
            x = turned(k).as_matrix()[:2, 0]
            a = np.arctan2(p[1], p[0]) - np.arctan2(x[1], x[0])
            return abs(q7_home + (a + np.pi) % (2 * np.pi) - np.pi)

        # Two turns 180 deg apart face it equally: take one joint 7 can reach, then the smaller turn.
        k = max(range(symmetry), key=lambda k: (round(facing(k), 3), q7(k) <= q7_limit,
                                                -min(k, symmetry - k), -q7(k)))
        return ExecutionResult(status=DemoStatus.PERFECT, outputs={"pose": Pose.from_numpy(p, turned(k))})
