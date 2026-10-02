"""How a checkpoint's inputs and outputs map onto GUIDE's robot.

A policy trained elsewhere agrees with this arm on the SHAPE of things -- 8 state
dimensions, 7 action dimensions, an end-effector pose and a gripper -- and disagrees
on almost every convention inside that shape. Getting one of them wrong produces a
policy that loads, runs, moves plausibly and never completes the task, which is the
most expensive failure mode this project has.

So the conventions come as a SET, chosen by name, rather than as a handful of flags
that can be half-applied:

``guide``   what block_bin's own datasets record. Three cameras, keyed by their own
            names; a rotation vector as ``as_rotvec()`` returns it; both finger
            dimensions the same positive number; actions as displacements in METRES
            with the gripper an absolute finger position.

``libero``  what ``lerobot/smolvla_libero`` was trained on, read off its own
            normaliser rather than assumed:

            * TWO cameras. The preprocessor renames ``observation.images.image`` and
              ``observation.images.image2`` to camera1/camera2; the third camera slot
              in its config.json is never filled.
            * The rotation vector lives in the POSITIVE hemisphere: wx spans
              [+0.35, +3.67] and never goes negative. GUIDE's gripper points down, so
              |rotvec| is about pi, where +n and -n are the same rotation and
              ``as_rotvec()`` returns either -- half of GUIDE's frames therefore land
              17.8 sigma outside this checkpoint's range purely as a sign.
            * The two finger dimensions MIRROR: q1 in [-0.001, +0.042], q2 in
              [-0.042, +0.001]. GUIDE repeats one positive number, which puts q2 4.8
              sigma out.
            * Actions are robosuite OSC_POSE units, normalised to +-1 and scaled by
              the controller, not metres. The gripper channel is BINARY +-1 (its q01
              is -1.0 and its q99 +0.92), where positive closes.

With the state conventions applied, every channel of GUIDE's home pose lands inside
+-0.9 sigma of this checkpoint's training distribution. The visual domain is still
entirely different, which is the thing the ablation is actually asking about.
"""

import numpy as np

# [dx, dy, dz, dwx, dwy, dwz, gripper] -- guide_dataset_tools.build.DELTA_NAMES,
# and the shape lerobot/smolvla_libero emits too.
DELTA_DIMS = 7

# GUIDE's cameras, keyed as its own datasets record them.
GUIDE_CAMERAS = {"top": "top", "base": "base", "wrist": "wrist"}

# What lerobot/smolvla_libero is fed, from its policy_preprocessor.json rename_map.
# `image` is LIBERO's scene view and `image2` its wrist view, so the base camera --
# looking at the table from the front -- is the nearer analogue of the first, and the
# top-down view has no counterpart and is dropped.
LIBERO_CAMERAS = {"image": "base", "image2": "wrist"}

# robosuite OSC_POSE output limits: an action of +-1 means this much displacement.
# The single biggest uncertainty in this adapter -- it is the controller's setting,
# not something recorded in the checkpoint -- so both are flags.
LIBERO_POSITION_SCALE = 0.05
LIBERO_ROTATION_SCALE = 0.5


def parse_camera_map(spec: str, default: dict) -> dict:
    """``'image=base,image2=wrist'`` -> {policy key: GUIDE camera}. Blank keeps default."""
    if not spec.strip():
        return dict(default)
    mapping = {}
    for pair in spec.split(","):
        if "=" not in pair:
            raise SystemExit(
                f"--camera-map wants 'policy_key=guide_camera' pairs, got {pair!r}. "
                f"GUIDE's cameras are {', '.join(GUIDE_CAMERAS)}."
            )
        key, camera = (part.strip() for part in pair.split("=", 1))
        if camera not in GUIDE_CAMERAS:
            raise SystemExit(
                f"--camera-map: {camera!r} is not a GUIDE camera "
                f"({', '.join(GUIDE_CAMERAS)})."
            )
        mapping[key] = camera
    return mapping


def canonical_rotvec(rotvec) -> np.ndarray:
    """The same rotation, expressed in the positive-wx hemisphere.

    Only meaningful near |rotvec| = pi, which is where GUIDE's downward gripper sits:
    a rotation of pi about n and about -n are identical, and which one ``as_rotvec()``
    hands back depends on the sign of a quaternion that is itself double-covered. Away
    from pi this would change the rotation rather than its spelling, so it is applied
    only under a convention that says the training data was canonicalised this way.
    """
    rotvec = np.asarray(rotvec, dtype=np.float64)
    return -rotvec if rotvec[0] < 0 else rotvec


def joint_state(observation: dict, joints: list[str]) -> np.ndarray:
    """The joint-space state layout, for a policy trained with delta actions only."""
    return np.array([observation[f"{j}.pos"] for j in joints], dtype=np.float32)


def state_for(convention: str, position, rotvec, grip: float) -> np.ndarray:
    """The 8-dim state, in the convention the checkpoint was trained on."""
    if convention == "libero":
        return np.array(
            [*position, *canonical_rotvec(rotvec), grip, -grip], dtype=np.float32
        )
    return np.array([*position, *rotvec, grip, grip], dtype=np.float32)


def delta_for(
    convention: str,
    action,
    scale: float = 1.0,
    position_scale: float = LIBERO_POSITION_SCALE,
    rotation_scale: float = LIBERO_ROTATION_SCALE,
    gripper_open: float = 0.04,
):
    """7-dim action -> (position delta m, rotation delta rotvec, absolute gripper m).

    ``guide``: the deltas are already metres and radians and the gripper is already an
    absolute finger position, so ``scale`` is the operator's calibration knob and
    nothing else is touched.

    ``libero``: positions and rotations are OSC units needing DIFFERENT scales -- a
    single multiplier would put rotation an order of magnitude out -- and the gripper
    is a binary command where positive means close.
    """
    values = np.asarray(action, dtype=np.float64).reshape(-1)
    if values.size != DELTA_DIMS:
        raise ValueError(
            f"Policy returned {values.size} dims, expected {DELTA_DIMS} "
            f"[dx dy dz dwx dwy dwz gripper]. Was this checkpoint trained on a dataset "
            f"built with --eef-delta-action? A joint-space policy belongs in "
            f"eval_policy.py."
        )

    if convention != "libero":
        return values[0:3] * scale, values[3:6] * scale, float(values[6])

    return (
        values[0:3] * position_scale * scale,
        values[3:6] * rotation_scale * scale,
        0.0 if values[6] > 0 else gripper_open,
    )
