"""Step through a recorded rollout offline: what the policy saw, meant, and got.

No simulator, no ROS, no policy -- it reads the directory ``debug_rollout`` wrote.

    python -m block_bin.replay_rollout ~/eval_debug/<run>          # every seed
    python -m block_bin.replay_rollout ~/eval_debug/<run> --seed 4  # open at one
    python -m block_bin.replay_rollout ~/eval_debug/<run>/seed_0004 # just that one

Point it at a whole run and it finds every ``seed_NNNN`` folder, loading them on
demand so a twenty-seed run still opens instantly. Point it at one seed, or at a
single trial, and that works too.

Each seed loads BOTH its trials and draws them together, which is the point: one
identical scene, told "left bin" once and "right bin" once, and the step where the two
tool paths separate is the step where the sentence started to matter.

    space        play / pause          left / right   one step
    , / .        ten steps             home / end     first / last
    n / b        next / previous seed  (the step and the selected trial are kept)
    drag the 3D panel to rotate it; v resets the camera
    t            switch which trial the plots follow
                 (both camera rows are always shown; the selected one is starred)
    g            jump to the step the gripper was first told to close

What each panel is for:

* Cameras -- what the POLICY saw at this step, not a spectator view. One row per
  trial, so the left-instructed and right-instructed inputs sit above each other at
  the same step: the scene is identical, so anything that differs between the rows
  is downstream of the sentence. The wrist frame at the grasp step is the fastest way
  to see a systematic offset -- if the block is not centred there when the fingers
  close, nothing downstream can save it.
* 3D -- both tools' actual paths, and from each current position that trial's QUEUED
  plan (dashed, same colour): the actions the policy has already committed to and has
  not executed yet. Two plans rather than one, because the question is whether the two
  rollouts are heading somewhere DIFFERENT. A red stub joins each arm to where its IK
  was aiming this step; a long stub means the arm is not tracking, and no amount of
  policy debugging will help until it does.

  Isotropic and fixed: one metre is the same length on all three axes and the bounds
  cover the whole seed, so a divergence is read at its true size and does not creep
  while you step. Drag to rotate; the camera survives stepping and seed changes.
* Joints -- measured solid, commanded dashed. They should be indistinguishable. A
  visible gap is the arm failing to follow, which is the same story the red stub
  tells in Cartesian space.
* Tracking -- pose error on the left axis, and on the right the wrist TILT away from
  the pose the rollout started in. The gripper starts pointing down and has to stay
  near there to close around a block on the table; past the dotted 30 deg line the
  fingers meet the table instead. Watch this before blaming the policy's aim.
* Gripper -- commanded against measured. This is where a failed grasp is obvious:
  the fingers close to ~0 on empty air and stall around the block's half-width when
  they have it. A close command with no stall is a miss, whatever the arm did next.

The plan is drawn through the same --action-scale and clamp ceilings the executor
used, so it is what WOULD have happened rather than a longer line the clamp trimmed.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Slider
from scipy.spatial.transform import Rotation

from block_bin.rollout_trace import find_pairs, frame_path, load_pair, seed_number

# debug_rollout writes left/ and right/; named/flipped is the earlier labelling and
# still opens.
TRIAL_COLOUR = {
    "left": "tab:blue", "right": "tab:orange", "named": "tab:blue", "flipped": "tab:orange"
}
# Looking down the table from behind and above the arm, which is roughly the base
# camera's angle -- so the 3D panel and the video agree about which way is left.
DEFAULT_VIEW = (28.0, -60.0, 0.0)
# Keys this viewer binds that matplotlib also binds by default: left/right drive the
# navigation stack, home resets the view, g toggles the grid. Both handlers fire, so
# stepping would pan the plots and jumping to the grasp would strip the gridlines.
OUR_KEYS = (" ", "left", "right", ",", ".", "home", "end", "g", "t", "n", "b", "v")
JOINT_COLOURS = plt.get_cmap("tab10").colors


def free_keys(keys=OUR_KEYS) -> None:
    """Take these keys back from matplotlib's default shortcuts.

    Quietly fighting the toolbar is worse than not binding them: the viewer would step
    a frame AND pan the axes on every arrow press.
    """
    for name, bound in list(plt.rcParams.items()):
        if name.startswith("keymap.") and isinstance(bound, list):
            plt.rcParams[name] = [key for key in bound if key not in keys]


def planned_path(step: dict, meta: dict) -> np.ndarray:
    """The queued actions integrated forward from where the tool is now.

    Position deltas add (apply_delta), so this is a running sum -- through the same
    scale and ceilings the executor applied, because a plan drawn without them is a
    trajectory the arm was never going to take.
    """
    start = np.asarray(step.get("eef_position", [0, 0, 0]), dtype=float)
    plan = step.get("plan") or []
    if not plan:
        return start.reshape(1, 3)

    scale = float(meta.get("action_scale", 1.0))
    max_linear = float(meta.get("max_step", 0.0) or 0.0)
    max_angular = float(meta.get("max_rotation_step", 0.0) or 0.0)

    points, position = [start], start.copy()
    for action in plan:
        values = np.asarray(action, dtype=float).reshape(-1)
        if values.size < 6:
            break
        dposition, drotvec = values[0:3] * scale, values[3:6] * scale
        factor = 1.0
        linear, angular = np.linalg.norm(dposition), np.linalg.norm(drotvec)
        if max_linear > 0 and linear > max_linear:
            factor = min(factor, max_linear / linear)
        if max_angular > 0 and angular > max_angular:
            factor = min(factor, max_angular / angular)
        position = position + dposition * factor
        points.append(position.copy())
    return np.asarray(points)


def tilt_from_start(steps: list):
    """Degrees the tool has rotated away from the pose it started the rollout in.

    The gripper starts pointing straight down and has to stay roughly there to close
    around a block on the table, so this is "how far has the wrist tumbled" -- the one
    number that says whether a grasp had any chance before the fingers even moved.

    A true relative rotation, not a difference of rotation vectors. The down-pose sits
    at |rotvec| = pi, exactly where the axis-angle representation is antipodally
    ambiguous and flips sign on physically identical poses, so componentwise
    differences there are meaningless and would show tumbling that never happened.
    """
    rotvecs = np.stack([column(steps, "eef_rotvec", i) for i in range(3)], axis=1)
    if not len(rotvecs) or np.isnan(rotvecs).any():
        return None
    poses = Rotation.from_rotvec(rotvecs)
    return np.degrees((poses[0].inv() * poses).magnitude())


def latest_poses(steps: list, index: int, meta: dict) -> dict:
    """Object positions as of this step, in the frame the tool is drawn in.

    Poses are polled on an interval, so most steps carry none and the last known
    position is the honest thing to draw.

    PoseRequest answers in WORLD frame while the policy's state -- and so the recorded
    tool path -- is scene-relative. Traces record ``scene_origin`` to state the
    difference; without the conversion the blocks and bins float a metre above the
    trajectory that is supposed to be reaching them. Older traces, recorded when the
    tool was itself world-frame, carry no origin and get no correction.
    """
    found = meta.get("objects_at_start", {})
    for step in range(index, -1, -1):
        if step < len(steps) and steps[step].get("poses"):
            found = steps[step]["poses"]
            break
    origin = meta.get("scene_origin")
    if not origin:
        return found
    shift = np.asarray(origin, dtype=float)
    return {
        name: (np.asarray(position, dtype=float) - shift).tolist()
        for name, position in found.items()
    }


def column(steps: list, key: str, index: int = None) -> np.ndarray:
    """One field over all steps, as an array; ``index`` picks a component."""
    values = []
    for step in steps:
        value = step.get(key)
        if value is None:
            values.append(np.nan)
        elif index is None:
            values.append(value)
        else:
            values.append(value[index] if index < len(value) else np.nan)
    return np.asarray(values, dtype=float)


class Replay:
    """The figure and everything that redraws when the step changes."""

    def __init__(self, pairs, trial: str, seed=None):
        self.pair_paths = [Path(path) for path in pairs]
        if not self.pair_paths:
            raise SystemExit("No seeds to replay.")
        self.wanted_trial = trial
        self._loaded = {}
        self.position = 0
        if seed is not None:
            match = [i for i, p in enumerate(self.pair_paths) if seed_number(p) == seed]
            if not match:
                raise SystemExit(
                    f"--seed {seed} is not in this run. Recorded: "
                    f"{[seed_number(p) for p in self.pair_paths]}"
                )
            self.position = match[0]
        self.index = 0
        self.playing = False
        self.select(self.position, redraw=False)

        cameras = self.pair[self.trial]["meta"].get("cameras") or []
        frames = self.pair[self.trial]["directory"] / "frames"
        self.cameras = [c for c in cameras if (frames / c).is_dir()][:3]

        # One camera row per trial rather than one row for whichever trial is selected.
        # These are the policy's inputs, and the pair exists to be compared: the two
        # rollouts see an IDENTICAL scene, so anything that differs between the rows at
        # the same step is a consequence of the sentence. Toggling between them would
        # hide exactly the comparison the tool is for.
        self.rows = list(self.pair)
        free_keys()
        self.figure = plt.figure(figsize=(16, 11 + 1.6 * (len(self.rows) - 1)))
        grid = self.figure.add_gridspec(
            len(self.rows) + 3, 4,
            height_ratios=[1.0] * len(self.rows) + [1.0, 1.0, 1.0],
            hspace=0.38, wspace=0.25, left=0.04, right=0.96, top=0.95, bottom=0.08,
        )

        # Rows are keyed by POSITION in the pair, not by trial name. For a left/right
        # pair the names are the same on every seed, but a study's episodes come one to
        # a directory and each is named after itself -- keyed by name, the row built for
        # ep_0000 finds nothing in ep_0001 and blanks its picture forever after.
        self.camera_axes = []
        for row in range(len(self.rows)):
            for position, camera in enumerate(self.cameras):
                axes = self.figure.add_subplot(grid[row, position])
                axes.axis("off")
                blank = axes.imshow(np.zeros((480, 640, 3), np.uint8))
                self.camera_axes.append((row, camera, axes, blank))

        self.info = self.figure.add_subplot(grid[0:len(self.rows), 3])
        self.info.axis("off")
        self.info_text = self.info.text(
            0, 1, "", va="top", ha="left", family="monospace", fontsize=8.5,
            transform=self.info.transAxes,
        )

        base = len(self.rows)
        # The 3D view gets the whole left column below the cameras. It is the panel you
        # actually interrogate -- drag to rotate it -- and at a quarter of the figure it
        # was too small to tell a 10 cm divergence from a 40 cm one.
        self.space = self.figure.add_subplot(grid[base:base + 3, 0:2], projection="3d")
        self.joints = self.figure.add_subplot(grid[base, 2:4])
        self.gripper = self.figure.add_subplot(grid[base + 1, 2:4])
        self.tracking = self.figure.add_subplot(grid[base + 2, 2:4])
        self.tilt = self.tracking.twinx()  # made once; twinx per redraw would stack them
        # Survives the clear() in draw(); without this every step snapped the camera
        # back to default and the view could not be rotated at all while stepping.
        self.view = DEFAULT_VIEW
        self.bounds = None

        self.slider = Slider(
            self.figure.add_axes([0.08, 0.03, 0.84, 0.02]),
            "step", 0, max(self.length - 1, 1), valinit=0, valstep=1,
        )
        self.slider.on_changed(lambda value: self.goto(int(value)))
        self.figure.canvas.mpl_connect("key_press_event", self.on_key)
        self.timer = self.figure.canvas.new_timer(interval=120)
        self.timer.add_callback(self.advance)

        self.retitle()
        self.draw_static()
        self.draw()

    # -------------------------------------------------------------------- the seeds
    def select(self, position: int, redraw: bool = True) -> None:
        """Load the seed at this position in the run, keeping the step where possible.

        Loaded on demand and cached: a twenty-seed run is a few hundred MB of frames,
        and reading every steps.jsonl up front would make the viewer slow to start for
        seeds you may never look at.
        """
        self.position = position % len(self.pair_paths)
        path = self.pair_paths[self.position]
        if path not in self._loaded:
            self._loaded[path] = load_pair(path)
        self.pair = self._loaded[path]
        self.trial = self.wanted_trial if self.wanted_trial in self.pair else next(iter(self.pair))
        self.length = max((len(t["steps"]) for t in self.pair.values()), default=0)
        if not self.length:
            raise SystemExit(f"{path} has no steps; nothing to replay.")
        if not redraw:
            return

        self.index = min(self.index, self.length - 1)
        self.bounds = None  # a different seed is a different scene
        # Slider bounds move with the seed: two rollouts of a run can differ in length
        # if one was aborted, and a stale valmax would let the slider run off the end.
        self.slider.valmax = max(self.length - 1, 1)
        self.slider.ax.set_xlim(0, self.slider.valmax)
        self.retitle()
        self.draw_static()
        self.slider.set_val(self.index)
        self.draw()

    def retitle(self) -> None:
        if self.figure.canvas.manager is not None:  # None under a headless backend
            self.figure.canvas.manager.set_window_title(
                f"seed {seed_number(self.pair_paths[self.position])}  "
                f"({self.position + 1}/{len(self.pair_paths)})  "
                f"{self.pair_paths[self.position].parent}"
            )

    # ---------------------------------------------------------------- static panels
    def draw_static(self) -> None:
        """The curves that do not depend on the current step, drawn once."""
        steps = self.pair[self.trial]["steps"]
        panels = (
            (self.joints, "joints: measured solid, commanded dashed"),
            (self.gripper, "gripper"),
            (self.tracking, "tracking error (m, log) and wrist tilt (deg)"),
        )
        for axes, title in panels:
            axes.clear()
            axes.set_title(title, fontsize=9)
            axes.grid(alpha=0.3)
        self.tilt.clear()

        for joint in range(7):
            colour = JOINT_COLOURS[joint % len(JOINT_COLOURS)]
            self.joints.plot(column(steps, "q_measured", joint), color=colour, lw=1.2)
            self.joints.plot(column(steps, "q_command", joint), color=colour, lw=0.9, ls="--")
        self.joints.set_ylabel("rad", fontsize=8)

        self.gripper.plot(
            column(steps, "grip_command"), color="tab:red", lw=1.3, label="commanded"
        )
        self.gripper.plot(
            column(steps, "grip_measured"), color="tab:green", lw=1.3, label="measured"
        )
        # Where the fingers stall with a block between them. Below it means empty air.
        self.gripper.axhline(0.025, color="grey", ls=":", lw=1)
        self.gripper.legend(fontsize=7, loc="upper right")
        self.gripper.set_ylabel("m", fontsize=8)

        target = np.stack([column(steps, "target_position", i) for i in range(3)], axis=1)
        actual = np.stack([column(steps, "eef_position", i) for i in range(3)], axis=1)
        self.tracking.plot(
            np.linalg.norm(target - actual, axis=1), color="tab:red", lw=1.2,
            label="tool to IK target",
        )
        self.tracking.plot(
            column(steps, "residual"), color="tab:purple", lw=1.0, label="IK residual"
        )
        self.tracking.set_yscale("log")
        self.tracking.legend(fontsize=7, loc="upper left")
        self.tracking.set_ylabel("m", fontsize=8)

        tilt = tilt_from_start(steps)
        if tilt is not None:
            self.tilt.plot(tilt, color="tab:brown", lw=1.4, label="tilt from start pose")
            # Roughly where a parallel-jaw gripper stops being able to close around a
            # block standing on a table: past this the fingers meet the table or sweep
            # past the block rather than closing around it.
            self.tilt.axhline(30, color="tab:brown", ls=":", lw=1)
            self.tilt.yaxis.set_label_position("right")  # else it lands on top of "m"
            self.tilt.set_ylabel("deg", fontsize=8, color="tab:brown")
            self.tilt.tick_params(axis="y", labelcolor="tab:brown", labelsize=7)
            self.tilt.legend(fontsize=7, loc="upper right")

        for axes in (self.joints, self.gripper, self.tracking):
            axes.set_xlim(0, max(self.length - 1, 1))
        self.cursors = [axes.axvline(0, color="k", lw=1.0) for axes in
                        (self.joints, self.gripper, self.tracking)]

    # ------------------------------------------------------------------ step redraw
    def draw(self) -> None:
        trace = self.pair[self.trial]
        steps, meta = trace["steps"], trace["meta"]
        index = min(self.index, len(steps) - 1) if steps else 0
        step = steps[index] if steps else {}

        trials = list(self.pair.items())
        for slot, camera, axes, artist in self.camera_axes:
            name, row = trials[slot] if slot < len(trials) else (None, None)
            row_index = min(index, len(row["steps"]) - 1) if row and row["steps"] else index
            path = frame_path(row, camera, row_index) if row else None
            # Blank rather than keep the last seed's picture: a stale frame beside a
            # fresh trajectory is the one kind of wrong a viewer must not be.
            artist.set_data(
                plt.imread(path) if path is not None else np.zeros((480, 640, 3), np.uint8)
            )
            # The selected row is the one the plots below belong to, so it is marked --
            # otherwise two identical-looking rows leave you guessing which is which.
            marker = " *" if name == self.trial else ""
            axes.set_title(
                f"told {name}: {camera}{marker}",
                fontsize=9,
                color=TRIAL_COLOUR.get(name, "black"),
                fontweight="bold" if name == self.trial else "normal",
            )

        # Rotation is the whole point of this panel, and clear() throws the camera away.
        self.view = (self.space.elev, self.space.azim, self.space.roll)
        self.space.clear()
        self.space.set_title("tool paths and the queued plans (drag to rotate)", fontsize=9)

        for name, other_trace in self.pair.items():
            other_steps = other_trace["steps"]
            if not other_steps:
                continue
            colour = TRIAL_COLOUR.get(name, "grey")
            path = np.stack(
                [column(other_steps, "eef_position", i) for i in range(3)], axis=1
            )
            self.space.plot(*path.T, color=colour, alpha=0.18, lw=1.0)
            upto = min(index, len(other_steps) - 1)
            self.space.plot(*path[: upto + 1].T, color=colour, lw=1.8, label=f"told {name}")
            self.space.scatter(*path[upto], color=colour, s=55)

            # Both plans, each in its trial's colour and from its own current position.
            # One plan at a time answered "where is this rollout going"; two answer
            # "are the two rollouts going to different places", which is the question.
            plan = planned_path(other_steps[upto], other_trace["meta"])
            if len(plan) > 1:
                self.space.plot(*plan.T, color=colour, ls="--", lw=1.6)
                self.space.scatter(*plan[1:].T, color=colour, s=10, alpha=0.8)

            target = np.asarray(other_steps[upto].get("target_position", []), dtype=float)
            if target.size == 3:
                self.space.plot(*np.stack([path[upto], target]).T, color="tab:red", lw=2.4)

        for name, position in latest_poses(steps, index, meta).items():
            marker, size = ("s", 110) if name.endswith("_bin") else ("o", 70)
            colour = name.split("_")[0] if name.endswith("_block") else "black"
            self.space.scatter(*position, marker=marker, s=size,
                               color=colour if colour in ("red", "green", "blue") else
                               ("gold" if colour == "yellow" else "black"),
                               edgecolors="k", linewidths=0.5)
            self.space.text(*position, f" {name}", fontsize=7)

        self.space.set_xlabel("x", fontsize=8)
        self.space.set_ylabel("y  (left bin is -y)", fontsize=8)
        self.space.set_zlabel("z", fontsize=8)
        self.space.legend(fontsize=8, loc="upper left")
        self.frame_space()

        for cursor in self.cursors:
            cursor.set_xdata([index, index])

        self.info_text.set_text(self.describe(step, index, meta, trace))
        self.figure.canvas.draw_idle()

    def frame_space(self) -> None:
        """One cube of scene, isotropic, fixed for the whole seed, camera preserved.

        Two things this fixes. Matplotlib autoscales each axis independently, so a
        rollout that moves 0.9 m in y and 0.4 m in z was drawn as if both spanned the
        same distance -- the left/right separation this tool exists to show was being
        stretched by whatever else happened to be in frame. And the limits were being
        recomputed per step, so the geometry crept while you stepped through it.

        Bounds come from everything in the seed -- both tool paths and every object --
        so the view is stable across steps and across the two trials.
        """
        if self.bounds is None:
            points = []
            for trace in self.pair.values():
                if trace["steps"]:
                    points.append(
                        np.stack(
                            [column(trace["steps"], "eef_position", i) for i in range(3)],
                            axis=1,
                        )
                    )
                objects = latest_poses(trace["steps"], len(trace["steps"]), trace["meta"])
                if objects:
                    points.append(np.asarray(list(objects.values()), dtype=float))
            if not points:
                return
            cloud = np.concatenate([p for p in points if p.size])
            # A trace recorded without positions is all NaN here, and NaN limits raise.
            cloud = cloud[np.isfinite(cloud).all(axis=1)]
            if not len(cloud):
                return
            low, high = cloud.min(axis=0), cloud.max(axis=0)
            centre = (low + high) / 2
            # One radius for all three axes: equal metres per unit in x, y and z.
            radius = max(float(np.max(high - low)) / 2, 0.05) * 1.1
            self.bounds = (centre, radius)

        centre, radius = self.bounds
        self.space.set_xlim(centre[0] - radius, centre[0] + radius)
        self.space.set_ylim(centre[1] - radius, centre[1] + radius)
        self.space.set_zlim(centre[2] - radius, centre[2] + radius)
        self.space.set_box_aspect((1, 1, 1))
        elev, azim, roll = self.view
        self.space.view_init(elev=elev, azim=azim, roll=roll)

    def describe(self, step: dict, index: int, meta: dict, trace: dict) -> str:
        summary = meta.get("summary", {})
        flags = [name for name in ("held", "clamped", "unreachable") if step.get(name)]
        lines = [
            f"seed {meta.get('seed')}  trial {meta.get('trial')}"
            + (f"   [{self.position + 1}/{len(self.pair_paths)}]"
               if len(self.pair_paths) > 1 else ""),
            f'"{meta.get("instruction", "")}"',
            # scene_side is the current field; named_side is what traces recorded before
            # the trials were relabelled called it.
            f"scene drew : {meta.get('scene_side', meta.get('named_side', '?'))}"
            f"   asked for: {meta.get('asked_for', '?')}",
            "",
            f"step       : {index} / {self.length - 1}",
            f"sim time   : {step.get('sim_seconds', float('nan')):.2f} s",
            f"residual   : {step.get('residual', float('nan')):.2e} m",
            f"grip cmd   : {step.get('grip_command', float('nan')):.4f}",
            f"grip meas  : {step.get('grip_measured', float('nan')):.4f}",
            f"flags      : {', '.join(flags) or '-'}",
            f"plan ahead : {len(step.get('plan') or [])} actions",
            "",
            f"ended in   : {summary.get('reached', '?')}",
        ]
        for side, distance in (summary.get("distance_to") or {}).items():
            lines.append(f"  block to {side:<5}: {distance:.3f} m")
        lines += ["", "space play  <- -> step  , . x10",
                  "t plots follow   g grasp step", "drag 3D to rotate   v reset view"]
        if len(self.pair_paths) > 1:
            lines.append("n / b     next / previous seed")
        return "\n".join(lines)

    # ---------------------------------------------------------------------- controls
    def goto(self, index: int) -> None:
        self.index = max(0, min(int(index), self.length - 1))
        self.draw()

    def advance(self) -> None:
        if self.index >= self.length - 1:
            self.playing = False
            self.timer.stop()
            return
        self.slider.set_val(self.index + 1)

    def grasp_step(self) -> int:
        """First step the gripper was commanded closed, or the current one."""
        for step in self.pair[self.trial]["steps"]:
            if (step.get("grip_command") or 1.0) < 0.02:
                return step["step"]
        return self.index

    def on_key(self, event) -> None:
        if event.key == " ":
            self.playing = not self.playing
            self.timer.start() if self.playing else self.timer.stop()
        elif event.key in ("right", "left", ".", ","):
            delta = {"right": 1, "left": -1, ".": 10, ",": -10}[event.key]
            self.slider.set_val(max(0, min(self.index + delta, self.length - 1)))
        elif event.key == "home":
            self.slider.set_val(0)
        elif event.key == "end":
            self.slider.set_val(self.length - 1)
        elif event.key == "g":
            self.slider.set_val(self.grasp_step())
        elif event.key == "t" and len(self.pair) > 1:
            names = list(self.pair)
            self.trial = names[(names.index(self.trial) + 1) % len(names)]
            self.wanted_trial = self.trial  # keep it selected across seeds
            self.draw_static()
            self.draw()
        elif event.key in ("n", "b") and len(self.pair_paths) > 1:
            self.select(self.position + (1 if event.key == "n" else -1))
        elif event.key == "v":
            # Onto the AXES, not self.view: draw() captures the camera from the axes
            # first thing, so assigning self.view here would be overwritten at once.
            self.space.view_init(*DEFAULT_VIEW)
            self.draw()


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "trace",
        type=str,
        help="A run directory of seed_NNNN folders, one seed_NNNN, or a single trial.",
    )
    parser.add_argument(
        "--trial", type=str, default="left", help="Which trial starts selected (left / right)."
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="Open at this seed instead of the first."
    )
    parser.add_argument("--save", type=str, default="", help="Write a PNG of one step and exit.")
    parser.add_argument("--step", type=int, default=0, help="Step to show (with --save).")
    args = parser.parse_args()

    pairs = find_pairs(Path(args.trace).expanduser())
    if len(pairs) > 1:
        print(f"{len(pairs)} seeds: {[seed_number(p) for p in pairs]}   (n / b to change)")
    replay = Replay(pairs, args.trial, args.seed)
    if args.save:
        # Headless use: one frame to a file, so a trace can be looked at over ssh or
        # pasted into a report without a display.
        replay.goto(args.step)
        replay.figure.savefig(args.save, dpi=110)
        print(args.save)
        return
    plt.show()


if __name__ == "__main__":
    main()
