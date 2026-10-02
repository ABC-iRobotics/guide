"""Everything one rollout did, on disk, in a form you can open without a simulator.

An evaluation answers "did it work". This answers "what was it trying to do", which
is the question you are left with once the answer to the first one is no. Per step it
keeps what the policy was shown, what it asked for, what the IK made of that, and
where the arm actually went -- so a failure can be attributed to the policy, the IK,
or the scene, instead of to whichever of the three you looked at first.

Layout, one directory per rollout::

    meta.json        the scene, the instruction, the bins, the blocks, the arguments
    steps.jsonl      one JSON object per control step, appended as it happens
    frames/top/0000.jpg, frames/wrist/0000.jpg, ...

Plain JSON and JPEG on purpose. The traces outlive the branch that wrote them, and a
format you can `less` and `feh` is worth more than one that is 30% smaller and needs
this package importable to read at all.

Written incrementally: a rollout that crashes or is killed leaves a readable prefix,
which is usually the interesting part.

Cost: three JPEG encodes a step, ~10 ms against a 200 ms control period. Real but
small, and it lands in the step timings -- do not benchmark a traced rollout.
"""

import json
from pathlib import Path

import cv2
import numpy as np

# Step fields that are numpy arrays and want rounding rather than full float64 noise.
DECIMALS = 5


def _plain(value):
    """JSON-safe, and short enough to read: arrays become rounded lists."""
    if isinstance(value, np.ndarray):
        return [round(float(v), DECIMALS) for v in value.reshape(-1)]
    if isinstance(value, (np.floating, float)):
        return round(float(value), DECIMALS)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, dict):
        return {k: _plain(v) for k, v in value.items()}
    return value


class RolloutTrace:
    """Writer. One instance per rollout; ``close`` finishes the directory."""

    def __init__(self, directory, cameras=(), quality=80, frame_interval=1, pose_interval=5):
        self.directory = Path(directory)
        self.cameras = tuple(cameras)
        self.quality = quality
        self.frame_interval = max(1, int(frame_interval))
        self.pose_interval = max(1, int(pose_interval))
        # Set by the driver: a no-argument callable returning {name: [x, y, z]} for
        # whatever the task considers its objects. Kept out here because run_episode
        # has no idea what a bin is, and should not have to.
        self.poses = None
        self._steps = None

    def open(self, meta: dict) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        for camera in self.cameras:
            (self.directory / "frames" / camera).mkdir(parents=True, exist_ok=True)
        (self.directory / "meta.json").write_text(json.dumps(_plain(meta), indent=2))
        self._steps = (self.directory / "steps.jsonl").open("w")

    def step(self, index: int, record: dict, observation: dict | None = None) -> None:
        """One control step. ``observation`` supplies the camera images, if any."""
        if self._steps is None:
            return
        row = {"step": index, **_plain(record)}

        if observation is not None and index % self.frame_interval == 0:
            for camera in self.cameras:
                image = observation.get(camera)
                if image is None:
                    continue
                # The robot hands out RGB; cv2 writes BGR.
                cv2.imwrite(
                    str(self.directory / "frames" / camera / f"{index:04d}.jpg"),
                    cv2.cvtColor(image, cv2.COLOR_RGB2BGR),
                    [cv2.IMWRITE_JPEG_QUALITY, self.quality],
                )
            row["frame"] = index

        if self.poses is not None and index % self.pose_interval == 0:
            # One service round trip per polled step. At the default interval that is
            # 1 Hz against a 5 Hz loop, which the success check already pays.
            row["poses"] = _plain(self.poses())

        self._steps.write(json.dumps(row) + "\n")
        self._steps.flush()

    def close(self, summary: dict) -> None:
        if self._steps is not None:
            self._steps.close()
            self._steps = None
        path = self.directory / "meta.json"
        meta = json.loads(path.read_text())
        meta["summary"] = _plain(summary)
        path.write_text(json.dumps(meta, indent=2))


def load_trace(directory) -> dict:
    """Read one rollout back. No ROS, no simulator, no policy -- just the files."""
    directory = Path(directory)
    meta_path = directory / "meta.json"
    if not meta_path.is_file():
        raise SystemExit(f"{directory} has no meta.json; not a rollout trace.")
    steps = []
    steps_path = directory / "steps.jsonl"
    if steps_path.is_file():
        steps = [json.loads(line) for line in steps_path.read_text().splitlines() if line.strip()]
    return {"directory": directory, "meta": json.loads(meta_path.read_text()), "steps": steps}


def frame_path(trace: dict, camera: str, step: int):
    """The image for this camera at or before this step, or None.

    Frames may be subsampled (``--frame-interval``), so the viewer asks for a step and
    gets the most recent picture -- holding the last frame is what a video player does
    and beats blanking the panel between samples.
    """
    frames = trace["directory"] / "frames" / camera
    if not frames.is_dir():
        return None
    for index in range(step, -1, -1):
        candidate = frames / f"{index:04d}.jpg"
        if candidate.is_file():
            return candidate
    return None


def seed_number(path) -> int:
    """The seed a ``seed_0007`` directory holds; -1 for anything not named that way."""
    digits = Path(path).name.rsplit("_", 1)[-1]
    return int(digits) if digits.isdigit() else -1


def find_pairs(directory) -> list:
    """Every recorded seed under a run directory, in seed order.

    A run holds ``seed_0000/``, ``seed_0001/``, ...; one seed holds ``named/`` and
    ``flipped/``. The viewer should open either, so this returns the seed directories
    when there are any and the directory itself when there are not -- which is also
    what makes a single-pair path keep working.

    Sorted by the NUMBER, not the name: a run written without zero padding would
    otherwise put seed_10 between seed_1 and seed_2.
    """
    directory = Path(directory)
    if not directory.is_dir():
        raise SystemExit(f"{directory} is not a directory.")
    seeds = sorted((p for p in directory.glob("seed_*") if p.is_dir()), key=seed_number)
    return seeds or [directory]


# Preferred display order. debug_rollout writes left/ and right/; earlier traces used
# named/ and flipped/, and those still open.
TRIAL_ORDER = ("left", "right", "named", "flipped")


def load_pair(directory) -> dict:
    """A seed's trials, keyed by name; or the one rollout if this is a trial directory.

    Any subdirectory holding a meta.json counts, rather than two hardcoded names --
    the labelling changed once already, and a viewer that cannot open last week's
    traces is a viewer people stop trusting.
    """
    directory = Path(directory)
    found = {
        child.name: load_trace(child)
        for child in sorted(directory.iterdir())
        if child.is_dir() and (child / "meta.json").is_file()
    }
    if not found:
        return {directory.name: load_trace(directory)}
    return dict(
        sorted(
            found.items(),
            key=lambda item: (
                TRIAL_ORDER.index(item[0]) if item[0] in TRIAL_ORDER else len(TRIAL_ORDER),
                item[0],
            ),
        )
    )
