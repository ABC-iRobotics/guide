import itertools
import json

import numpy as np
from scipy.spatial.transform import Rotation as R

from guide_core.scene.scene_orchestrator import SceneOrchestrator
from guide_core.types.geometry import Point, Pose, Rotation
from guide_core.types.randomization import Categorical, UniformVec

# davinci_sim/build_scene.py: SurRoL board, 12 pegs (d 3 mm, 18 mm tall), triangular ring blocks.
PEG_H = 0.018
BLOCK_HALF_H = 0.0072  # a block resting on the board has its origin this far above the board top
BLOCKS = "/PegTransfer/Blocks"
BOARD = "/PegTransfer/Board"
COLOURS = ("red", "green", "blue", "yellow", "orange", "purple")
# The board is turned 90 deg about z: board (px, py) -> scene (-py, 0.72 + px). Pegs 0-5 (board
# y > 0) are at scene x < 0, PSM1's side (its base at x -0.13), which the ECM image shows on the
# RIGHT; pegs 6-11 are PSM2's, image left (test_scene pins this against the USD).
SIDE_PEGS = {"psm1": tuple(range(6)), "psm2": tuple(range(6, 12))}
IMAGE_SIDE = {"psm1": "right", "psm2": "left"}
# Seated: the block's centre within this of the peg's axis, and at most this above the board top
# (a block caught on the peg top sits a peg height higher). The hole (inradius 4.16 mm, corners 4.8) lets
# the 1.5 mm peg sit up to 2.7-3.3 mm off the block's centre (Isaac: seated blocks read 2.6-2.7 mm off);
# a block down beside the peg sits >= 6.3 mm off (outer face 4.85 mm + the peg's radius).
SEATED_XY = 0.0035
SEATED_Z = BLOCK_HALF_H + 0.003
BLOCK_YAW = 120.0  # deg; the triangular ring repeats every 120

PROCEDURE = "Transfer the block to the other side."


def peg(i: int) -> str:
    return f"{BOARD}/peg_{i:02d}"


def draw_plan(randomizer) -> dict:
    """The episode's discrete draws, all recorded: the direction, the block to move, which colour
    sits on each of the giver's six pegs (all six blocks start there), the target peg on the other
    side, and each block's yaw on its peg. The block's peg is where the layout put its colour."""
    direction = randomizer.draw("direction", Categorical(("psm1_to_psm2", "psm2_to_psm1")))
    giver, receiver = direction.split("_to_")
    colour = randomizer.draw("colour", Categorical(COLOURS))
    layout = list(randomizer.draw("layout", Categorical(tuple(itertools.permutations(COLOURS)))))
    target = randomizer.draw("target_peg", Categorical(SIDE_PEGS[receiver]))
    yaw = randomizer.draw("block_yaw", UniformVec(np.zeros(6), np.full(6, BLOCK_YAW)))
    side_from, side_to = IMAGE_SIDE[giver], IMAGE_SIDE[receiver]
    return {
        "task": PROCEDURE,
        "tasks": [f"Move the {colour} block from the {side_from} to the {side_to} side."],
        "direction": direction,
        "giver": giver,
        "receiver": receiver,
        "colour": colour,
        "block": f"{BLOCKS}/{colour}",
        "source_peg": peg(SIDE_PEGS[giver][layout.index(colour)]),
        "target_peg": peg(target),
        "side_from": side_from,
        "side_to": side_to,
        # placement: layout[k] on the giver's k-th peg at yaw[k] degrees
        "layout": layout,
        "source_pegs": [peg(i) for i in SIDE_PEGS[giver]],
        "yaw": [float(y) for y in yaw],
    }


def on_peg(peg_position, yaw_deg) -> Pose:
    """A block resting on the board around a peg (its world position: the cylinder's centre)."""
    position = np.asarray(peg_position, float) + [0.0, 0.0, BLOCK_HALF_H - PEG_H / 2]
    return Pose(Point(position), Rotation(R.from_euler("z", yaw_deg, degrees=True)))


def seated(block: Pose, peg_pose: Pose) -> bool:
    b, p = block.position.to_numpy(), peg_pose.position.to_numpy()
    return bool(np.hypot(*(b[:2] - p[:2])) <= SEATED_XY and b[2] <= p[2] - PEG_H / 2 + SEATED_Z)


class Scene(SceneOrchestrator):
    """Peg transfer: one PSM lifts the drawn block off its peg, hands it to the other PSM, which
    seats it on the drawn peg of its own side. randomize.yaml moves the board, the handover marker
    and the lights; the blocks then go onto the giver's pegs where the board put them."""

    def randomize(self, *, seed=None, inject=None, zone=None, arrangement=None, zones=None):
        self.randomize_instructions = []
        context = super().randomize(seed=seed, inject=inject, zone=zone, arrangement=arrangement, zones=zones)
        # The executor runs this next (_cmd_randomize_scene): a USD write, so PhysX teleports the blocks.
        poses = self._peg_positions(self.plan["source_pegs"])
        self.randomize_instructions = [{"cmd": "set_world_poses", "kwargs": {
            "prim_path": [f"/Scene_{self._scene_id}{BLOCKS}/{c}" for c in self.plan["layout"]],
            "pose": [on_peg(p, y) for p, y in zip(poses, self.plan["yaw"])],
        }}]
        return context

    def _peg_positions(self, pegs):
        """Where the pegs are now: Replicator has already moved the board."""
        from isaacsim.core.prims import XFormPrim

        paths = [f"/Scene_{self._scene_id}{p}" for p in pegs]
        positions, _ = XFormPrim(paths, reset_xform_properties=False).get_world_poses()
        return positions

    def randomize_preprocess(self, randomizer):
        self.plan = draw_plan(randomizer)
        self.task = PROCEDURE
        return randomizer

    def randomize_postprocess(self, result):
        return json.dumps(self.plan)

    def is_success_preprocess(self, instructions):
        poses, clash = instructions
        scene = f"/Scene_{self._scene_id}"
        block = scene + self.plan["block"]
        jaws = [f"{scene}{self._config['robots'][r]['path']}/psm_tool_gripper{k}_link"
                for r in ("psm1", "psm2") for k in (1, 2)]
        return [
            {**poses, "kwargs": {"prim_path": block}},
            {**poses, "kwargs": {"prim_path": scene + self.plan["target_peg"]}},
            *({**clash, "kwargs": {**clash["kwargs"], "prim_path": block, "scope": jaw}} for jaw in jaws),
        ]

    def is_success_postprocess(self, result):
        block, peg_pose, *touching = result
        ok = seated(block, peg_pose) and not any(touching)
        if not ok:  # which half failed: the block off its peg, or a jaw still on it
            b, p = block.position.to_numpy(), peg_pose.position.to_numpy()
            self._logger.info(f"Not a success: block {np.hypot(*(b[:2] - p[:2])) * 1e3:.1f} mm off the peg's axis, "
                              f"{(b[2] - p[2] + PEG_H / 2) * 1e3:.1f} mm over its foot; jaws touching {list(touching)}")
        return ok

    def reset_preprocess(self, instructions):
        return instructions

    def reset_postprocess(self, result):
        return True  # the Reset service answers a bool; the commands raise on failure

    def check_warmup(self):
        return True
