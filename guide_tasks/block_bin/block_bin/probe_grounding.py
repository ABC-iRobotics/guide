"""Does the policy read the instruction, or has it memorised one routine?

A success rate cannot tell those apart. A policy that always puts the block in the
right bin scores ~50% on a task whose bin is drawn 50/50, and every one of those wins
is real -- it did put the block in the right bin, and it was asked to. The failures
look like clumsiness. Nothing in the number says the words were never read.

This is the paired test that separates them. The SAME seed is randomized twice, so
both rollouts see one identical scene -- same block colours, same block positions,
same bins, same everything the camera can see. The only difference is the sentence,
and the pair is symmetric: one trial is told LEFT and the other RIGHT, whichever side
the scene itself happened to draw.

    trial "left"   "Put the blue block in the left bin."
    trial "right"  "Put the blue block in the right bin."

Labelling them by the side they ask for rather than by agreement with the scene's own
draw is what makes the pair readable: every row of the report means the same thing,
and no verdict needs to know which way the randomizer went.

Two instruments read the result, because the strong one stops working exactly when
the policy is bad enough to be worth probing.

BLOCK -- which bin the block PHYSICALLY ended in. The criterion that counts.

    followed    each trial's block ended in the bin that trial asked for
    ignored     the block went to the SAME bin both times -- the words are decoration
    incomplete  the block reached no bin in one or both trials

ARM -- where the TOOL travelled, from its recorded path. A policy that never grasps
leaves every pair `incomplete` and the block instrument says nothing, while the arm
may still be visibly driving toward the instructed bin empty-handed. That is
the whole question when a policy reaches, closes on air, and carries on to a bin:
the language can be perfectly grounded and the manipulation still broken, and only
the arm shows it.

    followed    the tool leaned toward the bin each trial asked for, by
                --arm-threshold or more of differential lean
    opposed     it leaned the other way in both
    same        it moved identically whichever bin was asked for -- the words are unused

Both are paired and differential. Chance is 0%, not 50%: an ungrounded policy does
the same thing twice on the same scene, whatever bias it has. For the arm that also
removes the resting-position bias -- home is not equidistant from the two bins, so
only the CHANGE in lean between the two trials means anything.

The arm measure is also computed over the steps BEFORE the policy first asks to close
the gripper. Committing to a bin while the fingers are still empty is the premature-
transit signature, and that number states it directly.

Grading is deliberately NOT IsSuccess. The scene rebuilds its own criterion from its
own draw every episode (Scene.is_success_preprocess reads self.c/self.s), so on the
trial that contradicts the draw IsSuccess grades the bin nobody asked for, and a
perfectly executed rollout scores as a miss. Instead both trials are graded by the
same instrument -- where the block is -- read from the simulator through
CollisionRequest against each bin. On whichever trial matches the scene's draw
IsSuccess *is* valid, so the two graders are compared there and any disagreement is
reported; that is the instrument's own calibration check.

Running it -- Isaac WITH the camera topics, and the description publisher, exactly as
for an evaluation::

    ros2 launch guide_core bringup.launch.py camera_topics:=true
    ros2 launch block_bin eval_pink.launch.py

    python -m block_bin.probe_grounding \\
        --policy ~/models/smolvla_fr3_1_2_3_6_7_8/checkpoints/180000/pretrained_model \\
        --seeds 0-9 --zone 3 --results ~/eval_sweeps/grounding_180000.jsonl

Every eval_policy_pink flag applies (--fps, --n-action-steps, --seconds, --lead ...);
the probe adds --seeds and inherits the rest, so it drives the identical stack an
evaluation does. ``--seed-base`` is ignored: --seeds names the seeds.

Read the result with the completion rate next to it. A policy that never places the
block produces no complete pairs, and "0 of 0 followed" is not evidence of anything
-- the probe says so rather than printing a 0%. Grounding is only measurable on a
policy that can do the task at all.
"""

import json
import time

import numpy as np

from block_bin.eval_policy import SETTLE_SECONDS, parse_evaluation_args, sleep_sim
from block_bin.eval_policy_pink import (
    append_record,
    build_parser,
    run_episode,
    setup_evaluation,
)
from guide_msgs.srv import Collision, Pose as PoseSrv, Randomize

# Scene.is_success_preprocess grades containment in ``bin_{0 if self.s == 'left'
# else 1}``, so bin_0 IS left. Read it off that line, never off the bins' y in the
# stage -- the bins are randomized, and "left" is a name here, not a measurement.
SIDES = {"left": 0, "right": 1}


def parse_seeds(spec: str) -> list[int]:
    """'0-9', '0,3,7' or '0-4,10' -> the seeds to pair, in the order given."""
    seeds: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part.lstrip("-"):
            lo, _, hi = part.partition("-")
            start, end = int(lo), int(hi)
            if end < start:
                raise SystemExit(f"--seeds range '{part}' counts backwards.")
            seeds.extend(range(start, end + 1))
        else:
            seeds.append(int(part))
    if not seeds:
        raise SystemExit("--seeds is empty; try '0-9'.")
    return seeds


def instruction_for(colour: str, side: str) -> str:
    """The sentence the scene would have produced for this colour and bin.

    Mirrors Scene.randomize_preprocess. Duplicated on purpose -- the probe has to be
    able to write a sentence the scene never drew -- and checked against the scene's
    own string on every draw, so a wording change over in the task fails here
    loudly instead of quietly testing a sentence no policy was ever trained on.
    """
    return f"Put the {colour} block in the {side} bin."


def scene_draw(robot, scene_id: int, seed: int, zone) -> dict:
    """Randomize at this seed and read back what was drawn.

    Free to call: the draw is seeded, so re-randomizing at the same seed inside one
    simulator session reproduces this exact layout, which is what both trials then do.
    """
    response = robot.callService(
        robot.randomize,
        Randomize.Request(
            id=scene_id,
            use_zone=zone is not None,
            zone=zone or 0,
            use_seed=True,
            seed=int(seed),
        ),
    )
    drawn = json.loads(response.message)
    colour = drawn["target"].rsplit("/", 1)[-1].removesuffix("_block")
    side = next(name for name, index in SIDES.items() if drawn["goal"] == f"/bin_{index}")
    expected = instruction_for(colour, side)
    if drawn["task"] != expected:
        raise SystemExit(
            f"The scene's instruction is {drawn['task']!r} but this probe builds "
            f"{expected!r} for the same draw. Scene.randomize_preprocess has changed "
            f"its wording; update instruction_for to match before trusting a verdict."
        )
    return {"colour": colour, "side": side, "target": drawn["target"], "task": drawn["task"]}


def attach_scene_clients(robot, sim_namespace: str) -> None:
    """Give the robot the two service clients the scene queries here need.

    Lives beside where_is/bin_centres rather than in each caller's main: they are the
    functions that reach for robot.collision and robot.pose, so a caller that imports
    one of them and forgets a client gets an AttributeError halfway through a rollout
    it has already paid for. Idempotent.
    """
    if getattr(robot, "collision", None) is None:
        robot.collision = robot.node.create_client(
            srv_type=Collision,
            srv_name=f"{sim_namespace}/CollisionRequest",
            callback_group=robot._reentrant_callback_group,
        )
    if getattr(robot, "pose", None) is None:
        robot.pose = robot.node.create_client(
            srv_type=PoseSrv,
            srv_name=f"{sim_namespace}/PoseRequest",
            callback_group=robot._reentrant_callback_group,
        )


def bin_centres(robot, scene_id: int) -> dict:
    """XY centre of each bin, read from the stage.

    The bins are fixed in randomize.yaml for zoned generation, but read rather than
    hardcoded: the config says "uncomment both bins to restore the original free-bin
    task", and a probe that had memorised their positions would then quietly measure
    against furniture that had moved.
    """
    centres = {}
    for side, index in SIDES.items():
        pose = robot.callService(
            robot.pose, PoseSrv.Request(path=f"/Scene_{scene_id}/bin_{index}")
        ).pose
        centres[side] = np.array([pose.position.x, pose.position.y])
    return centres


def lean_of(path: list, centres: dict) -> dict | None:
    """How close the TOOL got to each bin, and which way it leaned.

    ``lean = closest_to_left - closest_to_right``: negative leans left, positive
    leans right. The absolute value is meaningless on its own -- home is not
    equidistant from the two bins, so an arm that never moved still has a lean. Only
    the DIFFERENCE between the two trials of a pair means anything, and that is what
    arm_response takes, which is why this returns the raw number and no verdict.
    """
    if not path:
        return None
    points = np.asarray(path, dtype=float)[:, :2]
    closest = {
        side: round(float(np.min(np.linalg.norm(points - centre, axis=1))), 3)
        for side, centre in centres.items()
    }
    return {"closest": closest, "lean": round(closest["left"] - closest["right"], 3)}


def arm_response(left: dict | None, right: dict | None, threshold: float):
    """Did the tool swing toward the bin each instruction named? -> (verdict, metres).

    ``lean`` is negative toward the left bin, so following means the left-instructed
    trial leans left and the right-instructed one leans right: ``lean(right) -
    lean(left) > 0``.

    Paired and differential on purpose. Asking "which bin was the arm nearer" would
    answer "the left one" for an arm that never left home, because home is nearer the
    left bin -- so the measure is how the lean CHANGED when the instruction changed,
    on a scene that did not change at all. An arm ignoring the words leans identically
    both times and scores 0 whatever its resting bias.

    This is the instrument that still works when the grasp fails: the block never
    moves, so where_is says "neither" every time, while the arm may still be visibly
    driving toward the instructed bin empty-handed.
    """
    if left is None or right is None:
        return "unmeasured", 0.0
    shift = round(right["lean"] - left["lean"], 3)
    if shift > threshold:
        return "followed", shift
    if shift < -threshold:
        return "opposed", shift
    return "same", shift


def where_is(robot, scene_id: int, target: str) -> dict:
    """Which bin the target block is in, straight from the simulator's own geometry.

    CollisionRequest is the sim's bounding-box overlap test against each bin, so this
    needs no hardcoded bin extents and follows the bins wherever the randomizer put
    them. Overlap is a weaker claim than the containment IsSuccess makes -- a block
    balanced on the rim counts here -- which is the right way round for this question:
    the probe is asking which bin the arm CHOSE, not whether the placement was tidy.
    """
    prim = f"/Scene_{scene_id}{target}"
    overlaps = {
        side: bool(
            robot.callService(
                robot.collision,
                Collision.Request(prim1=prim, prim2=f"/Scene_{scene_id}/bin_{index}"),
            ).collision
        )
        for side, index in SIDES.items()
    }
    hit = [side for side, touching in overlaps.items() if touching]

    block = robot.callService(robot.pose, PoseSrv.Request(path=prim))
    position = np.array([block.pose.position.x, block.pose.position.y, block.pose.position.z])
    distances = {
        side: round(float(np.linalg.norm(position[:2] - centre)), 3)
        for side, centre in bin_centres(robot, scene_id).items()
    }

    if len(hit) == 1:
        reached = hit[0]
    elif len(hit) == 2:
        # Both bounding boxes claim it. Not a placement -- the block is wedged between
        # them or still in the gripper above them -- so fall back to the nearer centre
        # rather than guessing, and let the recorded distances show how close it was.
        reached = min(distances, key=distances.get)
    else:
        reached = "neither"

    return {
        "reached": reached,
        "overlaps": overlaps,
        "distance_to": distances,
        "position": [round(float(v), 4) for v in position],
    }


def verdict(left_reached: str, right_reached: str) -> str:
    """One pair of trials -> followed / ignored / incomplete.

    The pair is the two INSTRUCTIONS -- "in the left bin" and "in the right bin" -- on
    one identical scene, so following is simply the block ending in the bin each one
    named. Nothing here depends on which side the scene happened to draw, which is why
    the trials are labelled by the side they ask for rather than relative to the draw.
    """
    if left_reached == "neither" or right_reached == "neither":
        return "incomplete"
    if left_reached == "left" and right_reached == "right":
        return "followed"
    if left_reached == right_reached:
        return "ignored"
    # Both bins reached, but not the pair that was asked for: it moved WITH the words
    # and got both backwards. Rare, and worth its own name -- averaging it into
    # "ignored" would hide a sign error in the instruction encoding.
    return "inverted"


def run_trial(evaluation, args, seed: int, zone, side: str, draw: dict, centres) -> dict:
    """One rollout on the seed's scene told to use ``side``, plus where the block ended."""
    robot = evaluation.robot
    task = instruction_for(draw["colour"], side)
    robot.node.get_logger().info(f'--- seed {seed} [{side} bin]: "{task}" ---')

    args.task = task
    record = run_episode(
        robot,
        evaluation.scene_id,
        zone,
        evaluation.policy,
        evaluation.processors,
        evaluation.device,
        evaluation.control,
        evaluation.ik,
        args,
        seed=seed,
    )

    # run_episode short-circuits its own settle when the loop already saw a success,
    # so settle here too -- the block may still be falling out of the gripper.
    sleep_sim(robot, SETTLE_SECONDS)
    placement = where_is(robot, evaluation.scene_id, draw["target"])

    path = record.get("eef_path", [])
    lean = lean_of(path, centres)
    # The same measure restricted to before the policy first asked to close, which is
    # the "moves to the side prematurely" hypothesis stated as a number: an arm already
    # committed to a bin while it still has nothing in its fingers.
    grasp_step = record.get("grasp_step")
    before_grasp = lean_of(path[:grasp_step] if grasp_step else path, centres)

    trial = {
        "seed": seed,
        "trial": side,
        "instruction": task,
        "colour": draw["colour"],
        "asked_for": side,
        # Which side the scene itself drew. Only bookkeeping for the probe -- it grades
        # by where the block went -- but it says which of the two trials IsSuccess can
        # legitimately be compared against.
        "scene_side": draw["side"],
        "reached": placement["reached"],
        "lean": lean,
        "lean_before_grasp": before_grasp,
        "grasp_step": grasp_step,
        "distance_to": placement["distance_to"],
        "overlaps": placement["overlaps"],
        "block_position": placement["position"],
        "is_success": record["success"],
        "steps": record["steps"],
        "travelled": record["travelled"],
        "max_home_error": record["max_home_error"],
        "grip_measured_min": record["grip_measured_min"],
        "unreachable": record["unreachable"],
        "held": record["held"],
        "time": time.time(),
    }

    if side == draw["side"]:
        # The one trial where IsSuccess is valid, so the two graders can be compared.
        # They are not the same test -- IsSuccess demands containment, this asks for
        # overlap -- so a success with reached != the drawn side is a real contradiction,
        # while the reverse is just the looser test being looser.
        trial["grader_agrees"] = not (record["success"] and placement["reached"] != draw["side"])
        if not trial["grader_agrees"]:
            robot.node.get_logger().warn(
                f"seed {seed}: IsSuccess says the task was completed but the block "
                f"overlaps {placement['reached']!r}, not {draw['side']!r}. The probe's "
                f"instrument and the scene's disagree -- do not trust this pair."
            )

    if args.results:
        append_record(args.results, {"policy": args.policy, **trial})
    return trial


def report(pairs: list[dict], threshold: float) -> None:
    """Both instruments, with the one that can still see this run leading."""
    print(
        "\n seed  colour   told left -> / told right ->    arm shift   arm         "
        "pre-grasp  grasp steps"
    )
    for pair in pairs:
        print(
            f" {pair['seed']:>4}  {pair['colour']:>6}   "
            f"{pair['left_reached']:>9} / {pair['right_reached']:<12}"
            f"{pair['arm_shift']:>+8.3f}   {pair['arm']:<11}"
            f"{pair['arm_shift_before_grasp']:>+7.3f}    {pair['grasped']}"
        )

    blocks = {name: 0 for name in ("followed", "ignored", "inverted", "incomplete")}
    arms = {name: 0 for name in ("followed", "opposed", "same", "unmeasured")}
    for pair in pairs:
        blocks[pair["verdict"]] += 1
        arms[pair["arm"]] += 1
    placed = len(pairs) - blocks["incomplete"]
    swung = arms["followed"] + arms["opposed"] + arms["same"]

    print(
        f"\nBlock  : {blocks['followed']} followed, {blocks['ignored']} ignored, "
        f"{blocks['inverted']} inverted, {blocks['incomplete']} incomplete."
        f"\nArm    : {arms['followed']} followed, {arms['opposed']} opposed, "
        f"{arms['same']} unmoved, {arms['unmeasured']} unmeasured."
    )

    if placed:
        rate = blocks["followed"] / placed
        print(
            f"\nGROUNDING (block placement, the criterion that counts): "
            f"{blocks['followed']}/{placed} pairs followed the instruction ({rate:.0%})."
        )
        if rate <= 0.2:
            print("The bin in the sentence is not reaching the action.")
        elif rate >= 0.8:
            print("The instruction moves the arm, and the block ends where it was told.")
    else:
        print(
            "\nThe block reached a bin in no trial of any pair, so placement cannot answer\n"
            "the question. Falling back to where the ARM went, which is a weaker claim --\n"
            "it shows the instruction reaching the motion, not the task being done."
        )

    if not swung:
        print("Nothing to fall back on either: no tool path was recorded.")
        return

    if arms["followed"] > arms["same"] + arms["opposed"]:
        print(
            f"\nARM RESPONSE: {arms['followed']}/{swung} pairs swung toward the bin each\n"
            f"instruction named, by {threshold:g} m or more of differential lean. The language\n"
            f"IS grounded -- the policy hears which bin and drives that way. What it does\n"
            f"not do is take the block with it, so this is a manipulation failure wearing\n"
            f"a grounding failure's clothes."
        )
        early = sum(1 for p in pairs if p["arm_before_grasp"] == "followed")
        never = sum(1 for p in pairs for step in p["grasped"] if step is None)
        print(
            f"Of those, {early}/{swung} had already committed to the instructed side BEFORE\n"
            f"policy first asked to close the gripper"
            + (f"; {never}/{2 * len(pairs)} rollouts never asked to close at all." if never
               else ". Every rollout did ask to close at some point.")
        )
        print(
            "That is the premature-transit signature: the bin phase starts while the\n"
            "fingers are still empty. Shorten the open-loop horizon (--n-action-steps)\n"
            "so the policy has to look again before it commits."
        )
    elif arms["same"] >= swung / 2:
        print(
            f"\nARM RESPONSE: {arms['same']}/{swung} pairs moved the SAME way whichever bin\n"
            f"was asked for. The instruction is not reaching the motion at all -- not a\n"
            f"grasp problem hiding a working policy, the words are not being used."
        )
    else:
        print(
            f"\nARM RESPONSE: mixed ({arms['followed']} followed, {arms['opposed']} opposed,\n"
            f"{arms['same']} unmoved). Too noisy to call; rerun with more seeds."
        )

    disagreed = [pair for pair in pairs if not pair.get("grader_agrees", True)]
    if disagreed:
        print(
            f"\nWARNING: on {len(disagreed)} pair(s) IsSuccess and the bin measurement "
            f"disagreed on the trial the scene drew (seeds {[p['seed'] for p in disagreed]}). "
            f"The instrument is not calibrated on those; exclude them."
        )


def main():
    parser = build_parser()
    # Inherited from eval_policy_pink, whose first line describes a plain evaluation.
    parser.description = __doc__.splitlines()[0]
    parser.add_argument(
        "--seeds",
        type=str,
        default="0-9",
        help="Seeds to pair: '0-9', '0,3,7' or '0-4,10'. Each seed is randomized twice "
        "-- identical scene, two instructions -- so this is 2N rollouts. Reproducible "
        "within one simulator session.",
    )
    parser.add_argument(
        "--arm-threshold",
        type=float,
        default=0.05,
        help="Metres of differential lean below which the arm is called unmoved by the "
        "instruction (default 0.05). The bins are ~0.9 m apart, so a real swing to the "
        "instructed side is tens of centimetres; a noise floor, not a decision point.",
    )
    args = parse_evaluation_args(parser)
    # Not optional here: the arm measurement IS the path.
    args.record_path = True

    seeds = parse_seeds(args.seeds)
    zone = None
    if args.zone.strip():
        if not args.zone.strip().isdigit():
            raise SystemExit(
                f"--zone must be a single zone here (got {args.zone!r}). Both trials of a "
                f"pair have to be the same scene, so the probe cannot spread a plan over "
                f"several zones the way an evaluation does."
            )
        zone = int(args.zone)

    print(
        f"Grounding probe: {len(seeds)} seeds x 2 instructions = {2 * len(seeds)} rollouts, "
        f"{'zone ' + str(zone) if zone is not None else 'unrestricted placement'}.\n"
        f"Worst case {2 * len(seeds) * args.seconds / 60:.0f} min of sim time."
    )

    evaluation = setup_evaluation(args)
    robot = evaluation.robot
    attach_scene_clients(robot, evaluation.sim_namespace)

    pairs: list[dict] = []
    try:
        for index, seed in enumerate(seeds, start=1):
            draw = scene_draw(robot, evaluation.scene_id, seed, zone)
            robot.node.get_logger().info(
                f"=== pair {index}/{len(seeds)}, seed {seed}: {draw['colour']} block, "
                f"left bin then right bin (scene drew {draw['side']}) ==="
            )

            centres = bin_centres(robot, evaluation.scene_id)
            left = run_trial(evaluation, args, seed, zone, "left", draw, centres)
            right = run_trial(evaluation, args, seed, zone, "right", draw, centres)

            arm, shift = arm_response(left["lean"], right["lean"], args.arm_threshold)
            early, early_shift = arm_response(
                left["lean_before_grasp"], right["lean_before_grasp"], args.arm_threshold
            )
            pairs.append(
                {
                    "seed": seed,
                    "colour": draw["colour"],
                    "scene_side": draw["side"],
                    "left_reached": left["reached"],
                    "right_reached": right["reached"],
                    "grader_agrees": (left if draw["side"] == "left" else right).get(
                        "grader_agrees", True
                    ),
                    "verdict": verdict(left["reached"], right["reached"]),
                    "arm": arm,
                    "arm_shift": shift,
                    "arm_before_grasp": early,
                    "arm_shift_before_grasp": early_shift,
                    "grasped": [left["grasp_step"], right["grasp_step"]],
                }
            )
            print(f"  pair {index}: block {pairs[-1]['verdict']}, arm {arm} ({shift:+.3f} m)")

            if evaluation.control.stop_run.is_set():
                robot.node.get_logger().warn("Probe ended early on request.")
                break
    except KeyboardInterrupt:
        robot.node.get_logger().info("Keyboard interrupt received. Exiting...")
    finally:
        if pairs:
            report(pairs, args.arm_threshold)
        else:
            print("\nNo pair finished, so there is nothing to report.")
        robot.disconnect()


if __name__ == "__main__":
    main()
