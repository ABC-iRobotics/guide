"""Record a pair of rollouts in full detail, for reading offline in replay_rollout.

Same experiment as ``probe_grounding`` -- one seed, one identical scene, told left
once and right once -- but it keeps everything instead
of a verdict. The probe tells you the arm follows the words; this tells you what the
arm and the policy were doing at step 63 when it closed on nothing.

Per rollout it writes what the policy was SHOWN (state, camera frames), what it ASKED
FOR (the executed action, and the whole queued chunk behind it), what the IK made of
that (target pose, joint solution, residual), and what actually HAPPENED (measured
joints, measured tool pose), plus the scene's own object positions. Attribution needs
all four: a tool in the wrong place because the policy aimed there and one in the
wrong place because the IK could not follow are the same measurement and opposite
bugs.

    python -m block_bin.debug_rollout \\
        --policy ~/models/smolvla_fr3_1_2_3_6_7_8/checkpoints/180000/pretrained_model \\
        --seeds 0-2 --zone 3 --seconds 40

    python -m block_bin.replay_rollout ~/eval_debug/<run>/seed_0000

Isaac with the camera topics and the description publisher, exactly as for an
evaluation::

    ros2 launch guide_core bringup.launch.py camera_topics:=true
    ros2 launch block_bin eval_pink.launch.py

Every eval_policy_pink flag applies. Rollouts here are NOT benchmarks: three JPEG
encodes and a pose query per step land in the step timings.

Output, one directory per run::

    seed_0000/left/{meta.json, steps.jsonl, frames/<cam>/NNNN.jpg}
    seed_0000/right/...

The pair is the unit deliberately, and it is symmetric: one rollout is told "in the
left bin" and the other "in the right bin", whichever side the randomizer drew. Both
see one identical scene, so any difference between them is the sentence and nothing
else -- the only way to read a conditioning effect off a policy that fails at the
task either way. The scene's own draw is kept in the metadata as ``scene_side``,
because IsSuccess only grades one of the two honestly.
"""

import json
from datetime import datetime
from pathlib import Path

from block_bin.eval_policy import parse_evaluation_args
from block_bin.eval_policy_pink import build_parser, run_episode, setup_evaluation
from block_bin.probe_grounding import (
    SIDES,
    attach_scene_clients,
    instruction_for,
    parse_seeds,
    scene_draw,
    where_is,
)
from block_bin.rollout_trace import RolloutTrace
from guide_msgs.srv import Pose as PoseSrv

# The blocks the scene randomizes, from block_bin.scene.Scene.colors. Three of them
# are disturbances; keeping all four is what shows the arm knocking one aside.
COLOURS = ("red", "yellow", "green", "blue")


def prim_position(robot, path: str):
    """World-frame [x, y, z] of one prim, or None if the stage has no such path."""
    response = robot.callService(robot.pose, PoseSrv.Request(path=path))
    if not response.success:
        return None
    return [response.pose.position.x, response.pose.position.y, response.pose.position.z]


def scene_objects(robot, scene_id: int) -> dict:
    """Every block and both bins, by name."""
    objects = {}
    for colour in COLOURS:
        position = prim_position(robot, f"/Scene_{scene_id}/blocks/{colour}_block")
        if position is not None:
            objects[f"{colour}_block"] = position
    for side, index in SIDES.items():
        position = prim_position(robot, f"/Scene_{scene_id}/bin_{index}")
        if position is not None:
            objects[f"{side}_bin"] = position
    return objects


def record_trial(evaluation, args, seed, zone, side, draw, directory) -> dict:
    """One rollout told to use ``side``, traced from first step to last."""
    robot = evaluation.robot
    task = instruction_for(draw["colour"], side)
    robot.node.get_logger().info(f'--- seed {seed} [{side} bin]: "{task}" ---')

    args.task = task
    trace = RolloutTrace(
        directory,
        cameras=args.cameras,
        quality=args.jpeg_quality,
        frame_interval=args.frame_interval,
        pose_interval=args.pose_interval,
    )
    trace.poses = lambda: scene_objects(robot, evaluation.scene_id)
    trace.open(
        {
            "seed": seed,
            "trial": side,
            "zone": zone,
            "instruction": task,
            "scene_task": draw["task"],
            "colour": draw["colour"],
            "asked_for": side,
            # Which side the randomizer drew. The pair does not depend on it -- both
            # sides are always run -- but IsSuccess only grades this one honestly.
            "scene_side": draw["side"],
            "target": draw["target"],
            "objects_at_start": scene_objects(robot, evaluation.scene_id),
            # PoseRequest answers in WORLD frame; the policy's state is in the SCENE
            # frame. Recording the scene's own world origin makes the trace
            # self-describing, so an analysis does not have to guess which frame a
            # column is in -- getting that wrong put a metre into every tool-to-block
            # distance the first time the base offset was corrected.
            "scene_origin": prim_position(robot, f"/Scene_{evaluation.scene_id}"),
            "policy": args.policy,
            "fps": args.fps,
            "n_action_steps": args.n_action_steps,
            "lead": args.lead,
            "action_scale": args.action_scale,
            # The viewer replays the queued plan through the same scale and
            # ceilings the executor used, so the drawn trajectory is what would
            # have happened, not a longer one the clamp would have cut back.
            "max_step": args.max_step,
            "max_rotation_step": args.max_rotation_step,
            "ik_tolerance": args.ik_tolerance,
            "base_offset": list(args.base_offset),
            "cameras": list(args.cameras),
            "recorded": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
    )

    try:
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
            trace=trace,
        )
    finally:
        # Closed even if the rollout threw: a trace whose summary is missing is still
        # readable, and the steps up to the fault are the ones worth looking at. The
        # final measurement is best-effort for the same reason -- losing a whole
        # traced rollout because a pose query failed at the end would be absurd.
        summary = {}
        try:
            placement = where_is(robot, evaluation.scene_id, draw["target"])
            summary = {
                "objects_at_end": scene_objects(robot, evaluation.scene_id),
                "reached": placement["reached"],
                "distance_to": placement["distance_to"],
            }
        except Exception as error:
            robot.node.get_logger().warn(f"Could not measure the final scene: {error}")
            summary = {"measurement_error": str(error)}
        trace.close(summary)
    return record


def run_name(policy: str) -> str:
    """<model>_<step> from a checkpoint path, for naming the output directory.

    LeRobot's layout is <model>/checkpoints/<step>/pretrained_model, so the model name
    is three levels up -- not one, which is the literal string "checkpoints" and would
    name every run on this machine the same thing.
    """
    parts = Path(policy).parts
    if len(parts) >= 4 and parts[-1] == "pretrained_model":
        return f"{parts[-4]}_{parts[-2]}"
    return Path(policy).name or "policy"


def main():
    parser = build_parser()
    parser.description = __doc__.splitlines()[0]
    parser.add_argument("--seeds", type=str, default="0-2", help="Seeds to pair: '0-2', '0,4'.")
    parser.add_argument(
        "--output",
        type=str,
        default="",
        help="Where the traces go (default ~/eval_debug/<model>_<timestamp>).",
    )
    parser.add_argument(
        "--cameras",
        type=str,
        default="top,base,wrist",
        help="Cameras to save frames from, or '' for none. The wrist view at the grasp "
        "step is usually the one that shows why the fingers closed on nothing.",
    )
    parser.add_argument("--jpeg-quality", type=int, default=80)
    parser.add_argument(
        "--frame-interval",
        type=int,
        default=1,
        help="Save a frame every N steps (default every step). Raise it if disk or the "
        "encode cost matters more than seeing every observation.",
    )
    parser.add_argument(
        "--pose-interval",
        type=int,
        default=5,
        help="Query the blocks and bins every N steps (default 5, i.e. 1 Hz at --fps 5). "
        "One service round trip per polled step.",
    )
    args = parse_evaluation_args(parser)
    args.record_path = True
    args.cameras = tuple(name.strip() for name in args.cameras.split(",") if name.strip())

    seeds = parse_seeds(args.seeds)
    zone = None
    if args.zone.strip():
        if not args.zone.strip().isdigit():
            raise SystemExit(f"--zone must be a single zone here (got {args.zone!r}).")
        zone = int(args.zone)

    output = Path(args.output).expanduser() if args.output else (
        Path.home() / "eval_debug" / f"{run_name(args.policy)}_"
        f"{datetime.now().strftime('%Y%m%d_%H%M')}"
    )
    output.mkdir(parents=True, exist_ok=True)
    print(
        f"Tracing {len(seeds)} seeds x 2 instructions = {2 * len(seeds)} rollouts into "
        f"{output}\nWorst case {2 * len(seeds) * args.seconds / 60:.0f} min of sim time."
    )

    evaluation = setup_evaluation(args)
    robot = evaluation.robot
    attach_scene_clients(robot, evaluation.sim_namespace)

    written = []
    try:
        for index, seed in enumerate(seeds, start=1):
            draw = scene_draw(robot, evaluation.scene_id, seed, zone)
            pair = output / f"seed_{seed:04d}"
            robot.node.get_logger().info(
                f"=== pair {index}/{len(seeds)}, seed {seed}: {draw['colour']} block, "
                f"left bin then right bin (scene drew {draw['side']}) ==="
            )

            for side in SIDES:
                record = record_trial(evaluation, args, seed, zone, side, draw, pair / side)
                print(
                    f"  seed {seed} told {side}: {record['steps']} steps, "
                    f"grasp at {record['grasp_step']}, success {record['success']}"
                )
            written.append(pair)
            print(f"  python -m block_bin.replay_rollout {pair}")

            if evaluation.control.stop_run.is_set():
                robot.node.get_logger().warn("Run ended early on request.")
                break
    except KeyboardInterrupt:
        robot.node.get_logger().info("Keyboard interrupt received. Exiting...")
    finally:
        (output / "meta.json").write_text(
            json.dumps(
                {
                    "policy": args.policy,
                    "seeds": seeds,
                    "zone": zone,
                    "seconds": args.seconds,
                    "pairs": [str(path) for path in written],
                },
                indent=2,
            )
        )
        print(f"\n{len(written)} pair(s) in {output}")
        robot.disconnect()


if __name__ == "__main__":
    main()
