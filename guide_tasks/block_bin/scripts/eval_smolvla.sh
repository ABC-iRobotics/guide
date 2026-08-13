#!/usr/bin/env bash
# Evaluate a LIBERO-style SmolVLA checkpoint on block_bin through MoveIt Servo.
#
# Defaults are read off /home/user/models/smolvla_fr3_07_27/checkpoints/005000:
#   action            shape [7]  -> forward EEF pose delta, gripper absolute in metres
#   observation.state shape [8]  -> dims 7 and 8 share every statistic, i.e. the
#                                   gripper repeated, so this is the --libero-state
#                                   layout and NOT the 7 joints + gripper one
#   chunk_size 50, n_action_steps 50, use_amp false, num_steps 10
#
# Override anything from the environment:
#   POLICY=~/models/other/checkpoints/010000/pretrained_model ./eval_smolvla.sh
#
# This is step 3 of 3. Run the other two first, in their own terminals:
#
#   1. Isaac + the scene. config/init.yaml needs `publish_camera_topics: true`,
#      otherwise the policy gets no images and the run dies on the first step.
#
#   2. MoveIt + MoveIt Servo, WITHOUT the demonstration solver:
#        ros2 launch block_bin eval_servo.launch.py
#
# Abort a rollout that has clearly missed, from a fourth terminal:
#   ros2 service call /Sim_0/Scene_0/stop_episode std_srvs/srv/SetBool "{data: false}"
#   (data: true ends the whole evaluation)

set -euo pipefail

POLICY=${POLICY:-/home/user/models/smolvla_fr3_07_27/checkpoints/005000/pretrained_model}
NAMESPACE=${NAMESPACE:-/Sim_0/Scene_0}
EPISODES=${EPISODES:-20}
SECONDS_PER_EPISODE=${SECONDS_PER_EPISODE:-60}

# The checkpoint's own n_action_steps is 50 -- one observation would drive 50 blind
# steps, i.e. 10 s at 5 Hz, so the grasp would happen on a ten-second-old view of the
# block. 10 makes it look again before closing the gripper.
HORIZON=${HORIZON:-10}

# A chunk costs ~0.5 s to predict (use_amp is off and num_steps is 10 in this config),
# i.e. 2-3 control periods, so the next one has to start with at least that many
# actions still queued or the arm stalls at every chunk boundary. 0 = old blocking
# behaviour, if the overlapped path ever misbehaves.
LEAD=${LEAD:-3}

# The demonstrations were sampled every 12 world steps at step_freq 60, i.e. 5 Hz of
# SIM time. This is also the divisor that turns each delta into a velocity for servo,
# so changing it rescales every motion -- leave it unless the dataset changed.
FPS=${FPS:-5.0}

# Keep inference off the GPU Isaac is rendering on. Set to cuda:0 on a single-GPU box.
DEVICE=${DEVICE:-cuda:1}

# Nudge if the policy consistently stops short of / past the block. 1.0 = as trained.
ACTION_SCALE=${ACTION_SCALE:-1.0}

# Blank = unrestricted placement. 'all' = every zone (--episodes then means per zone),
# '2,16' = only those zones, '2:4,16:10' = per-zone counts.
ZONE=${ZONE:-}

VENV_PYTHON=${VENV_PYTHON:-$HOME/ros2_ws/.venv/bin/python}

# ROS's setup.bash reads unset variables, so -u has to come off around it.
set +u
source /opt/ros/jazzy/setup.bash
source "$HOME/ros2_ws/install/setup.bash"
set -u

# Same-host DDS discovery needs the localhost cyclonedds config on this machine, or
# this process cannot see the simulator's services.
CDDS=$HOME/ros2_ws/install/guide_core/share/guide_core/config/cyclonedds_localhost.xml
[ -f "$CDDS" ] && export CYCLONEDDS_URI="file://$CDDS"

exec "$VENV_PYTHON" -m block_bin.eval_policy_servo \
    --namespace "$NAMESPACE" \
    --policy "$POLICY" \
    --episodes "$EPISODES" \
    --seconds "$SECONDS_PER_EPISODE" \
    --fps "$FPS" \
    --n-action-steps "$HORIZON" \
    --lead "$LEAD" \
    --state libero \
    --action-scale "$ACTION_SCALE" \
    --device "$DEVICE" \
    ${ZONE:+--zone "$ZONE"}
