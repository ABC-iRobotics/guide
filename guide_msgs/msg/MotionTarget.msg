# One motion target for a robot's controller (S2b), published at 20 Hz on
# /Sim_<s>/Scene_<i>/<robot>/motion_target. Metric everywhere: metres, radians.
builtin_interfaces/Time stamp          # sim time
uint8 MODE_POSE=0
uint8 MODE_TWIST=1
uint8 MODE_JOINTS=2
uint8 mode
string frame_id                        # the robot base link
geometry_msgs/Pose pose
geometry_msgs/Twist twist              # linear m/s, angular rad/s
float64[] joints                       # rad, the arm joints
# NaN = leave the gripper; publishers must set it (the default 0.0 closes the hand)
float64 gripper_width                  # metres, absolute
