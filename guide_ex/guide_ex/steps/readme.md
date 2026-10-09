# steps

The STEP layer of GUIDE-EX: one motion, gripper command or scene query each, carried out by
basic software (PRIMITIVE: MoveIt, the gripper action, the simulator's ROS services).

- `manipulation/cartesian_move.py`: `MoveToCartesianPose`, `MoveWithCartesianVelocity`
- `manipulation/joint_move.py`: `MoveToJointConfiguration`
- `end_effector/gripper_control.py`: `SetGripperState`
- `simulation/isaac/prim.py`: `GetPrimPose`, `GetPrimPoses`, `IsPrimClashing`
- `simulation/success.py`: `IsTaskSuccessful`
