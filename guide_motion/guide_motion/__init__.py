"""GUIDE's motion layer: Pink IK planning and OSCBF control. No Isaac, no ROS."""

import os

# JAX on GPU is out of scope (S2b spec 1); the CPU wheel otherwise warns at import.
os.environ.setdefault("JAX_PLATFORMS", "cpu")
# OSCBF and cbfpy need float64 (float32 states raise in cbfpy's jvp); oscbf's own __init__ sets it too.
os.environ.setdefault("JAX_ENABLE_X64", "1")
# ponytail: one XLA thread per call, measured 15-20 % faster next to PhysX's workers
# (S2b report); unset it here if the controller ever runs alone on a big machine.
os.environ.setdefault("XLA_FLAGS", "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1")
