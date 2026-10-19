"""GUIDE's motion layer: Pink IK planning and OSCBF control. No Isaac, no ROS.

Importing it sets JAX's environment for the whole process (CPU, float64, single-threaded Eigen; setdefault,
so values set before the import win): a JAX policy in the same process would run on CPU in float64 too.
"""

import os

# JAX on GPU is out of scope (S2b spec 1); the CPU wheel otherwise warns at import.
os.environ.setdefault("JAX_PLATFORMS", "cpu")
# OSCBF and cbfpy need float64 (float32 states raise in cbfpy's jvp); oscbf's own __init__ sets it too.
os.environ.setdefault("JAX_ENABLE_X64", "1")
# ponytail: XLA's CPU kernels without Eigen's thread pool, measured 15-20 % faster next to PhysX's
# workers (S2b report); unset it here if the controller ever runs alone on a big machine.
os.environ.setdefault("XLA_FLAGS", "--xla_cpu_multi_thread_eigen=false")
