from enum import Enum


class IsaacState(Enum):
    UNINITIALIZED = 0
    INITIALIZING = 1
    STOPPED = 2
    LOADING = 3
    READY = 4
    RUNNING = 5
    PAUSED = 6
    ERROR = 7
    SHUTTING_DOWN = 8

