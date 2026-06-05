from enum import Enum


class SceneState(Enum):
    IDLE = 0
    PREPARATION = 1
    RECORDING = 2
    FINALIZING = 3
