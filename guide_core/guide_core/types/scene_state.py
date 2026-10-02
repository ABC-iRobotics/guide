from enum import Enum


class SceneState(Enum):
    IDLE = 0
    PREPARATION = 1
    RECORDING = 2
    FINALIZING = 3
    # Episode open, nothing captured. SceneManager.step has no branch for it.
    PAUSED = 4
