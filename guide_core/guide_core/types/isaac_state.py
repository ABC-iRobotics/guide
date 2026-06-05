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


class IsaacStateErrorMessage(str, Enum):
    UNINITIALIZED = "Isaac Sim is not initialized. Call 'initialize()' before proceeding."
    INITIALIZING = "Simulation is not ready. Please wait until setup is completed."
    STOPPED = "No stage is loaded."
    LOADING = "An asset is loading, please wait until the loading is completed."
    READY = "Simulation is not running. Call 'start()' before proceeding."
    RUNNING = "Simulation is running. Please call 'pause()' or 'stop()' before proceeding."
    PAUSED = "Simulation is not running. Call 'start()' before proceeding."
    ERROR = "An error occurred. Call 'stop()' to clear it."
    SHUTTING_DOWN = "The system is shutting down."
