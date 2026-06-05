import carb
from isaacsim.core.utils.nucleus import is_file
from isaacsim.core.utils.stage import add_reference_to_stage, is_stage_loading
from pxr import Usd

from guide_core.types.isaac_state import IsaacState

UNINITIALIZED = IsaacState.UNINITIALIZED
INITIALIZING = IsaacState.INITIALIZING
STOPPED = IsaacState.STOPPED
LOADING = IsaacState.LOADING
READY = IsaacState.READY
RUNNING = IsaacState.RUNNING
PAUSED = IsaacState.PAUSED
ERROR = IsaacState.ERROR
SHUTTING_DOWN = IsaacState.SHUTTING_DOWN


def _cmd_clear_world(self) -> None:
    assert self.state in [PAUSED, STOPPED]

    scene = self._world.scene

    scene.clear()
    scene.add_default_ground_plane()

    # self._scenes = {}


def _cmd_add_scene(self, stage_config=None, root: str = "/World") -> None:

    assert self.state not in [UNINITIALIZED, INITIALIZING, RUNNING, ERROR]

    self.state = LOADING
    try:
        if stage_config is None:
            stage_config = self.stage_config

        self.update(2)

        self._logger.debug("Opening USD stage...")
        # open_stage + wait loading + returns stage
        self.__setup_stage(stage_config, root)

    except Exception as e:
        self._logger.debug(f"Error in _cmd_open_stage: {e}")
        self.state = ERROR
        return

    self.state = READY


def _cmd_start(self) -> None:
    # ? self._sim_enabled.set()

    assert self.state in [READY, PAUSED, STOPPED]

    self._world.play()

    self.state = RUNNING


def _cmd_add_physics_callback(self, callback_id: str, callback_fn) -> None:
    if self._world:
        self._world.add_physics_callback(callback_id, callback_fn)


def _cmd_pause(self) -> None:
    # ? self._sim_enabled.clear()

    assert self.state is RUNNING

    self._world.pause()

    self.state = PAUSED


def _cmd_stop(self) -> None:
    assert self.state not in [UNINITIALIZED, INITIALIZING, LOADING, SHUTTING_DOWN]

    self._world.stop()

    self.state = STOPPED


def _cmd_shutdown(self) -> None:
    self._cmd_stop()
    # self._stop_evt.set()

    self.state = SHUTTING_DOWN

    self.simulation_app.close()
    # self._scenes = {}
    self._world = None
    self._stage = None

    self.state = UNINITIALIZED


def __setup_stage(self, stage_config, root) -> "Usd.Stage":

    if stage_config is None:
        raise ValueError("No world configuration found in init.yaml")

    # Get USD path
    print(stage_config)
    USD_PATH = stage_config.get("usd_path_absolute")
    if not USD_PATH:
        package_name = stage_config.get("package", "")
        assets_root_path = package_name
        USD_PATH = assets_root_path + stage_config["usd_path"]
    try:
        result = is_file(USD_PATH)
        self._logger.debug(f"USD file exists: {result}")
    except Exception:
        result = False

    # Reference USD stage
    if result:
        add_reference_to_stage(usd_path=USD_PATH, prim_path=root)
    else:
        carb.log_error(
            f"the usd path {USD_PATH} could not be opened, please make sure that {USD_PATH} is a valid usd file in {assets_root_path}"
        )
        self.state = ERROR
        return

    # Mandatory waiting
    self._logger.debug("Loading stage...")
    self.update(10)

    while is_stage_loading():
        self.update()
        self._logger.debug("Loading")
    self._logger.debug("Loading Complete")

    self.update()
    while is_stage_loading():
        self.update()

    self._logger.debug("Stage is ready.")
