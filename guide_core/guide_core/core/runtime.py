"""Isaac Sim runtime wrapper.

This package is unit-tested in environments where Isaac Sim is not installed.
Therefore Isaac-specific imports are made optional and only used at runtime.
"""

from __future__ import annotations

import inspect
import logging
import time
from dataclasses import dataclass
from queue import Empty, Queue
from types import ModuleType
from typing import Any, Dict, Literal, Optional, Tuple

from guide_core.types.isaac_state import IsaacState

try:
    # Isaac Sim runtime
    from isaacsim import SimulationApp  # type: ignore
except Exception:  # pragma: no cover
    SimulationApp = None  # type: ignore

import importlib

import yaml

try:
    from ament_index_python.packages import get_package_share_directory  # type: ignore
except Exception:  # pragma: no cover

    def get_package_share_directory(_: str) -> str:  # type: ignore
        raise RuntimeError(
            "ament_index_python is not available. Provide an explicit config path or install ROS2 deps."
        )


UNINITIALIZED = IsaacState.UNINITIALIZED
INITIALIZING = IsaacState.INITIALIZING
STOPPED = IsaacState.STOPPED
LOADING = IsaacState.LOADING
READY = IsaacState.READY
RUNNING = IsaacState.RUNNING
PAUSED = IsaacState.PAUSED
ERROR = IsaacState.ERROR
SHUTTING_DOWN = IsaacState.SHUTTING_DOWN


Kind = Literal["module", "class", "function", "other"]


def import_and_bind(
    target: str,
    *,
    alias: Optional[str] = None,
    namespace: Optional[Dict[str, Any]] = None,
    require: Optional[Kind] = None,
) -> Tuple[Any, Kind, str]:
    """
    Dynamically import a module or a symbol (class/function/other) and bind it
    into a given global namespace.

    target:
    - "pkg.module"            -> imports a module
    - "pkg.module:Symbol"     -> imports a symbol from a module (explicit form)
    - "pkg.module.Symbol"     -> imports a symbol if module import fails

    alias:
    - name under which the object will be bound in the namespace
        (default: natural name of the object)

    namespace:
    - dictionary to bind into (typically globals()).
        Must be passed explicitly for clarity.

    require:
    - optional type constraint: "module" | "class" | "function" | "other"
    """
    if not target or not isinstance(target, str):
        raise ValueError("target must be a non-empty string")

    # Namespace where the imported object will be bound
    ns = namespace if namespace is not None else globals()

    obj: Any
    kind: Kind

    # 1) Two supported syntaxes:
    #    - "module:Symbol"  -> explicit attribute import
    #    - "module.Symbol"  -> ambiguous; try module first, then attribute
    if ":" in target:
        # Explicit attribute import
        module_name, symbol_name = target.split(":", 1)
        module = importlib.import_module(module_name)
        obj = getattr(module, symbol_name)  # raises AttributeError if missing
    else:
        # First try importing the target as a module
        try:
            obj = importlib.import_module(target)
        except ModuleNotFoundError:
            # If that fails, fall back to importing it as an attribute
            if "." not in target:
                raise
            module_name, symbol_name = target.rsplit(".", 1)
            module = importlib.import_module(module_name)
            try:
                obj = getattr(module, symbol_name)
            except AttributeError as ae:
                raise ImportError(
                    f"Neither module '{target}' nor symbol '{symbol_name}' in '{module_name}'"
                ) from ae

    # 2) Determine what kind of object we imported
    if isinstance(obj, ModuleType):
        kind = "module"
        default_bind_name = obj.__name__.split(".")[-1]
    elif inspect.isclass(obj):
        kind = "class"
        default_bind_name = obj.__name__
    elif inspect.isfunction(obj) or inspect.isbuiltin(obj):
        kind = "function"
        default_bind_name = getattr(obj, "__name__", "function")
    else:
        kind = "other"
        default_bind_name = getattr(obj, "__name__", alias or "value")

    # 3) Optional type enforcement
    if require is not None and kind != require:
        raise TypeError(
            f"Imported '{target}' is of kind '{kind}', but require='{require}' was specified"
        )

    # 4) Bind the object into the target namespace
    bind_name = alias or default_bind_name
    ns[bind_name] = obj

    return obj, kind, bind_name


@dataclass(frozen=True)
class Command:
    name: str
    args: tuple
    kwargs: dict
    reply_q: Queue[Any]


class IsaacSimRuntime:
    _instance: Optional["IsaacSimRuntime"] = None

    def __new__(cls, *args: Any, **kwargs: Any) -> "IsaacSimRuntime":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(
        self, config: Optional[dict] = None, debug: bool = False, logger: Any = None
    ) -> None:
        if getattr(self, "_initialized", False):
            return
        self._initialized = True

        # State
        self.state = UNINITIALIZED

        # Debug mode
        self._debug = debug

        # Logger
        if logger is not None:
            self._logger = logger
            try:
                from rclpy.logging import LoggingSeverity

                severity = LoggingSeverity.DEBUG if self._debug else LoggingSeverity.INFO
                self._logger.set_level(severity)
            except Exception:
                pass
        else:
            self._logger = logging.getLogger("IsaacSimRuntime")
            logging.basicConfig(level=logging.DEBUG if self._debug else logging.INFO)

        # Initialize Isaac Sim
        self.initialize(config)

    def initialize(self, config=None) -> None:
        """Creates Isaac Sim instance based on the given startup config. Initializes command queue.

        Args:
            config (dict): Total config. Startup, extensions, commands.
        """
        self._logger.debug(f"{self.state}")
        assert self.state == UNINITIALIZED

        # Get configuration. Default: self-defined init.yaml
        if config is None:
            config = self._load_config()

        startup_config, extensions, stage_config = self._parse_config(config)
        self.stage_config = stage_config

        self._step_hz: float = startup_config.get("step_freq", 60.0)
        self._dt = 1.0 / self._step_hz

        self.state = INITIALIZING
        try:
            if SimulationApp is None:
                raise RuntimeError(
                    "Isaac Sim is not available (isaacsim.SimulationApp import failed)."
                )
            # Start Isaac Sim
            self.simulation_app = SimulationApp(startup_config)

            # Set up command queue
            self._cmd_q: "Queue[Command]" = Queue()

            # Isaac Sim interfaces
            self._world = None
            self._stage = None

        except Exception as e:
            self._logger.error(f"Failed to initialize IsaacSim: {e}")
            self.state = ERROR
            return

        # Load imports and extensions
        self._import_isaac_extensions(extensions)

        # Commands are Isaac-dependent; keep them importable but optional in tests.
        self._import_commands()

        self._create_world()

        self.state = STOPPED

    @staticmethod
    def _load_config() -> dict:
        try:
            config_path = get_package_share_directory("guide_core") + "/config/init.yaml"
            with open(config_path) as f:
                return yaml.safe_load(f)
        except Exception:
            return {}

    def _parse_config(self, config: dict) -> tuple[dict, dict, dict]:
        assert self.state == UNINITIALIZED

        startup_config: dict = config.get("startup", {})
        self._logger.debug(f"Startup config: {startup_config}")

        extensions_config: list = config.get("extensions", [])
        self._logger.debug(f"Extensions config: {extensions_config}")

        stage_config: dict = config.get("world", {})
        self._logger.debug(f"Stage config: {stage_config}")

        return (startup_config, extensions_config, stage_config)

    # -------------------------
    # Import functions
    # ------------------------
    def _import_isaac_extensions(self, extensions_list: list[str]) -> None:
        assert self.state in [INITIALIZING, STOPPED, READY, PAUSED]

        # Imports for further use
        predefined_imports = (
            "carb",
            "omni",
            "omni.usd",
            "pxr.Usd",
            "pxr.UsdGeom",
            "pxr.Gf",
            "isaacsim.core.api.World",
            "isaacsim.core.api.SimulationContext",
            "isaacsim.core.utils.extensions",
            "isaacsim.core.prims.XFormPrim",
            "isaacsim.core.api.robots.Robot",
            "isaacsim.storage.native.get_assets_root_path",
            {"omni.replicator.core": "rep"},
            {"omni.graph.core": "og"},
        )

        for imp in predefined_imports:
            if isinstance(imp, dict):
                for key, alias in imp.items():
                    import_and_bind(key, namespace=globals(), alias=alias)
                continue
            self._logger.debug(f"Importing {imp}...")
            import_and_bind(imp, namespace=globals())

        # User defined extensions - ROS 2 and Clash Detection are mandatory
        extensions_list.extend(
            [
                {"isaacsim.util.clash_detection": ["ClashDetector"]},
                {"isaacsim.ros2.bridge": ["scripts.og_shortcuts.og_utils.Ros2JointStatesGraph"]},
                {"isaacsim.sensors.camera": ["Camera"]},
            ]
        )
        for item in extensions_list:
            # Enabling extension module
            if isinstance(item, str):
                self._logger.debug(f"Enabling extension: {item}")
                extensions.enable_extension(item)
                continue

            # Enabling extension classes
            if isinstance(item, dict):
                for ext_id, cls_list in item.items():
                    self._logger.debug(f"Enabling extension: {ext_id}")
                    extensions.enable_extension(ext_id)
                    for cls_name in cls_list:
                        import_and_bind(f"{ext_id}.{cls_name}", namespace=globals())
                continue

            carb.log_error(f"Unsupported extension spec type: {type(item)}")

    def _import_commands(self) -> None:
        assert self.state in [INITIALIZING, STOPPED, READY, PAUSED]

        from guide_core.core._registry import attach_cmd_functions

        self._logger.debug("Importing command functions...")
        attach_cmd_functions(self, debug=self._debug)

    def _check_assets(self) -> None:
        assert self.state in [INITIALIZING, STOPPED, READY, PAUSED]

        assets_root_path = get_assets_root_path()
        if assets_root_path is None:
            carb.log_error("Could not find Isaac Sim assets folder")
            self.state = ERROR

    def _create_world(self) -> None:
        try:
            self._logger.debug("Creating World...")
            self._world = World(
                stage_units_in_meters=1.0, physics_dt=self._dt, rendering_dt=self._dt
            )

            self._pc = self._world.get_physics_context()
            self._pc.enable_gpu_dynamics(True)

            self._pc.set_physics_dt(self._dt / 2, substeps=4)
            self._world.instance().set_simulation_dt(physics_dt=self._dt / 2, rendering_dt=self._dt)

        except Exception as e:
            self._logger.error(f"Error in create_world: {e}")
            self.state = ERROR
            return

        try:
            self.update()
            self.update()

            self._stage = self._world.stage
        except Exception as e:
            self._logger.error(f"Error in updating world: {e}")
            self.state = ERROR
            return

    # -------------------------
    # Call API
    # -------------------------
    def call(self, name: str, timeout: Optional[float] = None, *args: Any, **kwargs: Any) -> Any:
        reply_q: "Queue[Any]" = Queue(maxsize=1)
        self._cmd_q.put(Command(name=name, args=args, kwargs=kwargs, reply_q=reply_q))
        try:
            result = reply_q.get(timeout=timeout)
            if isinstance(result, Exception):
                self._logger.error(result)
                return None
            return result
        except Empty as e:
            raise TimeoutError(f"Runtime call timed out: {name}") from e

    # -------------------------
    # Stepping interface
    # -------------------------
    def step(self, n: int = 1) -> None:
        """Steps simulation by the given number of steps. If internal state is RUNNING, it also renders.

        Args:
            n (int, optional): Number of steps. Defaults to 1.
        """
        assert self.state == RUNNING

        for _ in range(n):
            self._world.step(render=True)

    def update(self, n: int = 1) -> None:
        """Updates the application by the given number of steps. If internal state is RUNNING, it also renders.

        Args:
            n (int, optional): Number of steps. Defaults to 1.
        """
        assert self.state not in [UNINITIALIZED]

        for _ in range(n):
            self.simulation_app.update()

    # -------------------------
    # Runtime loop
    # -------------------------
    def run_loop(self) -> None:
        """Runs the runtime loop. \\
        This is a blocking method, but needs to be run in the main thread. \\
        Ends when objects internal state is SHUTTING_DOWN.
        """

        while True:
            print(f"[DEBUG_FREEZE] run_loop: outer loop iteration, state={self.state}")
            while self.state not in [SHUTTING_DOWN, UNINITIALIZED]:
                start = time.time()

                self._process_commands(max_per_cycle=50)

                if self.state == RUNNING:
                    try:
                        self._world.step()
                    except BaseException as e:
                        print(f"[FATAL ERROR IN STEP] {type(e).__name__}: {e}")
                        import traceback

                        traceback.print_exc()

                sleep_s = max(0.0, start + self._dt - time.time())
                time.sleep(sleep_s)

            print(f"[DEBUG_FREEZE] run_loop: inner loop EXITED, state={self.state}")

    def _process_commands(self, max_per_cycle: int) -> None:
        for _ in range(max_per_cycle):
            try:
                cmd: Command = self._cmd_q.get_nowait()
                self._logger.debug(f"Got command: {cmd.name}")
            except Empty:
                return

            try:
                handler = getattr(self, f"_cmd_{cmd.name}")
            except AttributeError:
                cmd.reply_q.put(RuntimeError(f"Unknown command: {cmd.name}"))
                continue

            try:
                result = handler(*cmd.args, **cmd.kwargs)
                if result is None:
                    result = True
                cmd.reply_q.put(result)
                self._logger.debug("Command processed successfully")
                # [DEBUG_FREEZE] Check if state changed during command execution
                if self.state in [SHUTTING_DOWN, UNINITIALIZED]:
                    print(
                        f"[DEBUG_FREEZE] _process_commands: state changed to {self.state} AFTER executing '{cmd.name}'"
                    )
            except BaseException as e:
                cmd.reply_q.put(e)
                self._logger.error(
                    f"Error processing command '{cmd.name}': {type(e).__name__} - {e}",
                    exc_info=True,
                )
                # [DEBUG_FREEZE] Check if state changed during exception
                if self.state in [SHUTTING_DOWN, UNINITIALIZED]:
                    print(
                        f"[DEBUG_FREEZE] _process_commands: state changed to {self.state} AFTER exception in '{cmd.name}': {e}"
                    )

    def is_running(self) -> bool:
        return self.state == RUNNING
