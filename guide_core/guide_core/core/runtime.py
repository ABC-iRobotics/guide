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
            config (dict): Total config: ``startup``, ``extensions`` and ``world`` sections.
                Defaults to guide_core's ``config/init.yaml``.
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

        # PhysX ticks faster than the renderer; Isaac derives substeps from the ratio
        # (isaacsim.core.api SimulationContext.set_simulation_dt). This is the clock
        # `current_time_step_index` counts and SceneManager measures its record
        # interval against -- NOT step_freq. Keep them in one place.
        self._physics_hz: float = startup_config.get("physics_freq", 2 * self._step_hz)

        # Pace the loop to sim time. False lets a scene that renders faster than its
        # frame budget run ahead of real time, which is what batch demonstration
        # generation wants.
        self._realtime: bool = startup_config.get("realtime", True)

        # PhysX on the GPU. NVIDIA documents GPU dynamics as a win at scale -- many
        # bodies, many contacts -- and this scene is one arm, two bins and four
        # blocks. It also puts PhysX on the same card the renderer is saturating.
        # Where PhysX runs: "cpu", or "cuda:N" as CUDA numbers devices -- which is NOT
        # how nvidia-smi or render_device number them. CUDA defaults to FASTEST_FIRST,
        # nvidia-smi and Kit go by PCI bus id, so the same N can name different cards.
        # Pinning CUDA_DEVICE_ORDER would unify them but is a process-global change
        # that also relocates every component defaulting to device 0 -- measured at
        # ~7 ms a frame here, and unavailable on a host where the environment is not
        # ours to set. So both keys log the card they resolved to instead; check the
        # startup lines against nvidia-smi rather than trusting either convention.
        physics_device = str(startup_config.get("physics_device", "cpu")).strip().lower()
        self._gpu_dynamics: bool = physics_device != "cpu"
        self._physics_gpu: int = int(physics_device.split(":")[1]) if self._gpu_dynamics else -1

        # PhysX worker threads (/persistent/physics/numThreads). 0 runs the solver
        # synchronously on the calling thread; -1 leaves Isaac's default of 8.
        self._physics_threads: int = int(startup_config.get("physics_threads", -1))

        self.state = INITIALIZING
        try:
            if SimulationApp is None:
                raise RuntimeError(
                    "Isaac Sim is not available (isaacsim.SimulationApp import failed)."
                )
            # "cuda:N" -> the active_gpu index SimulationApp wants. Same N nvidia-smi
            # shows, because Kit numbers cards by PCI bus id too (see above; nothing
            # pins CUDA_DEVICE_ORDER). Kept single-GPU: these two cards have no peer
            # access, and Isaac's multi-GPU renderer deadlocks on them ("Failed to
            # begin render graph ... semaphore timed out").
            render_device = str(startup_config.get("render_device", "")).strip().lower()
            if render_device.startswith("cuda:"):
                startup_config["active_gpu"] = int(render_device.split(":")[1])
                startup_config["multi_gpu"] = False
                self._logger.info(f"Renderer on {render_device}.")

            if self._physics_gpu >= 0:
                # Has to go in before the app starts: SimulationApp turns physics_gpu
                # into --/physics/cudaDevice=N on the Kit command line, and PhysX builds
                # its CUDA context during startup. Setting the carb value afterwards
                # logs a pin that never happens -- measured: the setting read cuda:1
                # while nvidia-smi showed the A2000 idle at 1% throughout.
                startup_config["physics_gpu"] = self._physics_gpu
                self._logger.info(f"PhysX on cuda:{self._physics_gpu}.")

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

        # User defined extensions.
        # NOTE: the OmniGraph ROS 2 shortcut classes live in isaacsim.ros2.ui in
        # Isaac Sim 6.0 (they were isaacsim.ros2.bridge.impl.og_shortcuts in 5.x).
        # isaacsim.util.clash_detection was removed in 5.x but is back in 6.0, so
        # _cmd_clash.py's optional ClashDetector import works again (full mesh clash).
        extensions_list.extend(
            [
                {"isaacsim.ros2.ui": ["og_utils.Ros2JointStatesGraph"]},
                {"isaacsim.sensors.camera": ["Camera"]},
                {"isaacsim.util.clash_detection": ["ClashDetector"]},
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

    def _create_world(self) -> None:
        try:
            self._logger.debug("Creating World...")

            if self._physics_threads >= 0:
                import carb.settings

                carb.settings.get_settings().set_int(
                    "/persistent/physics/numThreads", self._physics_threads
                )
                self._logger.info(
                    f"PhysX worker threads: {self._physics_threads}"
                    f"{' (synchronous)' if self._physics_threads == 0 else ''}."
                )

            self._world = World(
                stage_units_in_meters=1.0,
                physics_dt=1.0 / self._physics_hz,
                rendering_dt=self._dt,
            )

            self._pc = self._world.get_physics_context()
            self._pc.enable_gpu_dynamics(self._gpu_dynamics)

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
    def update(self, n: int = 1) -> None:
        """Ticks the Kit app ``n`` times via ``SimulationApp.update()``, in any state but
        UNINITIALIZED.

        Args:
            n (int, optional): Number of steps. Defaults to 1.
        """
        assert self.state not in [UNINITIALIZED]

        for _ in range(n):
            self.simulation_app.update()

    # -------------------------
    # Runtime loop
    # -------------------------
    # Frames between frame-budget reports: ~5 s of sim time at 60 Hz.
    FRAME_LOG_EVERY = 300

    def _gate_render(self, frame_index: int = 0, enabled: bool | None = None) -> None:
        """Hand the render-product gate to the scene manager, if there is one yet.

        ``enabled=True`` overrides the schedule and holds the products open, which is
        what the loop does whenever it is not RUNNING.
        """
        scene_manager = getattr(getattr(self, "_simulator", None), "_scene_manager", None)
        if scene_manager is None:
            return
        try:
            scene_manager.gate_render(frame_index, self._step_hz, enabled=enabled)
        except Exception as e:
            self._logger.debug(f"Could not gate render products: {e}")

    def run_loop(self) -> None:
        """Runs the runtime loop. \\
        This is a blocking method, but needs to be run in the main thread. \\
        Ends when objects internal state is SHUTTING_DOWN.
        """
        frames = 0
        step_s = 0.0
        loop_s = 0.0
        window_start = time.perf_counter()
        # Monotonic count of RENDERED frames, which is the clock the camera render
        # products have to be gated on -- see SceneManager.gate_render.
        frame_index = 0

        while self.state not in [SHUTTING_DOWN, UNINITIALIZED]:
            start = time.perf_counter()

            self._process_commands(max_per_cycle=50)
            if self.state in (SHUTTING_DOWN, UNINITIALIZED):
                break  # a shutdown command closed Isaac: touch nothing more

            if self.state != RUNNING:
                # Nothing renders, so don't make commands wait a frame period for the
                # next pass: every call() in the between-episode reset/randomize/home
                # chain queues up here.
                #
                # Leave the render products ON while stopped. The gate below only runs
                # while RUNNING, so whatever it last wrote would otherwise stick for
                # the whole reset/randomize chain -- five frames in six that is "off",
                # and a camera topic that goes quiet across a scene change looks to a
                # policy evaluation exactly like a simulator that died.
                self._gate_render(enabled=True)
                time.sleep(0.001)
                continue

            # Open the render window for exactly this frame, before the step that
            # renders it. Doing it here rather than in the physics callback is the
            # whole point: see SceneManager.gate_render.
            self._gate_render(frame_index=frame_index)

            step_start = time.perf_counter()
            try:
                self._world.step()
            except BaseException as e:
                self._logger.error(f"Error in simulation step: {e}", exc_info=True)
            frame_index += 1

            now = time.perf_counter()
            step_s += now - step_start
            loop_s += now - start
            frames += 1
            if frames >= self.FRAME_LOG_EVERY:
                # RTF < 1 means the frame costs more than its budget -- the sleep below
                # is already 0 and the sim is falling behind. Split out so a slow
                # command handler is not mistaken for a slow renderer.
                wall = now - window_start
                self._logger.info(
                    f"[runtime] {1e3 * step_s / frames:.1f} ms/step + "
                    f"{1e3 * (loop_s - step_s) / frames:.1f} ms/cmds "
                    f"(budget {1e3 * self._dt:.1f} ms), RTF {frames * self._dt / wall:.2f}"
                )
                frames, step_s, loop_s = 0, 0.0, 0.0
                window_start = time.perf_counter()

            if self._realtime:
                time.sleep(max(0.0, start + self._dt - time.perf_counter()))

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
            except BaseException as e:
                cmd.reply_q.put(e)
                self._logger.error(
                    f"Error processing command '{cmd.name}': {type(e).__name__} - {e}",
                    exc_info=True,
                )

    def is_running(self) -> bool:
        return self.state == RUNNING
