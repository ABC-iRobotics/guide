from __future__ import annotations

import logging
import os
from abc import ABC, abstractmethod
from importlib import resources
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import yaml
from scipy.spatial.transform import Rotation as R

from guide_core.scene.scene_recorder import DEFAULT_FPS, SceneRecorder
from guide_core.types.geometry import Point, Pose, Rotation
from guide_core.types.randomization import replicator_guide
from guide_core.types.randomization import (
    RandomizationRecord,
    Randomizer,
    SeedTree,
    draw_instructions,
    grid_from_yaml,
    pose_from_yaml,
    single_grid,
)
from guide_core.types.randomization._quat import as_range
from guide_core.types.scene_context import SceneContext
from guide_core.types.scene_state import SceneState

logger = logging.getLogger("SceneOrchestrator")


def depth_to_uint16_mm(depth: np.ndarray) -> np.ndarray:
    """Isaac's float32 metres -> the (H, W, 1) uint16 millimetres lerobot records.

    Not our convention -- lerobot's. ``hw_to_dataset_features`` flags a 1-channel
    camera as ``is_depth_map``, which routes the stream through ``DepthEncoderConfig``
    (HEVC Main 12, lossless), and ``quantize_depth`` reads a non-floating dtype as
    millimetres (``lerobot/datasets/depth_utils.py``). Handing it uint16 mm means no
    conversion happens anywhere between here and the encoder.

    ``distance_to_image_plane`` returns +inf where the ray hit nothing. Those become 0
    -- the value the quantizer treats as "no reading" -- rather than saturating to
    65.5 m and dragging the depth range with them.
    """
    metres = np.nan_to_num(np.asarray(depth, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    millimetres = np.clip(metres * 1000.0, 0.0, 65535.0).astype(np.uint16)
    return millimetres.reshape(millimetres.shape[0], millimetres.shape[1], 1)


class SceneOrchestrator(ABC):

    scene_id: int

    _offset: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    _pkg_name: str

    _config: dict
    _usd_path: str
    bounding_box: dict
    origin: list

    reset_instructions: list[dict] = []
    randomize_instructions: list[dict] = []

    state: SceneState
    recorder: SceneRecorder

    def __init__(
        self,
        scene_id: int,
        sim_id: int = 0,
        path: Optional[str] = None,
        config_path: str = "/config/init.yaml",
        reset_path: str = "/config/reset.yaml",
        randomize_path: str = "/config/randomize.yaml",
        success_path: str = "/config/success.yaml",
        logger: Any = None,
        master_seed: Optional[int] = None,
    ):
        self._scene_id = scene_id
        self._sim_id = sim_id
        if logger is None:
            import logging

            self._logger = logging.getLogger("SceneOrchestrator")
        else:
            self._logger = logger

        # The procedure-level prompt, drawn at randomization.
        self.task = ""
        # The GUIDE-EX task and subtask being worked on, set through
        # SceneManager.set_prompt; "" = none. The task is each frame's LeRobot task.
        self.prompts = {"task": "", "subtask": ""}

        # Getting init.yaml
        package_name = self.__class__.__module__.split(".")[0]

        if path is None:
            self._path = resources.files(package_name)
            config_file = resources.files(package_name).joinpath(config_path)
        else:
            self._path = Path(path)
            config_file = Path(f"{path}/{config_path}")

        # with resources.files(package_name).joinpath(config_path).open('r') as f:
        with config_file.open("r") as f:
            self._config = yaml.safe_load(f)

        self._get_usd_params(package_name)

        self._get_limits()
        self._get_origin()

        # Each instruction file is either the instruction list or the Replicator dialect
        # (spike: see types/randomization/replicator_guide.py). Replicator files parse into a
        # graph once the scene's USD is on the stage (build_replicator); their instruction
        # lists stay empty so the hooks and the executor see nothing to run.
        self.replicator_files: dict[str, Path] = {}
        self.replicator: dict[str, dict] = {}

        def instructions(phase: str, rel: str) -> list:
            file = Path(f"{path}/{rel}")
            if replicator_guide.is_replicator_yaml(file):
                self.replicator_files[phase] = file
                return []
            return self.parse_instruction(file)

        # Getting reset.yaml
        self.reset_instructions = instructions("reset", reset_path)

        # Getting randomize.yaml
        self.randomize_instructions = instructions("randomize", randomize_path)

        # At most one grid-enabled instruction per scene (raises on a second).
        self._grid = single_grid(self.randomize_instructions)

        # Getting success.yaml (a query; stays on the executor -- see docs/design)
        self.success_instructions = self.parse_instruction(Path(f"{path}/{success_path}"))

        # Single RNG authority for this scene (master seed injected at
        # registration, else auto from system entropy -- captured + logged).
        self._seed_tree = SeedTree.create(master_seed)
        self._episode_index = 0
        self._last_context: Optional[SceneContext] = None
        self._logger.info(
            f"[SceneOrchestrator] scene {self._scene_id} master seed = {self._seed_tree.master}"
        )

        self.state = SceneState.IDLE

        # Start separate recorder process
        dataset_name = f"dataset_{self._sim_id}_{self._scene_id}"
        from guide_core.core.recorder_manager import RecorderServer

        # The scene module loads via spec_from_file_location as "scene", so
        # self.__class__.__module__ ("scene") is not the real package. The scene
        # directory name is the actual package (e.g. "block_bin").
        self._package_name = Path(str(self._path)).name or package_name

        try:
            client = RecorderServer.get_client()
            self.recorder = client.get_recorder(self._package_name, dataset_name, self._config)
            # Seed the metadata sidecar with run-level constants (master seed, ids,
            # and the zone-grid layout so a dataset is self-describing).
            try:
                run_meta = {
                    "master_seed": self._seed_tree.master,
                    "scene_id": self._scene_id,
                    "sim_id": self._sim_id,
                }
                if self._grid is not None:
                    g = self._grid
                    run_meta["grid"] = {
                        "region_low": [float(x) for x in g.low],
                        "region_high": [float(x) for x in g.high],
                        "resolution": g.resolution,
                        "ncols": g.ncols,
                        "nrows": g.nrows,
                        "num_zones": g.num_zones,
                    }
                run_meta["instance_colors"] = {k: list(v) for k, v in self.instance_colors.items()}
                self.recorder.set_run_meta(run_meta)
            except Exception as e:
                self._logger.debug(f"Could not set run meta on recorder: {e}")
        except ConnectionRefusedError:
            self._logger.error(
                "RecorderServer not running! Ensure it was started in the Main Process."
            )
            self.recorder = None

    def _get_usd_params(self, package_name):
        usd_path = self._config["usd_path"]
        # 'package://<pkg>/<path>' loads another package's installed asset (the ROS
        # convention), so a task can reuse a scene without copying it.
        if usd_path.startswith("package://"):
            from ament_index_python.packages import get_package_share_directory

            pkg, _, rel = usd_path.removeprefix("package://").partition("/")
            self._usd_path = Path(get_package_share_directory(pkg)) / rel
        else:
            self._usd_path = self._path.joinpath(usd_path.lstrip("/"))

    def _get_limits(self):
        # limits: [[x_min, x_max], [y_min, y_max], [z_min, z_max]] around the origin
        limits = self._config.get("limits", None)

        assert limits is not None

        low, high = as_range(limits)
        # Creating bounding box; *n entries are distances on the negative side
        self.bounding_box = {
            "xp": float(high[0]),
            "xn": float(-low[0]),
            "yp": float(high[1]),
            "yn": float(-low[1]),
            "zp": float(high[2]),
            "zn": float(-low[2]),
        }

    def _get_origin(self):
        self.origin = self._config.get("origin", [0.0, 0.0, 0.0])

        assert self.origin is not None

    def set_offset(self, offset: Tuple[float, float, float]):
        self._offset = offset

    def create_robot_graphs(self):
        robot_list: List[Dict] = []
        for name, data in self._config.get("robots", {}).items():
            path = data["path"]
            assert path is not None
            default_joint_states = data["default_joint_states"]
            assert default_joint_states is not None
            robot_list.append(
                {
                    "namespace": f"/{name}",
                    "articulation_root": f"{path}",
                    "path": f"/{name}",
                    "default_joint_states": default_joint_states,
                }
            )
        return robot_list

    def create_camera_graphs(self):
        # ROS 2 camera publisher graphs are optional and OFF by default. Each one is a
        # second render product for a camera the recorder already renders, and it
        # publishes raw images every rendered frame: three 640x480 rgb8 streams at 60 Hz
        # is ~166 MB/s of reliable traffic over the localhost DDS transport, which both
        # doubles the renderer's per-frame work and starves small service replies
        # (e.g. PoseRequest). Dataset generation never reads them -- the recorder
        # captures GUIDE-side via render-product annotators.
        #
        # Turn them on only for inference, where a deployed policy does need live images
        # over ROS. Either switch does it, so an eval run does not have to edit the
        # task's YAML and a task that always needs them does not have to remember a
        # launch argument:
        #   ros2 launch guide_core bringup.launch.py camera_topics:=true
        #   publish_camera_topics: true   (task config/init.yaml)
        env_flag = os.environ.get("GUIDE_CAMERA_TOPICS", "")
        publish = bool(self._config.get("publish_camera_topics", False)) or (
            env_flag.strip().lower() in ("1", "true", "yes", "on")
        )

        if not publish:
            self._logger.info(
                "[SceneOrchestrator] publish_camera_topics=false: skipping ROS 2 camera "
                "publisher graphs (recorder still captures images via annotators)."
            )
            return []

        return self.resolve_cameras()

    #: Appended to a dataset feature key to keep a camera's depth stream apart from its
    #: own rgb stream. Only used when the camera publishes both -- see resolve_cameras.
    DEPTH_SUFFIX = "_depth"
    #: Same, for a camera's instance-segmentation stream.
    INSTANCE_SUFFIX = "_instance"
    #: Colour of each tracked object in the instance stream, in ``dataset.tracked_objects``
    #: order (tab10); everything untracked is black. The legend goes into guide_info.json.
    INSTANCE_PALETTE = [
        (31, 119, 180), (255, 127, 14), (44, 160, 44), (214, 39, 40), (148, 103, 189),
        (140, 86, 75), (227, 119, 194), (127, 127, 127), (188, 189, 34), (23, 190, 207),
    ]

    def resolve_cameras(self) -> List[Dict]:
        """The camera plan: one entry per configured camera, derived once from the config.

        Everything downstream reads this instead of walking ``config/init.yaml`` again --
        the recorder's render products and annotators, the dataset feature names, and the
        ROS 2 publisher graphs. Two of those three used to derive their own answer
        separately, which is how a camera could be recorded at one resolution and
        published at another.

        A camera carries up to two streams, ``rgb`` (default on) and ``depth`` (default
        off). Only ``rgb`` reaches ROS 2; depth is captured in-process by the render
        product's annotator and only ever lands in the dataset, like the semantic
        labels. The dataset feature names follow from the pair:

        ======  ======  ==============  ======================
        rgb     depth   rgb feature     depth feature
        ======  ======  ==============  ======================
        true    false   ``<key>``       --
        true    true    ``<key>``       ``<key>_depth``
        false   true    --              ``<key>``
        ======  ======  ==============  ======================

        A camera whose only stream is depth keeps the plain key: the suffix is there to
        keep one camera's two streams apart, not to label the modality. ``<key>`` is the
        ``dataset.images`` key rather than the camera name -- the two are free to differ,
        and a camera absent from ``dataset.images`` gets no features at all, because it
        is published for a live policy but never recorded.
        """
        # dataset.images is a list of single-entry dicts: {feature key: camera name}.
        feature_key: Dict[str, str] = {}
        for img_item in self._config.get("dataset", {}).get("images", []):
            for ds_key, cam_name in img_item.items():
                feature_key[cam_name] = ds_key

        encoding = self._config.get("camera_encoding", "rgb")

        cameras: List[Dict] = []
        for name, data in self._config.get("cameras", {}).items():
            path = data["path"]
            assert path is not None
            width = data["width"]
            assert width is not None
            height = data["height"]
            assert height is not None
            topic = data["topic"]
            assert topic is not None

            rgb = bool(data.get("rgb", True))
            depth = bool(data.get("depth", False))
            instance = bool(data.get("instance", False))
            if not (rgb or depth or instance):
                self._logger.warning(
                    f"[SceneOrchestrator] camera '{name}' has every stream off; "
                    f"skipping it entirely."
                )
                continue

            ds_key = feature_key.get(name)
            cameras.append(
                {
                    "name": f"{name}",
                    "camera_path": f"{path}",
                    "path": f"/{name}",
                    "width": width,
                    "height": height,
                    "frame": f"{name}",
                    "topic": f"{topic}",
                    "rgb": rgb,
                    "depth": depth,
                    "encoding": encoding,
                    # The topics publish at the rate the dataset was recorded at; the
                    # runtime turns it into a frame-skip count, since only it knows the
                    # render rate. See _set_publish_rate.
                    "fps": self.record_frequency,
                    "rgb_feature": ds_key if (rgb and ds_key) else None,
                    "depth_feature": (
                        (f"{ds_key}{self.DEPTH_SUFFIX}" if rgb else ds_key)
                        if (depth and ds_key)
                        else None
                    ),
                    "instance_feature": (
                        (f"{ds_key}{self.INSTANCE_SUFFIX}" if (rgb or depth) else ds_key)
                        if (instance and ds_key)
                        else None
                    ),
                }
            )
        return cameras

    def parse_instruction(self, path: Path):

        relative_path_keywords = ["prim_path", "articulation_root", "scope"]

        with path.open("r") as f:
            file = yaml.safe_load(f) or {}
        instructions = file.get("instructions") or []
        instruction_list = []
        for instruction in instructions:
            for key, value in instruction.get("kwargs", {}).items():
                if key in relative_path_keywords:
                    if isinstance(value, list):
                        instruction["kwargs"][key] = [f"/Scene_{self._scene_id}{v}" for v in value]
                    else:
                        instruction["kwargs"][key] = f"/Scene_{self._scene_id}{value}"
                if key == "pose":
                    # The randomization spec (PoseDist) is owned by the
                    # randomization package; the Randomizer draws it in
                    # randomize(). We attach the spec to the instruction and put
                    # a concrete *base* Pose in kwargs for immediate execution.
                    instruction["pose_dist"] = pose_from_yaml(instruction["kwargs"]["pose"])
                    # Optional zone grid over the position range (grid_from_yaml
                    # returns None unless the position carries grid.enabled).
                    instruction["grid"] = grid_from_yaml(
                        (instruction["kwargs"]["pose"] or {}).get("position")
                    )

                    pose_spec = instruction["kwargs"]["pose"]
                    position_base = np.array(
                        (pose_spec.get("position") or {}).get("value", [0.0, 0.0, 0.0])
                    )
                    orientation_base = np.array(
                        (pose_spec.get("orientation") or {}).get("value", [0.0, 0.0, 0.0])
                    )
                    instruction["kwargs"][key] = Pose(
                        Point(position_base),
                        Rotation(R.from_euler("xyz", orientation_base, degrees=True)),
                    )
            instruction_list.append(instruction)

        assert instruction_list is not None

        return instruction_list

    @abstractmethod
    def reset_preprocess(self, instructions):
        raise NotImplementedError("Reset preprocess function is not implemented for this scene.")

    @abstractmethod
    def reset_postprocess(self, result):
        raise NotImplementedError("Reset postprocess function is not implemented for this scene.")

    @abstractmethod
    def randomize_preprocess(self, randomizer):
        raise NotImplementedError(
            "Randomize preprocess function is not implemented for this scene."
        )

    @abstractmethod
    def randomize_postprocess(self, result):
        raise NotImplementedError(
            "Randomize postprocess function is not implemented for this scene."
        )

    @abstractmethod
    def is_success_preprocess(self, instructions):
        raise NotImplementedError("Success preprocess function is not implemented for this scene.")

    @abstractmethod
    def is_success_postprocess(self, result):
        raise NotImplementedError("Success postprocess function is not implemented for this scene.")

    @abstractmethod
    def check_warmup(self):
        raise NotImplementedError("Check warmup function is not implemented for this scene.")

    def _recorded_annotators(self) -> Dict[str, Any]:
        return {
            **getattr(self, "rgb_annotators", {}),
            **getattr(self, "depth_annotators", {}),
            **getattr(self, "instance_annotators", {}),
        }


    def zone_target(self) -> Optional[str]:
        """Prim path to place in the requested zone (e.g. the selected target).

        Scenes with a zoned target (see docs/design/zoned-randomization.md) override
        this to return the scene-prefixed prim path chosen this episode (typically
        after ``randomize_preprocess`` picks it). Default: no zoned target.
        """
        return None

    def randomize(self, *, seed=None, inject=None, zone=None) -> SceneContext:
        """Draw this episode's randomization deterministically and capture it.

        Order: the scene's discrete/selection draws (``randomize_preprocess``) run
        FIRST so the scene can declare the zone target (e.g. the color-picked block)
        before the pose draws that place it; then every randomize instruction's
        PoseDist is drawn into a concrete Pose (kwargs['pose']). ``zone`` (>=0)
        restricts the ``zone_target`` prim to that grid cell; everything else is
        free. If ``inject`` is given, drawn values come from it instead of the RNG.
        """
        if seed is not None:
            rng, used = self._seed_tree.generator(int(seed))
        else:
            rng, used = self._seed_tree.generator(self._scene_id, self._episode_index)

        record = RandomizationRecord(seed=used)
        randomizer = Randomizer(rng, record, inject=inject)

        # Discrete/selection draws first -> the scene can declare its zone target.
        try:
            self.randomize_preprocess(randomizer)
        except NotImplementedError:
            pass

        draw_instructions(
            self.randomize_instructions,
            randomizer,
            self._pose_from_vec7,
            self._resolve_prims,
            zone=zone,
            zone_target=self.zone_target(),
        )

        if "randomize" in self.replicator:
            samples = replicator_guide.fire(self.replicator["randomize"], used, zone, self.zone_target())
            record.values.update({f"replicator/{k}": v for k, v in samples.items()})

        self._last_context = SceneContext(
            scene_id=self._scene_id,
            episode_index=self._episode_index,
            record=record,
            zone=zone,
        )
        self._episode_index += 1
        return self._last_context

    def build_replicator(self) -> None:
        """Parse every Replicator-dialect file once the scene's USD is on the stage."""
        for phase, file in self.replicator_files.items():
            self.replicator[phase] = replicator_guide.build(file, f"/Scene_{self._scene_id}")
        if "randomize" in self.replicator:
            self._grid = self.replicator["randomize"].get("grid")

    def reset_fire(self) -> None:
        """Run the Replicator reset file, if the task ships one."""
        if "reset" in self.replicator:
            replicator_guide.fire(self.replicator["reset"])

    @staticmethod
    def _pose_from_vec7(vec) -> Pose:
        v = np.asarray(vec, dtype=float).reshape(7)
        # vec is [x, y, z, w, x, y, z]; SciPy wants quat [x, y, z, w]
        return Pose(Point(v[:3]), Rotation(R.from_quat([v[4], v[5], v[6], v[3]])))

    def _resolve_prims(self, expr: str) -> list:
        """Concrete prim paths matching an Isaac prim-path pattern.

        Resolves with the SAME mechanism ``_cmd_set_local_poses`` uses —
        ``XFormPrim`` — so the set matches exactly what the command targets:
        only *xformable* prims (the object Xforms), NOT their meshes/materials.
        (``find_matching_prim_paths`` regex-matches every prim under the pattern,
        which both over-selects non-xformable prims and diverges from the command.)
        Sorted for a deterministic, reproducible draw order.
        """
        try:
            from isaacsim.core.prims import XFormPrim

            view = XFormPrim(prim_paths_expr=expr, reset_xform_properties=False)
            return sorted(str(p) for p in view.prim_paths)
        except Exception as e:
            self._logger.warning(f"Could not resolve prims for pattern '{expr}': {e}")
            return []

    def get_start_state(self) -> dict:
        """Per-robot starting joint configuration: ``{robot_name: {dof_name: value}}``.

        Read live from each robot's articulation view. The robot is not reset to a
        fixed home between episodes, so this captures the configuration the episode
        actually starts from.
        """
        state: dict = {}
        views = getattr(self, "robots_views", None) or {}
        for r_name, view in views.items():
            try:
                names = list(view.dof_names)
                pos = view.get_joint_positions()
                if hasattr(pos, "ndim") and pos.ndim > 1:
                    pos = pos[0]
                state[r_name] = {n: float(v) for n, v in zip(names, pos)}
            except Exception as e:
                self._logger.debug(f"Could not read start state for robot '{r_name}': {e}")
        return state

    def create_render_products(self, rep):
        self.rgb_annotators = {}
        self.depth_annotators = {}
        self.instance_annotators = {}
        self.render_products = []
        # Label the tracked prims before any annotator renders.
        self.apply_semantics()

        for camera in self.resolve_cameras():
            if not (camera["rgb_feature"] or camera["depth_feature"] or camera["instance_feature"]):
                continue  # published for a live policy, but not part of the dataset

            # One render product per camera, shared by both annotators: rgb and depth
            # come off the same RTX pass, so a depth camera costs no extra render.
            res = (camera["width"], camera["height"])
            rp = rep.create.render_product(f"/Scene_{self._scene_id}{camera['camera_path']}", res)
            self.render_products.append(rp)

            if camera["rgb_feature"]:
                annotator = rep.AnnotatorRegistry.get_annotator("rgb")
                annotator.attach([rp])
                self.rgb_annotators[camera["rgb_feature"]] = annotator

            if camera["depth_feature"]:
                # distance_to_image_plane, not distance_to_camera: the plane distance is
                # what a depth sensor reports and what a pinhole unprojection expects.
                annotator = rep.AnnotatorRegistry.get_annotator("distance_to_image_plane")
                annotator.attach([rp])
                self.depth_annotators[camera["depth_feature"]] = annotator

            if camera["instance_feature"]:
                # Per-object masks of the tracked objects. Every tracked object carries
                # its own label (apply_semantics), so the label segmentation separates
                # exactly the tracked instances; record_step paints each label in a fixed
                # colour across frames, episodes and cameras. Not the instance AOVs: on
                # this scene (Isaac 6.0.1) both instance_segmentation_fast and
                # instance_id_segmentation_fast segfault in rtx.syntheticdata
                # (Sdf_PathNode::GetPathToken) as soon as a frame is rendered, gated or
                # not. This one works -- provided the render product is never gated
                # (SceneManager.gate_render keeps it on).
                # device="cuda": the ids stay a warp array on the render GPU and
                # colorize_instances copies them with that array's own device. The
                # default host copy goes through Warp's device numbering, which on a
                # two-GPU host is not Kit's, and fails (wp_memcpy_d2h invalid argument).
                annotator = rep.AnnotatorRegistry.get_annotator(
                    "semantic_segmentation", init_params={"colorize": False}, device="cuda"
                )
                annotator.attach([rp])
                self.instance_annotators[camera["instance_feature"]] = annotator

        self.setup_dataset()

    def set_render_products_enabled(self, enabled: bool) -> None:
        """Switch the recorder's camera render products on or off.

        Once created, a render product renders its camera every rendered frame for the
        life of the stage, whether or not anything reads it. Three 640x480 RTX passes
        per frame is most of this scene's frame budget, and the recorder consumes them
        ten times a second -- while idle, between episodes, and after a run has
        finished it consumes them not at all. So they are off unless a capture is
        actually about to read them (SceneManager.step) or the scene is warming up.

        Isaac's own code drives render products this way; see
        ``omni/replicator/core/scripts/annotators.py`` and
        ``isaacsim/replicator/nurec_utils/render.py``.
        """
        # Idempotent: gate_render calls this once per rendered frame and most frames
        # change nothing.
        if getattr(self, "_render_products_enabled", None) == enabled:
            return
        self._render_products_enabled = enabled

        for rp in getattr(self, "render_products", []):
            try:
                rp.hydra_texture.set_updates_enabled(enabled)
            except Exception as e:  # a render product destroyed with its stage
                self._logger.debug(f"Could not toggle render product updates: {e}")

    @property
    def record_frequency(self) -> float:
        """Hz at which the recorder captures and the ROS 2 camera topics publish.

        One key, ``dataset.fps``, drives all three of the rates that describe a
        dataset: the interval SceneManager samples on, the rate the camera topics are
        gated to, and the ``fps`` stamped into the dataset metadata. They used to be
        independent -- SceneManager read a ``getattr(scene, "record_frequency", 10)``
        that nothing in the repo ever assigned, the metadata said 30, and the topics
        published every rendered frame -- so the number a dataset reported was not the
        number it was recorded at. See ``scene_recorder.DEFAULT_FPS``.
        """
        return float(self._config.get("dataset", {}).get("fps", DEFAULT_FPS))

    def resolve_semantics(self) -> Dict[str, str]:
        """Identifier -> scene-scoped prim-path pattern, from ``dataset.tracked_objects``.

        These are the labels an instance-segmentation mask is keyed by. Each value is an
        Isaac prim-path pattern in the same syntax ``reset.yaml`` and ``randomize.yaml``
        use for ``prim_path`` -- written relative to the scene root, so one task config
        works whichever ``Scene_N`` it is instantiated into.

        A pattern may match several prims (``/blocks/*``), in which case they all carry
        the same identifier: the segmentation annotator still separates them as
        instances, it just reports them under one name. Give a prim its own key when it
        needs its own name.
        """
        tracked = self._config.get("dataset", {}).get("tracked_objects", {}) or {}
        return {str(label): f"/Scene_{self._scene_id}{expr}" for label, expr in tracked.items()}

    @property
    def instance_colors(self) -> Dict[str, tuple]:
        """Tracked-object label -> RGB in the instance stream, in config order."""
        labels = list(self.resolve_semantics())
        palette = self.INSTANCE_PALETTE
        return {label: palette[i % len(palette)] for i, label in enumerate(labels)}

    def colorize_instances(self, data: dict) -> np.ndarray:
        """Paint a semantic_segmentation frame: each tracked object in its colour,
        everything else black.

        ``idToLabels`` maps each id to its labels, e.g. {"class": "red_block"}; labels a
        prim inherits arrive comma-joined ("robot, camera"), so any part naming a
        tracked object claims the id. One lookup-table pass per frame.
        """
        ids = data["data"]
        ids = ids.numpy() if hasattr(ids, "numpy") else np.asarray(ids)  # warp array on GPU
        ids = ids.reshape(ids.shape[:2])
        colors = self.instance_colors
        lut = np.zeros((int(ids.max()) + 1 if ids.size else 1, 3), np.uint8)
        for label_id, labels in data.get("info", {}).get("idToLabels", {}).items():
            names = labels.get("class", "") if isinstance(labels, dict) else labels
            parts = [p.strip() for p in str(names).split(",")]
            label = next((p for p in parts if p in colors), None)
            if label is not None and int(label_id) < len(lut):
                lut[int(label_id)] = colors[label]
        return lut[ids]

    def apply_semantics(self) -> Dict[str, list]:
        """Stamp the configured identifiers onto the stage. Returns {label: [prim paths]}.

        A prim carrying no label is invisible to the instance-segmentation annotator, so
        this is what makes a mask addressable by name instead of by whatever id the
        renderer happened to hand out. Labels go on the ``class`` taxonomy, which is the
        one the segmentation annotators read, and USD resolves them down to descendants
        -- so labelling an asset's root Xform covers its meshes.

        Applied once, when the scene is prepared: it needs the stage populated, and
        ``_resolve_prims`` needs the simulation backend up.
        """
        from isaacsim.core.experimental.utils.semantics import add_labels

        labelled: Dict[str, list] = {}
        for label, expr in self.resolve_semantics().items():
            paths = self._resolve_prims(expr)
            if not paths:
                self._logger.warning(
                    f"[SceneOrchestrator] semantics: '{expr}' matched no prims, so "
                    f"nothing in the scene will carry the identifier '{label}'."
                )
                continue
            for path in paths:
                try:
                    add_labels(path, labels=[label])
                except Exception as e:
                    self._logger.warning(f"Could not label '{path}' as '{label}': {e}")
                    continue
                labelled.setdefault(label, []).append(path)

        if labelled:
            self._logger.info(
                f"[SceneOrchestrator] semantic identifiers: "
                f"{ {k: len(v) for k, v in labelled.items()} }"
            )
        return labelled

    def setup_dataset(self):
        # Deferred import: Isaac Sim extension modules only become importable
        # after SimulationApp has started and enabled the extension.
        from isaacsim.core.prims import SingleArticulation

        self.robots_views = {}
        self.ee_views = {}
        self.obs_masks = {}  # robot_name -> attr -> { 'indices': [], 'keys': [] }
        self.act_masks = {}  # robot_name -> attr -> { 'indices': [], 'keys': [] }

        dataset_cfg = self._config.get("dataset", {})
        obs_cfg = dataset_cfg.get("observations", [])
        act_cfg = dataset_cfg.get("action", [])

        def parse_mapping(cfg_list):
            mapping = {}
            cartesian_keys = {"x", "y", "z", "wx", "wy", "wz"}
            for item in cfg_list:
                for ds_key, target in item.items():
                    if ds_key in cartesian_keys:
                        continue
                    parts = target.split(".")
                    if len(parts) >= 3:
                        r_name = parts[0]
                        attr = parts[-1]
                        j_name = ".".join(parts[1:-1])
                        mapping[ds_key] = (r_name, j_name, attr)
            return mapping

        obs_mapping = parse_mapping(obs_cfg)
        act_mapping = parse_mapping(act_cfg)

        required_robots = {r for r, _, _ in obs_mapping.values()}
        # Create views for robots strictly found in config using SingleArticulation
        for r_name in required_robots:
            if r_name in self._config.get("robots", {}):
                robot_cfg = self._config["robots"][r_name]
                robot_path = robot_cfg.get("path", f"/{r_name}")
                prim_path = f"/Scene_{self._scene_id}{robot_path}"
                try:
                    view = SingleArticulation(
                        prim_path=prim_path, name=f"{r_name}_view_{self._scene_id}"
                    )
                except TypeError:
                    view = SingleArticulation(
                        prim_paths_expr=prim_path, name=f"{r_name}_view_{self._scene_id}"
                    )
                if not view.handles_initialized:
                    view.initialize()
                self.robots_views[r_name] = view
                self._logger.info(f"Robot view '{r_name}': dof_names={view.dof_names}")

                # Setup end-effector view if configured
                ee_name = robot_cfg.get("end_effector_name")
                if ee_name:
                    import omni.usd

                    # Isaac Sim 5.x: the batched XFormPrimView was unified into
                    # XFormPrim (also matches multiple prims via prim_paths_expr).
                    from isaacsim.core.prims import XFormPrim

                    stage = omni.usd.get_context().get_stage()

                    # Try direct path first
                    ee_path = f"{prim_path}/{ee_name}"
                    actual_ee_path = None

                    if stage.GetPrimAtPath(ee_path).IsValid():
                        actual_ee_path = ee_path
                    else:
                        # Fallback: search the stage for a prim with this name under the robot root
                        robot_prim = stage.GetPrimAtPath(prim_path)
                        if robot_prim.IsValid():
                            from pxr import Usd

                            for prim in Usd.PrimRange(robot_prim):
                                if prim.GetName() == ee_name:
                                    actual_ee_path = str(prim.GetPath())
                                    break

                    if actual_ee_path:
                        try:
                            ee_view = XFormPrim(
                                prim_paths_expr=actual_ee_path,
                                name=f"{r_name}_ee_view_{self._scene_id}",
                            )

                            # Safely initialize without assuming handles_initialized exists
                            if hasattr(ee_view, "handles_initialized"):
                                if not ee_view.handles_initialized:
                                    ee_view.initialize()
                            else:
                                ee_view.initialize()

                            self.ee_views[r_name] = ee_view
                            self._logger.info(
                                f"Created XFormPrimView for end_effector: {actual_ee_path}"
                            )
                        except Exception as e:
                            self._logger.warning(
                                f"Failed to create ee_view for {actual_ee_path}: {e}"
                            )
                    else:
                        self._logger.warning(
                            f"Could not find any prim matching end_effector_name '{ee_name}' under '{prim_path}'"
                        )

        def find_dof_index(dof_names, j_name):
            """Find DOF index by exact match or suffix match (e.g. 'joint1' matches 'fr3_joint1')."""
            if j_name in dof_names:
                return dof_names.index(j_name)
            for idx, dof in enumerate(dof_names):
                if dof.endswith(j_name):
                    return idx
            return None

        def build_mask(mapping, mask_dict):
            for ds_key, (r_name, j_name, attr) in mapping.items():
                target_robot = r_name
                if target_robot not in self.robots_views:
                    for rv_name, rv in self.robots_views.items():
                        if find_dof_index(rv.dof_names, j_name) is not None:
                            target_robot = rv_name
                            break

                if target_robot in self.robots_views:
                    dof_names = self.robots_views[target_robot].dof_names
                    dof_idx = find_dof_index(dof_names, j_name)
                    if dof_idx is not None:
                        if target_robot not in mask_dict:
                            mask_dict[target_robot] = {}
                        if attr not in mask_dict[target_robot]:
                            mask_dict[target_robot][attr] = {"indices": [], "keys": []}

                        mask_dict[target_robot][attr]["indices"].append(dof_idx)
                        mask_dict[target_robot][attr]["keys"].append(ds_key)
                    else:
                        self._logger.warning(
                            f"Joint '{j_name}' not found in dof_names of robot '{target_robot}'. Available: {dof_names}"
                        )
                else:
                    self._logger.warning(
                        f"Robot '{target_robot}' not found in robots_views for ds_key='{ds_key}'"
                    )

        build_mask(obs_mapping, self.obs_masks)
        build_mask(act_mapping, self.act_masks)
        self._logger.info(f"obs_masks: {self.obs_masks}")
        self._logger.info(f"act_masks: {self.act_masks}")

    def record_step(self, current_step: int):
        observation = {"x": 0.0, "y": 0.0, "z": 0.0, "wx": 0.0, "wy": 0.0, "wz": 0.0}
        action = {"x": 0.0, "y": 0.0, "z": 0.0, "wx": 0.0, "wy": 0.0, "wz": 0.0}

        # ~~~~~~~~~~~~~~ Observations ~~~~~~~~~~~~~ #
        # Images
        if hasattr(self, "rgb_annotators"):
            for ds_key, annotator in self.rgb_annotators.items():
                data = annotator.get_data()
                if data is not None and getattr(data, "size", 0):
                    # Isaac Sim annotators return RGBA (4 channels), strip alpha for RGB
                    if data.ndim == 3 and data.shape[2] == 4:
                        data = data[:, :, :3]
                    observation[ds_key] = data

        if hasattr(self, "depth_annotators"):
            for ds_key, annotator in self.depth_annotators.items():
                data = annotator.get_data()
                if data is not None and getattr(data, "size", 0):
                    observation[ds_key] = depth_to_uint16_mm(data)

        for ds_key, annotator in getattr(self, "instance_annotators", {}).items():
            data = annotator.get_data()
            if data is not None and getattr(data.get("data"), "size", 0):
                observation[ds_key] = self.colorize_instances(data)

        # A frame missing a stream is dropped, not recorded: the recorder builds the
        # dataset's features from the first frame and every later one must match them.
        missing = [k for k in self._recorded_annotators() if k not in observation]
        if missing:
            self._logger.warning(f"[SceneOrchestrator] dropping frame, no data yet for {missing}")
            return None

        # Joint states
        if hasattr(self, "obs_masks"):
            for r_name, attr_masks in self.obs_masks.items():
                view = self.robots_views[r_name]
                for attr, mask in attr_masks.items():
                    joint_indices = mask["indices"]
                    if attr == "pos":
                        vals = view.get_joint_positions(joint_indices=joint_indices)
                    elif attr == "vel":
                        vals = view.get_joint_velocities(joint_indices=joint_indices)
                    elif attr == "acc":
                        vals = view.get_joint_accelerations(joint_indices=joint_indices)
                    elif attr == "eff":
                        vals = view.get_joint_efforts(joint_indices=joint_indices)
                    else:
                        continue

                    if vals is not None:
                        if hasattr(vals, "ndim") and vals.ndim > 1:
                            vals = vals[0]
                        elif not hasattr(vals, "__iter__"):
                            vals = [vals]
                        for val, key in zip(vals, mask["keys"]):
                            observation[key] = float(val)

        # ~~~~~~~~~~~~~~~~ Actions ~~~~~~~~~~~~~~~~ #
        if hasattr(self, "act_masks"):
            for r_name, attr_masks in self.act_masks.items():
                view = self.robots_views[r_name]
                for attr, mask in attr_masks.items():
                    joint_indices = mask["indices"]

                    vals = None
                    if hasattr(view, "get_applied_action"):
                        action_obj = view.get_applied_action()
                        if action_obj is not None:
                            if attr == "pos":
                                vals = action_obj.joint_positions
                            elif attr == "vel":
                                vals = action_obj.joint_velocities
                            elif attr == "eff":
                                vals = action_obj.joint_efforts

                            if vals is not None:
                                vals = np.asarray(vals)
                                if joint_indices is not None and len(joint_indices) > 0:
                                    vals = vals[joint_indices]

                    if vals is not None:
                        if hasattr(vals, "ndim") and vals.ndim > 1:
                            vals = vals[0]
                        elif not hasattr(vals, "__iter__"):
                            vals = [vals]
                        for val, key in zip(vals, mask["keys"]):
                            action[key] = float(val)

        # Cartesian Pose Delta from End Effector View
        if hasattr(self, "ee_views") and self.ee_views:
            dataset_cfg = self._config.get("dataset", {})
            cartesian_robot = dataset_cfg.get("cartesian_velocity_robot")
            if cartesian_robot is None:
                cartesian_robot = list(self.ee_views.keys())[0]

            if cartesian_robot in self.ee_views:
                ee_view = self.ee_views[cartesian_robot]

                # Get world pose of the end effector directly
                curr_pos, curr_rot = ee_view.get_world_poses()

                if curr_pos is not None:
                    if curr_pos.ndim > 1:
                        curr_pos = curr_pos[0]
                        curr_rot = curr_rot[0]

                    # Isaac Sim quaternions are usually [w, x, y, z]
                    r_curr = R.from_quat([curr_rot[1], curr_rot[2], curr_rot[3], curr_rot[0]])
                    abs_rotvec = r_curr.as_rotvec()

                    # Calculate deltas based on last recorded data
                    if (
                        not hasattr(self, "_last_recorded_obs_pose")
                        or self._last_recorded_obs_pose is None
                    ):
                        # For the first frame, delta is zero
                        obs_delta_pos = np.zeros(3)
                        obs_delta_rotvec = np.zeros(3)
                    else:
                        last_curr_pos, last_curr_rot = self._last_recorded_obs_pose

                        # Observation Delta (current - last_current)
                        obs_delta_pos = curr_pos - last_curr_pos
                        r_last_curr = R.from_quat(
                            [last_curr_rot[1], last_curr_rot[2], last_curr_rot[3], last_curr_rot[0]]
                        )
                        obs_delta_r = r_curr * r_last_curr.inv()
                        obs_delta_rotvec = obs_delta_r.as_rotvec()

                    # Store current for the next step
                    self._last_recorded_obs_pose = (curr_pos.copy(), curr_rot.copy())

                    # Add absolute pose to observation
                    observation["x"] = float(curr_pos[0])
                    observation["y"] = float(curr_pos[1])
                    observation["z"] = float(curr_pos[2])

                    observation["wx"] = float(abs_rotvec[0])
                    observation["wy"] = float(abs_rotvec[1])
                    observation["wz"] = float(abs_rotvec[2])

                    # Without kinematics solver, we don't have the target pose.
                    # We copy the actual execution delta to the action.
                    action["x"] = float(obs_delta_pos[0])
                    action["y"] = float(obs_delta_pos[1])
                    action["z"] = float(obs_delta_pos[2])

                    action["wx"] = float(obs_delta_rotvec[0])
                    action["wy"] = float(obs_delta_rotvec[1])
                    action["wz"] = float(obs_delta_rotvec[2])

        return {
            "timestamp": current_step,
            "observation": observation,
            "action": action,
            # The frame's LeRobot task is the GUIDE-EX task under way; a tree that announces
            # no task (block_bin) leaves the scene's own prompt. The procedure and the subtask
            # go in as language prompts.
            "task": self.prompts["task"] or self.task,
            "prompts": {"procedure": self.task, "subtask": self.prompts["subtask"]},
        }

    def clear_recording_history(self):
        self._last_recorded_obs_pose = None

    @abstractmethod
    def reset_lightweight(self):
        raise NotImplementedError("Lightweight reset is not implemented for this scene.")

    def finalize(self):
        self.recorder.put_record_data("FINALIZE")
        self.recorder.set_start_recording()
