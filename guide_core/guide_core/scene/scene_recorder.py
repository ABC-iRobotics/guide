from __future__ import annotations

import datetime
import json
import logging
import queue
import shutil
import tempfile
import traceback
from pathlib import Path
from threading import Event, Thread

import numpy as np

# Schema version for the GUIDE metadata sidecar (guide_info.json / guide_episodes.jsonl).
GUIDE_META_SCHEMA = 1

#: Capture rate in Hz when a task config carries no ``dataset.fps``.
#:
#: One number for three consumers that used to hold three different ones: the interval
#: SceneManager samples the scene on, the rate the ROS 2 camera topics publish at, and
#: the ``fps`` written into the dataset metadata. While the key went unset those were
#: 10 (a ``getattr`` default nothing ever assigned), 60 (every rendered frame), and 30
#: (this file's old literal) -- so every dataset recorded in that window claims a rate
#: it was not recorded at. 10 is what the recorder actually did.
DEFAULT_FPS = 10


#: Where each GUIDE-EX layer's prompt is recorded. The TASK's ("Put the red cube on the blue
#: cube.") is each frame's LeRobot ``task`` (a tree that announces no task, like block_bin's,
#: leaves the scene's own prompt there; cube_stack's announces one on every frame). The
#: PROCEDURE's ("Stack the cubes.") and the SUBTASK's ("Pick up the red cube.") are
#: ``language_persistent`` styles: ``subtask`` is LeRobot's, ``procedure`` is GUIDE's,
#: registered with LeRobot by ``register_guide_styles`` wherever it is written or resolved.
PROMPT_STYLES = ("procedure", "subtask")


def register_guide_styles() -> None:
    """Make LeRobot accept GUIDE's ``procedure`` style as a persistent one (idempotent)."""
    from lerobot.datasets import language

    language.EXTENDED_STYLES.add("procedure")
    language.PERSISTENT_STYLES.add("procedure")


def write_language(root: Path, language: dict) -> bool:
    """Write saved episodes' procedure and subtask prompts into a finalized dataset, the
    LeRobot way.

    ``language`` maps an episode index to ``{style: [(frame_index, prompt), ...]}``, one
    entry per change. LeRobot (>= 0.6) keeps these in the ``language_persistent`` column as
    rows of that style, each active from its timestamp until the next one of its style --
    what a training recipe's ``active_at(t, style=subtask)`` reads. ``add_frame`` drops
    language values at record time, so they go in afterwards through LeRobot's own
    annotation writer, timestamps taken from the dataset's frames. A LeRobot without
    language columns gets nothing written: returns False.
    """
    try:
        from lerobot.annotations.steerable_pipeline.executor import Executor
        from lerobot.annotations.steerable_pipeline.reader import iter_episodes
        from lerobot.annotations.steerable_pipeline.staging import EpisodeStaging
        from lerobot.annotations.steerable_pipeline.writer import LanguageColumnsWriter
    except ImportError:
        return False

    register_guide_styles()
    root = Path(root)
    records = list(iter_episodes(root))
    with tempfile.TemporaryDirectory() as staging:
        for record in records:
            rows = [
                {
                    "role": "assistant",
                    "content": prompt,
                    "style": level,
                    "timestamp": record.frame_timestamps[frame_index],
                    "camera": None,
                    "tool_calls": None,
                }
                for level, changes in language.get(record.episode_index, {}).items()
                for frame_index, prompt in changes
            ]
            if rows:
                EpisodeStaging(Path(staging), record.episode_index).write("plan", rows)
        LanguageColumnsWriter().write_all(records, Path(staging), root)
    Executor._ensure_annotation_metadata_in_info(root)
    return True


class SceneRecorder(Thread):
    def __init__(self, package_name: str, task_name: str, config: dict):
        super().__init__(daemon=True)
        self.record_queue = queue.Queue(maxsize=60)
        self.start_recording_event = Event()
        self.stop_recording_event = Event()
        self.idle_event = Event()
        self.stop_flag = Event()
        self.shutdown_event = Event()
        # Set when a FINALIZE or SHUTDOWN has been written; _finalized_path is that dataset's
        # directory, "" when nothing was recorded since the previous one.
        self.finalized_event = Event()
        self._finalized_path = ""

        self.start_recording_event.clear()
        self.stop_recording_event.set()
        self.idle_event.set()

        self.package_name = package_name
        self.task_name = task_name
        self.config = config

        # Frames dropped because the record queue was full (recorder fell behind);
        # see put_record_data. Non-zero means the encoder can't keep up, not a stall.
        self._dropped_frames = 0

        # Base directory for datasets, set per StartRecording request; empty => ~/dataset.
        self._output_path = ""

        # GUIDE metadata sidecar written into <dataset>/meta/:
        #  - _run_meta: run-level constants (master_seed, ids) pushed once via set_run_meta()
        #  - _pending_episode_meta: per-episode payload (seed, values, task, target/goal,
        #    per-robot start_state, main_object pose) pushed by the orchestrator per episode
        self._run_meta: dict = {}
        self._pending_episode_meta: dict | None = None
        self._info_written = False

        # Procedure and subtask prompts: this episode's changes per style as (frame_index,
        # prompt), and those of every saved episode, written into the dataset when it is
        # finalized.
        self._episode_frames = 0
        self._language: dict = {}
        self._saved_language: dict = {}

        self.dataset = None
        self.LeRobotDataset = None

    def set_start_recording(self):
        self.start_recording_event.set()

    def clear_start_recording(self):
        self.start_recording_event.clear()

    def set_output_path(self, path: str):
        """Set the dataset base directory for the next dataset (empty => ~/dataset)."""
        self._output_path = path or ""

    def set_run_meta(self, meta: dict):
        """Run-level sidecar constants (master_seed, scene/sim id). Pushed once at registration."""
        self._run_meta = dict(meta or {})

    def set_pending_episode_meta(self, meta: dict):
        """Per-episode sidecar payload (seed, values, task, target/goal) for the next saved episode."""
        self._pending_episode_meta = dict(meta) if meta else None

    def wait_start_recording(self, timeout=None):
        return self.start_recording_event.wait(timeout)

    def clear_stop_recording(self):
        self.stop_recording_event.clear()

    def wait_stop_recording(self, timeout=None):
        return self.stop_recording_event.wait(timeout)

    def put_record_data(self, data):
        # This runs on the sim's physics-callback thread (via the recorder proxy),
        # so a full queue must NEVER block -- otherwise a recorder that falls behind
        # stalls the whole simulation and services stop responding.
        # Control signals (FINALIZE_EPISODE / DISCARD_EPISODE / FINALIZE / SHUTDOWN)
        # must always be delivered, but must NOT block either: stop_recording /
        # finalize_recording call this while holding the per-scene lock that
        # start_recording also needs, so a blocking put on a full queue wedges that
        # lock and every recording service hangs (and FINALIZE never reaches the
        # recorder, so the dataset is never finalized). Instead, evict droppable
        # frames to guarantee a slot -- control is rare, frames are droppable.
        if isinstance(data, str):
            while True:
                try:
                    self.record_queue.put_nowait(data)
                    return
                except queue.Full:
                    try:
                        self.record_queue.get_nowait()  # drop one buffered frame
                        self._dropped_frames += 1
                    except queue.Empty:
                        pass
        try:
            self.record_queue.put_nowait(data)
        except queue.Full:
            self._dropped_frames += 1
            logger = getattr(self, "_logger", None)
            if logger and (self._dropped_frames == 1 or self._dropped_frames % 100 == 0):
                logger.warning(
                    f"Recorder queue full: dropped {self._dropped_frames} frame(s); the "
                    f"recorder is behind but the simulation is not stalled."
                )

    def is_idle(self):
        return self.idle_event.is_set()

    def wait_shutdown(self, timeout=None):
        return self.shutdown_event.wait(timeout)

    def clear_finalized(self):
        self.finalized_event.clear()

    def wait_finalized(self, timeout=None):
        """The finalized dataset's directory ("" = nothing recorded), or None on timeout."""
        return self._finalized_path if self.finalized_event.wait(timeout) else None

    def _attach_file_log(self):
        """Send this recorder's logs to a file (idempotent, best-effort).

        Path: ~/.ros/log/guide_recorder_<task>_<pid>.log when that dir exists,
        else ~/guide_recorder_<task>_<pid>.log."""
        try:
            self._logger.setLevel(logging.INFO)
            if any(isinstance(h, logging.FileHandler) for h in self._logger.handlers):
                return
            import os

            log_dir = Path.home() / ".ros" / "log"
            log_dir = log_dir if log_dir.is_dir() else Path.home()
            fh = logging.FileHandler(log_dir / f"guide_recorder_{self.task_name}_{os.getpid()}.log")
            fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
            self._logger.addHandler(fh)
            self._logger.propagate = False
        except Exception:
            pass  # logging must never break recording

    def run(self):
        # We initialize the logger inside the process so it's isolated
        self._logger = logging.getLogger(f"SceneRecorder_{self.task_name}")
        # The recorder lives in a forked BaseManager process whose stdout is not
        # captured, so these logs (Saving episode / Episode saved / Exception in
        # recorder loop / Finalizing) otherwise vanish -- which is exactly the
        # trail needed to see where the recorder stalls. Mirror them to a file.
        self._attach_file_log()
        self._logger.info("Starting SceneRecorder thread...")

        try:
            from lerobot.datasets.lerobot_dataset import LeRobotDataset

            self.LeRobotDataset = LeRobotDataset
            self._logger.info("Successfully imported LeRobotDataset.")
        except ImportError:
            self.LeRobotDataset = None
            self._logger.warning(
                "Failed to import LeRobotDataset. Recording will be simulated but not written."
            )

        try:
            while not self.stop_flag.is_set():
                self._logger.info("Writer loop waiting for start recording event...")
                # Wake for a queued item too: a FINALIZE still writing clears the start event
                # that a SHUTDOWN queued meanwhile set, which would leave it waiting forever.
                while not self.start_recording_event.wait(1.0) and self.record_queue.empty():
                    pass
                if self.stop_flag.is_set():
                    self._logger.info("Stop flag set. Breaking writer loop.")
                    break

                self._logger.info("Recording session started.")
                episode_start_time = None

                while self.start_recording_event.is_set() or not self.record_queue.empty():
                    item = self.record_queue.get()

                    if item == "FINALIZE_EPISODE":
                        self._logger.info(
                            "Received FINALIZE_EPISODE indicator. Finalizing episode..."
                        )
                        self._finalize_episode()
                    elif item == "DISCARD_EPISODE":
                        self._logger.info(
                            "Received DISCARD_EPISODE indicator. Discarding episode..."
                        )
                        self._discard_episode()
                    elif item == "FINALIZE":
                        self._logger.info("Received FINALIZE indicator. Finalizing dataset...")
                        self._finalized_path = self._finalize_dataset()
                        self.finalized_event.set()
                        self.idle_event.set()
                        self.start_recording_event.clear()
                        break
                    elif item == "SHUTDOWN":
                        self._logger.info("Received SHUTDOWN indicator. Finalizing and exiting...")
                        self._finalized_path = self._finalize_dataset()
                        self.finalized_event.set()
                        self.stop_flag.set()
                        break
                    elif isinstance(item, dict):
                        self._logger.debug("Processing next queue frame dict.")
                        episode_start_time = self._process_frame(item, episode_start_time)

        except Exception as e:
            tb_str = traceback.format_exc()
            self._logger.error(f"Exception in recorder loop: {e}\n{tb_str}")
        finally:
            self._logger.info("Recorder loop exited. Finalizing dataset if not done.")
            self._finalize_dataset()
            self.shutdown_event.set()

    def _initialize_dataset(self, first_item: dict):
        if self.dataset is not None or self.LeRobotDataset is None:
            return

        # Second-resolution timestamp: lerobot's LeRobotDatasetMetadata.create does
        # `root.mkdir(exist_ok=False)`, so a second generation started within the same
        # minute reused this exact path and crashed with FileExistsError ("cannot call
        # the generation after generating one dataset"). Seconds make each run unique;
        # guard against an unlikely same-second collision with a numeric suffix.
        timestamp_str = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        # The dataset folder is created inside the StartRecording request's `path`; an
        # empty string falls back to ~/dataset. `~` is expanded.
        base_dir = (
            Path(self._output_path).expanduser() if self._output_path else Path.home() / "dataset"
        )
        dataset_path = base_dir / f"{self.task_name}_{timestamp_str}"
        _suffix = 1
        while dataset_path.exists():
            dataset_path = base_dir / f"{self.task_name}_{timestamp_str}_{_suffix}"
            _suffix += 1

        self._logger.info(f"Dataset path initialized at: {dataset_path}")

        # ~~~~~~~~~~~~~~ Observations ~~~~~~~~~~~~~ #
        obs_features = {}
        if "observation" in first_item:
            for k, v in first_item["observation"].items():
                if isinstance(v, np.ndarray) and v.ndim == 3:
                    obs_features[k] = v.shape
                else:
                    obs_features[k] = float

            obs_features = {
                **obs_features,
                "x": float,
                "y": float,
                "z": float,
                "wx": float,
                "wy": float,
                "wz": float,
            }

        # ~~~~~~~~~~~~~~~~ Actions ~~~~~~~~~~~~~~~~ #
        action_features = {}
        if "action" in first_item:
            for k in first_item["action"].keys():
                action_features[k] = float

            action_features = {
                **action_features,
                "x": float,
                "y": float,
                "z": float,
                "wx": float,
                "wy": float,
                "wz": float,
            }

        from lerobot.utils.feature_utils import hw_to_dataset_features

        obs_features = hw_to_dataset_features(obs_features, "observation", use_video=True)
        action_features = hw_to_dataset_features(action_features, "action", use_video=True)
        features = {**action_features, **obs_features}

        self._logger.info(f"Creating LeRobotDataset with features: {list(features)}")

        self.dataset = self.LeRobotDataset.create(
            repo_id=self.package_name,
            fps=self.config.get("dataset", {}).get("fps", DEFAULT_FPS),
            features=features,
            root=str(dataset_path),
            use_videos=True,
            # Async image writing so the recorder drains the queue fast enough to
            # keep up with the sim (synchronous writing was the bottleneck that
            # filled the queue and stalled the main loop). 0 -> synchronous.
            image_writer_threads=4,
            image_writer_processes=0,
        )
        self._logger.info("Successfully created LeRobotDataset.")

        # Run-level sidecar (master seed, config, provenance), written once.
        self._write_run_info(dataset_path)

    def _process_frame(self, item: dict, episode_start_time: float) -> float:
        if "timestamp" not in item:
            return episode_start_time

        current_time = item.pop("timestamp")
        if episode_start_time is None:
            episode_start_time = current_time
            self._initialize_dataset(item)

        if self.dataset is None:
            return episode_start_time

        relative_time = current_time - episode_start_time

        from lerobot.utils.feature_utils import build_dataset_frame

        observation_frame = build_dataset_frame(
            self.dataset.features, item.get("observation", {}), prefix="observation"
        )
        action_frame = build_dataset_frame(
            self.dataset.features, item.get("action", {}), prefix="action"
        )

        task_str = item.pop("task", self.task_name)
        prompts = item.pop("prompts", {})
        frame = {**observation_frame, **action_frame, "task": task_str}

        self.dataset.add_frame(frame)
        # LeRobot numbers an episode's frames 0, 1, ... in the order they are added. A
        # level is recorded where it changes to a new prompt. "" is not recorded: LeRobot
        # keeps a persistent row active until the next one and stores no empty row, so a
        # level keeps its last prompt until another one replaces it.
        for level, prompt in prompts.items():
            changes = self._language.setdefault(level, [])
            if prompt and prompt != (changes[-1][1] if changes else ""):
                changes.append((self._episode_frames, prompt))
        self._episode_frames += 1
        self._logger.info(
            f"Frame added successfully at time={current_time:.2f} (relative={relative_time:.2f}). Total frames: {len(self.dataset)}"
        )
        return episode_start_time

    def _finalize_episode(self):
        if self.dataset is not None:
            # LeRobot counts only saved episodes, so the index the about-to-be-saved
            # episode takes is the current total (read before save_episode increments it).
            meta_obj = getattr(self.dataset, "meta", None)
            if meta_obj is not None and hasattr(meta_obj, "total_episodes"):
                episode_index = int(meta_obj.total_episodes)
            else:
                episode_index = int(getattr(self.dataset, "num_episodes", 0))
            self._logger.info("Saving episode...")
            self.dataset.save_episode(parallel_encoding=False)
            self._logger.info("Episode successfully saved.")
            self._write_episode_meta(episode_index)
            if any(self._language.values()):
                self._saved_language[episode_index] = self._language
        self._language, self._episode_frames = {}, 0
        self._pending_episode_meta = None
        self.start_recording_event.clear()
        self.stop_recording_event.set()
        self.idle_event.set()

    def _discard_episode(self):
        if self.dataset is not None:
            self._logger.info("Discarding episode...")
            writer = self.dataset.writer
            episode_index = int(np.asarray(writer.episode_buffer["episode_index"]).reshape(-1)[0])
            self.dataset.clear_episode_buffer()  # waits for the image writer first
            # lerobot 0.6 deletes the buffered frames of image features only. Cameras are
            # video features, and the retry -- same episode index -- overwrites just its
            # own frames: a longer attempt's tail was encoded after the episode's last.
            for key in self.dataset.meta.video_keys:
                shutil.rmtree(writer._get_image_file_dir(episode_index, key), ignore_errors=True)
            self._logger.info("Episode buffer cleared.")
        self._language, self._episode_frames = {}, 0
        # Drop the pending sidecar payload so only saved episodes are recorded.
        self._pending_episode_meta = None
        self.start_recording_event.clear()
        self.stop_recording_event.set()
        self.idle_event.set()

    # ---- GUIDE metadata sidecar -----------------------------------------------

    def _write_run_info(self, dataset_path):
        """Write <dataset>/meta/guide_info.json (run constants) once per dataset."""
        if self._info_written:
            return
        try:
            meta_dir = Path(dataset_path) / "meta"
            meta_dir.mkdir(parents=True, exist_ok=True)
            info = {
                "guide_meta_schema": GUIDE_META_SCHEMA,
                "created": datetime.datetime.now().isoformat(timespec="seconds"),
                "scene": {
                    "dataset_name": self.task_name,
                    "scene_id": self._run_meta.get("scene_id"),
                    "sim_id": self._run_meta.get("sim_id"),
                },
                "task": {
                    "package": self.package_name,
                    "version": self._package_version(self.package_name),
                },
                "randomization": {
                    "master_seed": self._run_meta.get("master_seed"),
                    "grid": self._run_meta.get("grid"),
                },
                # Legend of the *_instance streams: tracked-object label -> RGB.
                "instance_colors": self._run_meta.get("instance_colors"),
                "config": self._curate_config(self.config),
                "provenance": self._collect_provenance(),
            }
            (meta_dir / "guide_info.json").write_text(json.dumps(info, indent=2, sort_keys=True))
            self._info_written = True
            self._logger.info(f"Wrote GUIDE run info to {meta_dir / 'guide_info.json'}")
        except Exception as e:
            self._logger.error(f"Failed to write guide_info.json: {e}")

    def _write_episode_meta(self, episode_index: int):
        """Append one JSON line to <dataset>/meta/guide_episodes.jsonl for a saved episode."""
        try:
            line = {"episode_index": int(episode_index), "schema_version": GUIDE_META_SCHEMA}
            if self._pending_episode_meta:
                line.update(self._pending_episode_meta)
            # Authoritative LeRobot index (never the orchestrator draw counter).
            line["episode_index"] = int(episode_index)
            meta_dir = Path(self.dataset.root) / "meta"
            meta_dir.mkdir(parents=True, exist_ok=True)
            with open(meta_dir / "guide_episodes.jsonl", "a") as f:
                f.write(json.dumps(line, sort_keys=True) + "\n")
        except Exception as e:
            self._logger.error(f"Failed to write guide episode meta: {e}")

    @staticmethod
    def _curate_config(cfg: dict) -> dict:
        """Curated scene config for the sidecar: full robot + camera blocks (so future
        domain randomization of joints / camera params & alignment is reproducible) plus
        the USD asset and dataset config. ``origin``/``limits`` are omitted — they live
        in the task's own configuration."""
        if not isinstance(cfg, dict):
            return {}
        keys = ("usd_path", "robots", "cameras", "dataset", "startup", "world")
        return {k: cfg[k] for k in keys if k in cfg}

    @staticmethod
    def _package_version(pkg: str):
        # 1. pip / dist metadata (e.g. isaacsim).
        try:
            from importlib.metadata import PackageNotFoundError, version

            try:
                return version(pkg)
            except PackageNotFoundError:
                pass
        except Exception:
            pass
        # 2. ROS package.xml <version> (ament packages carry no pip dist metadata).
        try:
            import xml.etree.ElementTree as ET

            from ament_index_python.packages import get_package_share_directory

            share = get_package_share_directory(pkg)
            v = ET.parse(Path(share) / "package.xml").getroot().findtext("version")
            return v.strip() if v else None
        except Exception:
            return None

    def _collect_provenance(self) -> dict:
        import os
        import subprocess
        import sys

        prov = {
            "ros_distro": os.environ.get("ROS_DISTRO"),
            "python": sys.version.split()[0],
            "isaac_sim": self._package_version("isaacsim"),
            "guide_commit": None,
        }
        # Short commit — best-effort; an installed (copy) build has no .git.
        try:
            import guide_core

            src = Path(guide_core.__file__).resolve().parent
            out = subprocess.run(
                ["git", "-C", str(src), "rev-parse", "--short", "HEAD"],
                capture_output=True,
                text=True,
                timeout=3,
            )
            prov["guide_commit"] = out.stdout.strip() or None
        except Exception:
            pass
        return prov

    def _finalize_dataset(self) -> str:
        dataset_dir = ""
        if self.dataset is not None:
            dataset_root = self.dataset.root
            dataset_dir = str(dataset_root)
            self._logger.info(f"Finalizing dataset at {dataset_root}...")
            self.dataset.finalize()
            self.dataset = None
            # The next recording is a new dataset, with its own guide_info.json.
            self._info_written = False
            self._logger.info("Dataset finalized successfully.")

            if self._saved_language:
                try:
                    written = write_language(dataset_root, self._saved_language)
                    self._logger.info(
                        f"Procedure and subtask prompts of {len(self._saved_language)} episode(s) "
                        f"{'written' if written else 'skipped: this LeRobot has no language columns'}."
                    )
                except Exception as e:
                    self._logger.error(f"Writing prompts failed: {e}\n{traceback.format_exc()}")
                self._saved_language = {}

            # Verify dataset files exist on disk (local only, no Hub access)
            try:
                self._logger.info(f"Verifying dataset files at {dataset_root}...")
                root = Path(dataset_root)
                info_path = root / "meta" / "info.json"
                if info_path.exists():
                    with open(info_path) as f:
                        info = json.load(f)
                    self._logger.info(
                        f"Dataset verification: info.json loaded. Total episodes: {info.get('total_episodes', '?')}, Total frames: {info.get('total_frames', '?')}"
                    )
                else:
                    self._logger.warning(
                        f"Dataset verification: info.json not found at {info_path}"
                    )

                data_dir = root / "data"
                if data_dir.exists():
                    parquet_files = list(data_dir.rglob("*.parquet"))
                    self._logger.info(
                        f"Dataset verification: {len(parquet_files)} parquet file(s) found."
                    )
                else:
                    self._logger.warning("Dataset verification: no data directory found.")
            except Exception as e:
                tb_str = traceback.format_exc()
                self._logger.error(f"Dataset verification failed: {e}\n{tb_str}")

        self.start_recording_event.clear()
        self.stop_recording_event.set()
        self.idle_event.set()
        return dataset_dir
