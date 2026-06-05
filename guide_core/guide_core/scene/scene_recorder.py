from __future__ import annotations

import datetime
import json
import logging
import queue
import traceback
from pathlib import Path
from threading import Event, Thread

import numpy as np


class SceneRecorder(Thread):
    def __init__(self, package_name: str, task_name: str, config: dict):
        super().__init__(daemon=True)
        self.record_queue = queue.Queue(maxsize=60)
        self.start_recording_event = Event()
        self.stop_recording_event = Event()
        self.idle_event = Event()
        self.stop_flag = Event()
        self.shutdown_event = Event()

        self.start_recording_event.clear()
        self.stop_recording_event.set()
        self.idle_event.set()

        self.package_name = package_name
        self.task_name = task_name
        self.config = config

        self.dataset = None
        self.LeRobotDataset = None

    def set_start_recording(self):
        self.start_recording_event.set()

    def clear_start_recording(self):
        self.start_recording_event.clear()

    def wait_start_recording(self, timeout=None):
        return self.start_recording_event.wait(timeout)

    def clear_stop_recording(self):
        self.stop_recording_event.clear()

    def wait_stop_recording(self, timeout=None):
        return self.stop_recording_event.wait(timeout)

    def put_record_data(self, data):
        self.record_queue.put(data)

    def set_idle(self):
        self.idle_event.set()

    def is_idle(self):
        return self.idle_event.is_set()

    def wait_shutdown(self, timeout=None):
        return self.shutdown_event.wait(timeout)

    def run(self):
        # We initialize the logger inside the process so it's isolated
        self._logger = logging.getLogger(f"SceneRecorder_{self.task_name}")
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
                self.start_recording_event.wait()
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
                        self._finalize_dataset()
                        self.idle_event.set()
                        self.start_recording_event.clear()
                        break
                    elif item == "SHUTDOWN":
                        self._logger.info("Received SHUTDOWN indicator. Finalizing and exiting...")
                        self._finalize_dataset()
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

        timestamp_str = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M")
        dataset_path = Path.home() / "dataset" / f"{self.task_name}_{timestamp_str}"

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

        from lerobot.datasets.utils import hw_to_dataset_features

        obs_features = hw_to_dataset_features(obs_features, "observation", use_video=True)
        action_features = hw_to_dataset_features(action_features, "action", use_video=True)
        features = {**action_features, **obs_features}

        self._logger.info(f"Creating LeRobotDataset with features: {list(features)}")

        self.dataset = self.LeRobotDataset.create(
            repo_id=self.package_name,
            fps=self.config.get("dataset", {}).get("fps", 30),
            features=features,
            root=str(dataset_path),
            use_videos=True,
            image_writer_threads=0,
            image_writer_processes=0,
        )
        self._logger.info("Successfully created LeRobotDataset.")

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

        from lerobot.datasets.utils import build_dataset_frame

        observation_frame = build_dataset_frame(
            self.dataset.features, item.get("observation", {}), prefix="observation"
        )
        action_frame = build_dataset_frame(
            self.dataset.features, item.get("action", {}), prefix="action"
        )

        task_str = item.pop("task", self.task_name)
        frame = {**observation_frame, **action_frame, "task": task_str}

        self.dataset.add_frame(frame)
        self._logger.info(
            f"Frame added successfully at time={current_time:.2f} (relative={relative_time:.2f}). Total frames: {len(self.dataset)}"
        )
        return episode_start_time

    def _finalize_episode(self):
        if self.dataset is not None:
            self._logger.info("Saving episode...")
            self.dataset.save_episode(parallel_encoding=False)
            self._logger.info("Episode successfully saved.")
        self.start_recording_event.clear()
        self.stop_recording_event.set()
        self.idle_event.set()

    def _discard_episode(self):
        if self.dataset is not None:
            self._logger.info("Discarding episode...")
            self.dataset.clear_episode_buffer()
            self._logger.info("Episode buffer cleared.")
        self.start_recording_event.clear()
        self.stop_recording_event.set()
        self.idle_event.set()

    def _finalize_dataset(self):
        if self.dataset is not None:
            dataset_root = self.dataset.root
            self._logger.info(f"Finalizing dataset at {dataset_root}...")
            self.dataset.finalize()
            self.dataset = None
            self._logger.info("Dataset finalized successfully.")

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
