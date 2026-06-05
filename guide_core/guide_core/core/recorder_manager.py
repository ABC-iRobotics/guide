import os
import signal
import socket
import subprocess
import time
from multiprocessing.managers import BaseManager

from guide_core.scene.scene_recorder import SceneRecorder


class RecorderManagerSingleton:
    """A wrapper class to manage multiple scene recorders safely within the manager."""

    def __init__(self):
        self.recorders = {}

    def get_recorder(self, package_name: str, task_name: str, config: dict):
        # We uniquely identify the recorder by task_name
        if task_name not in self.recorders:
            recorder = SceneRecorder(package_name, task_name, config)
            recorder.start()
            self.recorders[task_name] = recorder
        return self.recorders[task_name]


class RecorderBaseManager(BaseManager):
    pass


RecorderBaseManager.register("SceneRecorder", callable=None)
RecorderBaseManager.register(
    "RecorderManagerSingleton",
    RecorderManagerSingleton,
    method_to_typeid={"get_recorder": "SceneRecorder"},
)


def _kill_port_holder(port: int) -> bool:
    """Kill any process currently listening on the given TCP port. Returns True if a process was killed."""
    try:
        result = subprocess.run(["fuser", f"{port}/tcp"], capture_output=True, text=True, timeout=5)
        pids = result.stdout.strip().split()
        if not pids:
            return False
        my_pid = os.getpid()
        for pid_str in pids:
            try:
                pid = int(pid_str)
                if pid != my_pid:
                    os.kill(pid, signal.SIGKILL)
            except (ValueError, ProcessLookupError, PermissionError):
                pass
        # Give the OS a moment to release the socket
        time.sleep(0.5)
        return True
    except (FileNotFoundError, subprocess.TimeoutExpired):
        # fuser not available or timed out – fall through
        return False


def _is_port_free(host: str, port: int) -> bool:
    """Check whether a TCP port is available for binding."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            s.bind((host, port))
            return True
        except OSError:
            return False


class RecorderServer:
    _manager = None
    _singleton_proxy = None

    @classmethod
    def start_server(
        cls, address=("127.0.0.1", 50050), authkey=b"isaac_sim_recorder", max_retries: int = 3
    ):
        if cls._manager is not None:
            return cls._singleton_proxy

        host, port = address

        for attempt in range(max_retries):
            # If the port is occupied, try to free it
            if not _is_port_free(host, port):
                print(
                    f"[RecorderServer] Port {port} is in use (attempt {attempt + 1}/{max_retries}). Trying to free it..."
                )
                _kill_port_holder(port)
                time.sleep(1.0)
                if not _is_port_free(host, port):
                    print(f"[RecorderServer] Port {port} is still occupied after cleanup.")
                    continue

            try:
                cls._manager = RecorderBaseManager(address=(host, port), authkey=authkey)
                cls._manager.start()
                cls._singleton_proxy = cls._manager.RecorderManagerSingleton()
                print(f"[RecorderServer] Started on {host}:{port}")
                return cls._singleton_proxy
            except (OSError, EOFError) as e:
                print(f"[RecorderServer] Failed to start on {host}:{port}: {e}")
                cls._manager = None
                time.sleep(1.0)

        raise RuntimeError(
            f"[RecorderServer] Could not start after {max_retries} attempts on {host}:{port}. "
            f"Kill leftover processes manually: fuser -k {port}/tcp"
        )

    @classmethod
    def get_client(cls, address=("127.0.0.1", 50050), authkey=b"isaac_sim_recorder"):
        class ClientRecorderManager(BaseManager):
            pass

        ClientRecorderManager.register("SceneRecorder")
        ClientRecorderManager.register("RecorderManagerSingleton")

        manager = ClientRecorderManager(address=address, authkey=authkey)
        manager.connect()
        return manager.RecorderManagerSingleton()
