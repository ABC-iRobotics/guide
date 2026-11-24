"""Make a task runnable from a Register path: fetch it, build it with its dependencies, launch it.

A Register path is one of
  - an installed package name ("block_bin")          -> used as it is
  - a directory                                      -> a task bundle, or one of GUIDE's own
                                                        scene layouts (e.g. guide_core/dummy_scene)
  - s3://bucket/key.tar.gz holding a task bundle     -> downloaded and unpacked first
A task bundle holds exactly one task package (<pkg>/<pkg>/scene.py beside its package.xml), the
packages it depends on, and optionally requirements.txt (pip). System dependencies come from
the package.xml files through rosdep. Bundles are built into one overlay, <workdir>/install,
which this process and every launch then use.
"""

from __future__ import annotations

import contextlib
import os
import shlex
import shutil
import signal
import subprocess
import sys
import tarfile
import time
from pathlib import Path

DEFAULT_WORKDIR = Path.home() / ".guide" / "tasks"


def _run(cmd: list[str]) -> None:
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(f"{cmd[0]} failed: {(result.stderr or result.stdout).strip()[-2000:]}")


def split_s3(url: str) -> tuple[str, str]:
    bucket, _, key = url.removeprefix("s3://").partition("/")
    return bucket, key


def s3_client():
    import boto3  # in the venv already: an isaacsim dependency
    from botocore.config import Config

    # Endpoint and credentials from the environment (AWS_ENDPOINT_URL,
    # AWS_SHARED_CREDENTIALS_FILE, ...); path-style, which Ceph RGW and MinIO serve without
    # wildcard DNS.
    return boto3.client("s3", config=Config(s3={"addressing_style": "path"}))


def share_dir(name: str) -> Path | None:
    """The installed package's share directory; None if it is not installed."""
    from ament_index_python.packages import get_package_share_directory

    try:
        return Path(get_package_share_directory(name))
    except (LookupError, ValueError):  # PackageNotFoundError is a KeyError
        return None


def task_package(bundle: Path) -> str | None:
    """The bundle's task package (<pkg>/<pkg>/scene.py beside a package.xml); None if it has none."""
    found = [
        d.name
        for d in [bundle, *sorted(p for p in bundle.iterdir() if p.is_dir())]
        if (d / "package.xml").is_file() and (d / d.name / "scene.py").is_file()
    ]
    if len(found) > 1:
        raise ValueError(f"{bundle} holds more than one task package: {found}")
    return found[0] if found else None


class TaskBringup:
    def __init__(self, sim_id: int, workdir: Path = DEFAULT_WORKDIR, run=_run,
                 popen=subprocess.Popen, s3=None):
        self.sim_id = sim_id
        self.workdir = Path(workdir)
        self.install = self.workdir / "install"
        self._run = run
        self._popen = popen
        self._s3 = s3
        self._launches = []

    def prepare(self, path: str) -> tuple[str, str | None]:
        """What to register, and the package whose bringup can be launched for it (or None)."""
        pkg = path
        if path.startswith("s3://") or share_dir(path) is None:
            bundle = self._fetch(path)
            pkg = task_package(bundle)
            if pkg is None:
                return str(bundle), None  # one of GUIDE's own scene layouts: nothing to build
            share = share_dir(pkg)
            # Built unless this overlay has it: a copy installed elsewhere (the image's) is not this bundle.
            if share is None or not share.resolve().is_relative_to(self.install.resolve()):
                self._build(bundle, pkg)
        share = share_dir(pkg)
        return pkg, pkg if share and (share / "launch" / "bringup.launch.py").is_file() else None

    def _fetch(self, path: str) -> Path:
        if not path.startswith("s3://"):
            bundle = Path(path).expanduser()
            if not bundle.is_dir():
                raise FileNotFoundError(f"{path} is neither an installed package nor a directory")
            return bundle
        bucket, key = split_s3(path)
        archive = self.workdir / "downloads" / Path(key).name
        archive.parent.mkdir(parents=True, exist_ok=True)
        (self._s3 or s3_client()).download_file(bucket, key, str(archive))
        name = archive.name.removesuffix(".gz").removesuffix(".tar").removesuffix(".tgz")
        bundle = self.workdir / "src" / name
        shutil.rmtree(bundle, ignore_errors=True)  # a re-register must not keep the old version's files
        with tarfile.open(archive) as tar:
            tar.extractall(bundle, filter="data")  # no absolute paths, no escaping links
        return bundle

    def _build(self, bundle: Path, pkg: str) -> None:
        self._run(["rosdep", "install", "--from-paths", str(bundle), "--ignore-src", "-y"])
        reqs = bundle / "requirements.txt"
        if reqs.is_file():
            pins = os.environ.get("GUIDE_PINS")
            self._run(
                ["uv", "pip", "install", "--python", sys.executable, "-r", str(reqs)]
                + (["-c", pins] if pins else [])
            )
        self._run([
            "colcon", "--log-base", str(self.workdir / "log"), "build", "--merge-install",
            "--base-paths", str(bundle), "--build-base", str(self.workdir / "build"),
            "--install-base", str(self.install), "--packages-up-to", pkg,
        ])
        self.activate()

    def activate(self) -> None:
        """Make the overlay visible here: the scene's ament lookups and its imports."""
        prefix = str(self.install)
        paths = os.environ.get("AMENT_PREFIX_PATH", "")
        if prefix not in paths.split(os.pathsep):
            os.environ["AMENT_PREFIX_PATH"] = os.pathsep.join(p for p in (prefix, paths) if p)
        py = f"python{sys.version_info.major}.{sys.version_info.minor}"
        site = str(self.install / "lib" / py / "site-packages")
        if site not in sys.path:
            sys.path.insert(0, site)

    def launch(self, pkg: str, scene_id: int) -> None:
        """Start the task's MoveIt + solver for one scene: <pkg>/launch/bringup.launch.py."""
        cmd = (
            f"exec ros2 launch {shlex.quote(pkg)} bringup.launch.py "
            f"sim_id:={self.sim_id} first_scene:={scene_id} num_env:=1"
        )
        setup = self.install / "setup.bash"
        if setup.is_file():
            cmd = f"source {shlex.quote(str(setup))} && {cmd}"
        self._launches.append(self._popen(["bash", "-c", cmd], start_new_session=True))

    def shutdown(self, timeout: float = 30.0) -> None:
        for p in self._launches:
            if p.poll() is None:
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(p.pid, signal.SIGINT)
        deadline = time.monotonic() + timeout
        for p in self._launches:
            try:
                p.wait(max(0.0, deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(p.pid, signal.SIGKILL)
