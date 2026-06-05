import multiprocessing
import time
from multiprocessing.managers import BaseManager


class IsaacSimServer:
    def __init__(self, config, debug):
        # Local imports to prevent CUDA initialization in the parent process during load
        from guide_core.core.runtime import IsaacSimRuntime
        from guide_core.scene.scene_manager import SceneManager

        self.runtime = IsaacSimRuntime(config=config, debug=debug)
        self.scene_manager = SceneManager(logger=self.runtime._logger)

        if self.runtime._world:
            self.runtime._world.add_physics_callback(
                "scene_manager_step", self.scene_manager.step(self.runtime)
            )

    def get_runtime(self):
        return self.runtime

    def get_scene_manager(self):
        return self.scene_manager


def run_isaac_server(config, debug, address, authkey):
    """Runs in the background process."""
    # Ensure SimulationApp initializes in the main thread of this process
    server_instance = IsaacSimServer(config, debug)

    class IsaacBaseManager(BaseManager):
        pass

    IsaacBaseManager.register("get_runtime", callable=lambda: server_instance.get_runtime())
    IsaacBaseManager.register(
        "get_scene_manager", callable=lambda: server_instance.get_scene_manager()
    )

    manager = IsaacBaseManager(address=address, authkey=authkey)
    server = manager.get_server()
    print("[IsaacManager] Starting Isaac Sim Server RPC in background thread...")
    from threading import Thread

    t = Thread(target=server.serve_forever, daemon=True)
    t.start()

    print("[IsaacManager] Starting Isaac Sim physics loop in main thread...")
    try:
        server_instance.runtime.run_loop()
    except BaseException as e:
        print(f"[IsaacManager] [FATAL] run_loop terminated: {type(e).__name__}: {e}")
        import traceback

        traceback.print_exc()
    print("[IsaacManager] [FATAL] Isaac Sim process main function exiting!")


def start_isaac_process(config, debug, address=("127.0.0.1", 50060), authkey=b"isaac_sim"):
    """Starts the background process and returns the connected manager client."""
    ctx = multiprocessing.get_context("forkserver")
    p = ctx.Process(target=run_isaac_server, args=(config, debug, address, authkey), daemon=True)
    p.start()

    class IsaacBaseManager(BaseManager):
        pass

    IsaacBaseManager.register("get_runtime")
    IsaacBaseManager.register("get_scene_manager")

    manager = IsaacBaseManager(address=address, authkey=authkey)

    # Wait for server to start
    connected = False
    for _ in range(300):  # Wait up to 30 seconds
        try:
            manager.connect()
            connected = True
            break
        except ConnectionRefusedError:
            time.sleep(0.1)

    if not connected:
        raise RuntimeError("Failed to connect to IsaacSimServer background process!")

    return manager, p
