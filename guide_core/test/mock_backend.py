from guide_core.types.isaac_state import IsaacState


class MockIsaacRuntime:
    """
    Minimális mock az IsaacSimRuntime helyettesítésére.
    Pontosan azokat a metódusokat implementálja,
    amiket a RuntimeFacade használ.
    """

    def __init__(self):
        self.state = IsaacState.STOPPED
        self.stage_config = {}
        self._running = False
        self._tick_count = 0

    # --- lifecycle ---
    def _cmd_add_scene(self, stage_config):
        self.state = IsaacState.READY
        return True

    def _cmd_start(self):
        self.state = IsaacState.RUNNING
        self._running = True
        return True

    def _cmd_pause(self):
        self.state = IsaacState.PAUSED
        self._running = False
        return True

    def _cmd_stop(self):
        self.state = IsaacState.STOPPED
        self._running = False
        return True

    def _cmd_shutdown(self):
        self.state = IsaacState.UNINITIALIZED
        self._running = False
        return True

    # --- call API ---
    def call(self, name, timeout, *args, **kwargs):
        name = "_cmd_" + name
        if not hasattr(self, name):
            raise RuntimeError(f"Unknown command: {name}")
        return getattr(self, name)(*args, **kwargs)

    def _process_commands(self, max_per_cycle=50):
        return True

    def step(self, n=1):
        self._tick_count += 1
        return True

    # --- update ---
    def update(self):
        if self._running:
            self._tick_count += 1

    def get_ticks(self):
        return self._tick_count
