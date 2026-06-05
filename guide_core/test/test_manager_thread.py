import time
from multiprocessing.managers import BaseManager
from threading import Thread


class Simulator:
    def __init__(self):
        self.state = 0
        self.running = True
        self.t = Thread(target=self._loop)
        self.t.start()

    def _loop(self):
        while self.running:
            self.state += 1
            time.sleep(0.1)

    def get_state(self):
        return self.state

    def stop(self):
        self.running = False
        self.t.join()


class MyManager(BaseManager):
    pass


MyManager.register("Simulator", Simulator)

if __name__ == "__main__":
    manager = MyManager()
    manager.start()

    sim = manager.Simulator()
    print("Initial:", sim.get_state())
    time.sleep(0.5)
    print("After 0.5s:", sim.get_state())
    sim.stop()
