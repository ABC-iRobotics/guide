import time
from multiprocessing.managers import BaseManager

import numpy as np


class Simulator:
    def get_image(self):
        return np.random.randint(0, 255, (720, 1280, 4), dtype=np.uint8)


class MyManager(BaseManager):
    pass


MyManager.register("Simulator", Simulator)

if __name__ == "__main__":
    manager = MyManager()
    manager.start()

    sim = manager.Simulator()
    start = time.time()
    for _ in range(30):
        img = sim.get_image()
    print(f"30 frames took: {time.time() - start:.3f}s")
