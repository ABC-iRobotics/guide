from multiprocessing.managers import BaseManager


class Simulator:
    def __init__(self):
        self.state = 0

    def step(self):
        self.state += 1
        return self.state


class MyManager(BaseManager):
    pass


MyManager.register("Simulator", Simulator)

if __name__ == "__main__":
    manager = MyManager()
    manager.start()

    sim = manager.Simulator()
    print(sim.step())
    print(sim.step())
