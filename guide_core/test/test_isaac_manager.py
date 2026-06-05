import multiprocessing
import time
from multiprocessing.managers import BaseManager


class RealSimulator:
    def __init__(self):
        print("RealSimulator initialized in thread:", multiprocessing.current_process().name)
        self.state = 0

    def step(self):
        self.state += 1
        return self.state


def run_server():
    sim = RealSimulator()

    class MyManager(BaseManager):
        pass

    MyManager.register("get_sim", callable=lambda: sim)
    manager = MyManager(address=("127.0.0.1", 50010), authkey=b"secret")
    server = manager.get_server()
    print("Server starting...")
    server.serve_forever()


if __name__ == "__main__":
    ctx = multiprocessing.get_context("spawn")
    p = ctx.Process(target=run_server)
    p.start()
    time.sleep(1)

    class MyManager(BaseManager):
        pass

    MyManager.register("get_sim")
    m = MyManager(address=("127.0.0.1", 50010), authkey=b"secret")
    m.connect()
    proxy = m.get_sim()
    print("Client got state:", proxy.step())
    p.terminate()
