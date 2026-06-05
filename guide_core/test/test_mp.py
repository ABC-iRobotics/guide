import multiprocessing as mp

ctx = mp.get_context("spawn")


class Worker(ctx.Process):
    def __init__(self):
        super().__init__()
        self.q = ctx.Queue()

    def run(self):
        print("Worker run")


if __name__ == "__main__":
    w = Worker()
    w.start()
    w.join()
