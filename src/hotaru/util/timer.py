import time


class Timer:

    def __enter__(self):
        self.time = time.time()
        self.ptime = time.process_time()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.time = time.time() - self.time
        self.ptime = time.process_time() - self.ptime

    def get(self):
        return self.time, self.ptime
