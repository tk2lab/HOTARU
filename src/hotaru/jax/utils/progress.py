import time

from tqdm.auto import tqdm


class SimpleProgress:
    def __init__(self, total):
        self.total = total
        self.n = 0
        self.update(0)

    def update(self, n):
        self.n += n

    @property
    def value(self):
        return 100 * self.n / self.total

    @property
    def label(self):
        return f"{self.n} / {self.total}"


class Progress(tqdm):
    """Progress"""

    def __init__(self, *args, shard=None, batch=None, **kwargs):
        super().__init__(*args, **kwargs)
        if shard is not None:
            self.iterable = self.iterable.shard(shard, 0)
            self.total = (self.total + shard - 1) // shard
        if batch is not None:
            self.iterable = self.iterable.batch(batch)
            self.counter_type = "batch"
        else:
            self.counter_type = None

    def __iter__(self):
        if self.disable:
            for obj in self.iterable:
                yield obj
            return

        counter_type = self.counter_type
        mininterval = self.mininterval
        min_start_t = self.start_t + self.delay
        n = self.n
        last_print_n = self.last_print_n
        last_print_t = self.last_print_t
        try:
            for obj in self.iterable:
                yield obj
                if counter_type == "batch":
                    if isinstance(obj, tuple):
                        obj = obj[0]
                    if hasattr(obj, "numpy"):
                        obj = obj.numpy()
                    n += len(obj)
                else:
                    n += 1
                if n - last_print_n >= self.miniters:
                    cur_t = time.time()
                    dt = cur_t - last_print_t
                    if dt >= mininterval and cur_t >= min_start_t:
                        self.update(n - last_print_n)
                        last_print_n = self.last_print_n
                        last_print_t = self.last_print_t
        finally:
            self.n = n
            self.close()
