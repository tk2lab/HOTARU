import time

import tensorflow as tf
from tqdm.auto import tqdm


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


class ProgbarLogger(tf.keras.callbacks.Callback):
    """ProgbarLogger"""

    def on_train_begin(self, logs=None):
        if self.params.get("verbose", 1) >= 1:
            self.progress = Progress(
                desc=self.model.name,
                total=self.params.get("epochs"),
                unit="epoch",
            )
            self.prev = None

    def on_epoch_begin(self, epoch, logs=None):
        if self.params.get("verbose", 1) >= 2:
            self.progress_batch = Progress(
                desc="local step",
                total=self.params.get("steps"),
                unit="step",
                leave=False,
            )

    def on_batch_end(self, batch, logs=None):
        if self.params.get("verbose", 1) >= 2:
            self.progress_batch.update(1)

    def on_epoch_end(self, epoch, logs=None):
        if self.params.get("verbose", 1) >= 2:
            self.progress_batch.close()
        if self.params.get("verbose", 1) >= 1:
            if self.prev is not None:
                diff = self.prev - logs["loss"]
                self.progress.set_postfix(dict(loss=logs["loss"], diff=diff))
            self.progress.update(1)
            self.prev = logs["loss"]
            if self.model.stop_training:
                self.progress.close()
