from keras.callbacks import ProgbarLogger
from tqdm import tqdm


class TqdmProgbar(ProgbarLogger):
    def __init__(self, **kwargs):
        kwargs.setdefault('ncols', 150)
        self.kwargs = kwargs

    def on_train_begin(self, logs=None):
        _ = logs
        self.pbar = None
        self.postfix = {}

    def on_epoch_begin(self, epoch, logs=None):
        if self.pbar is None and self.target is not None:
            total = (self.epochs - epoch) * self.target
            self.pbar = tqdm(total=total, **self.kwargs)
        self.set_postfix(logs)

    def on_train_batch_end(self, batch, logs=None):
        _ = batch
        self.set_postfix(logs)
        self.update()

    def on_epoch_end(self, epoch, logs=None):
        _ = epoch
        self.set_postfix(logs)

    def on_train_end(self, logs=None):
        _ = logs
        if self.pbar is not None:
            self.pbar.close()
            self.pbar = None

    def set_current(self, t):
        if self.pbar is not None:
            self.pbar.update(t - self.pbar.n)

    def update(self, n=1):
        if self.pbar is not None:
            self.pbar.update(n)

    def set_postfix(self, logs):
        if (self.pbar is not None) and (logs is not None):
            self.postfix.update(logs)
            self.pbar.set_postfix(self.postfix)
