import tensorflow as tf
from tqdm import tqdm


class Progress(tqdm):
    def __init__(self, iterable=None, length=None, label=None, **args):
        super().__init__(iterable, desc=label, total=length, **args)


class ProgressCallback(tf.keras.callbacks.Callback):
    """Progress Callback"""

    def __init__(self, name, total=None):
        self.name = name
        self.total = total

    def on_train_begin(self, logs=None):
        self.progress = Progress(
            label=self.name,
            length=self.params.get("epochs", self.total),
            unit="epoch",
        )

    def on_epoch_end(self, epoch, logs=None):
        self.progress.set_postfix(dict(score=logs["score"]))
        self.progress.update(1)

    def on_train_end(self, logs=None):
        self.progress.close()
