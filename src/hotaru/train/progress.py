import tensorflow as tf
import click


class ProgressCallback(tf.keras.callbacks.Callback):

    def __init__(self, name, total):
        self.name = name
        self.total = total

    def on_train_begin(self, logs=None):
        self.progress = click.progressbar(label=self.name, length=self.total)
        self.progress.entered = True
        self.progress.render_progress()

    def on_epoch_end(self, epoch, logs=None):
        self.progress.update(1)

    def on_train_end(self, logs=None):
        self.progress.render_finish()
