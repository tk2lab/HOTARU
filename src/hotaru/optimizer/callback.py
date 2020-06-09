import tensorflow as tf


class Callback(tf.keras.callbacks.Callback):

    def on_epoch_begin(self, epoch, logs=None):
        self.model.optimizer.start = self.model.optimizer.iterations
