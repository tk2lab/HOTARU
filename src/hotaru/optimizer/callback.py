import tensorflow.keras.backend as K
import tensorflow as tf


class Callback(tf.keras.callbacks.Callback):

    def on_epoch_begin(self, epoch, logs=None):
        iterations = K.get_value(self.model.optimizer.iterations)
        self.model.optimizer.start = iterations

    def on_epoch_end(self, epoch, logs=None):
        vs = self.model.trainable_weights
        for v in vs:
            o = self.model.optimizer.get_slot(v, 'old')
            K.set_value(v, K.get_value(o))
