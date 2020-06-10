import tensorflow as tf
import tensorflow.keras.backend as K


class Callback(tf.keras.callbacks.Callback):

    def on_epoch_begin(self, epoch, logs=None):
        self.model.optimizer.start = self.model.optimizer.iterations

    def on_epoch_end(self, epoch, logs=None):
        vs = self.model.trainable_weights
        for v in vs:
            o = self.model.optimizer.get_slot(v, 'old')
            K.set_value(v, K.get_value(o))
