import tensorflow.keras.backend as K
import tensorflow as tf


def hotaru_loss(variance):
    return 0.5 * K.log(variance)


class HotaruLoss(tf.keras.losses.Loss):

    def __init__(self, extract, variance, name='Variance'):
        super().__init__(name=name)
        self._extract = extract
        self._variance = variance

    def call(self, y_true, y_pred):
        footprint, spike = self._extract(y_pred)
        variance = self._variance((footprint, spike))
        return hotaru_loss(variance)
