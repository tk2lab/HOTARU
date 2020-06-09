import tensorflow as tf
import tensorflow.keras.backend as K


class Extract(tf.keras.layers.Layer):

    def __init__(self, nx, nu, name='Extract'):
        super().__init__(name=name, dtype=tf.float32)
        self.nk = self.add_weight('nk', (), tf.int32, trainable=False)
        self.nx = nx
        self.nu = nu

    def call(self, inputs):
        nk, nx, nu = self.nk, self.nx, self.nu
        inputs = tf.slice(inputs, [0, 0], [nk, nx + nu])
        return tf.split(inputs, [nx, nu], axis=1)
