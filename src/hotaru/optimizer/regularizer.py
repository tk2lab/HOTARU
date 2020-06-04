# -*- coding: utf-8 -*-

import tensorflow as tf

from ..util.module import Module


class L1(Module):

    def __init__(self, l=None, rank=2, name='MaxNormL1'):
        super().__init__(name=name)
        self.l = l
        self.axis = lambda: tf.range(1, rank)

    def loss(self, x):
        x = tf.math.abs(x)
        return self.l * tf.reduce_sum(tf.math.abs(x), axis=self.axis())

    def prox(self, y, eta):
        return tf.nn.relu(y - eta * self.l)


class MaxNormL1(Module):

    def __init__(self, l=None, rank=2, name='MaxNormL1'):
        super().__init__(name=name)
        self.l = l
        self.axis = lambda: tf.range(1, rank)

    def loss(self, x):
        x = tf.math.abs(x)
        m = tf.reduce_max(x, axis=self.axis())
        s = tf.reduce_sum(x, axis=self.axis())
        cond = tf.where(m > 0.0)
        return self.l * tf.reduce_sum(tf.gather(s, cond) / tf.gather(m, cond))

    def prox(self, y, eta):
        y = tf.nn.relu(y)
        m = tf.reduce_max(y, axis=self.axis(), keepdims=True)
        return tf.nn.relu(tf.where(tf.equal(y, m), y, y - eta * self.l / m))
