# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np

from ..util.module import Module


class DynamicsBase(Module):

    def build(self):
        self.add_variable('uvkernel', (None,))
        self.add_variable('vukernel', (None,))
        self.add_variable('pad', (), tf.int32)

    def u_to_v(self, u):
        return tf.keras.backend.conv1d(
            u[..., tf.newaxis], self.uvkernel[::-1, tf.newaxis, tf.newaxis],
            1, 'valid', 'channels_last',
        )[..., 0]

    def v_to_u(self, v):
        v = tf.pad(v, [[0,0],[2,0]], 'CONSTANT')
        u = tf.keras.backend.conv1d(
            v[..., tf.newaxis], self.vukernel[::-1, tf.newaxis, tf.newaxis],
            1, 'valid', 'channels_last',
        )[..., 0]
        return tf.pad(u, [[0, 0], [self.pad, 0]], 'CONSTANT')


class DoubleExp(DynamicsBase):

    def build(self):
        super().build()
        self.add_variable('hz', ())
        self.add_variable('tau1', ())
        self.add_variable('tau2', ())
        self.add_variable('ltau', ())

    def set(self, **args):
        for k, v in args.items():
            getattr(self, k).assign(v)
        t = tf.range(1.0, 2.0 + self.ltau*self.hz, dtype=tf.float32) / self.hz
        e1 = tf.math.exp(-t / self.tau1)
        e2 = tf.math.exp(-t / self.tau2)
        kernel = tf.math.abs(e1 - e2)
        kernel /= tf.reduce_max(kernel)
        ginv = tf.stack([tf.constant(1.0), -e1[0]-e2[0], e1[0]*e2[0]]) / kernel[0]
        self.uvkernel.assign(kernel)
        self.vukernel.assign(ginv)
        self.pad.assign(tf.size(kernel) - 1)
