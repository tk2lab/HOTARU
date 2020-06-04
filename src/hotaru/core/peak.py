# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np

from ..util.module import Module
from ..util.filter import gaussian, gaussian_laplace_multi


class Peak(Module):

    def build(self, base):
        self.base = base

        self.add_variable('gauss', ())
        self.add_variable('min_radius', ())
        self.add_variable('max_radius', ())
        self.add_variable('num_radius', (), tf.int32)
        self.add_variable('thr_gl', ())
        self.add_variable('thr_dist', ())

        self.add_variable('ps', (None,3), tf.int32)
        self.add_variable('rs', (None,))
        self.add_variable('gs', (None,))

    @property
    def radius(self):
        return tf.linspace(self.min_radius, self.max_radius, self.num_radius)

    def find(self):
        self.ps.assign(tf.zeros([0,3], tf.int32))
        self.rs.assign(tf.zeros([0]))
        self.gs.assign(tf.zeros([0]))

        imgs = self.base.imgs.to_dataset()
        pos = tf.cast(tf.where(self.base.imgs.mask), tf.int32)
        shape = tf.shape(self.base.imgs.mask)
        e = tf.constant(0)
        for img in imgs:
            s, e = e, e + tf.shape(img)[0]
            img = tf.stack([tf.scatter_nd(pos, x, shape) for x in img])
            self._append_find(img, s)
            tf.print('*', end='')
        tf.print()
        tf.print(tf.size(self.rs))

    #@tf.function(input_signature=[])
    def reduce(self):
        flg = tf.argsort(self.gs)[::-1]
        y, x, r = tf.cast(self.ps[:,1], tf.float32), tf.cast(self.ps[:,2], tf.float32), self.rs
        idx = tf.TensorArray(tf.int32, size=0, dynamic_size=True)
        p = tf.constant(0)
        while tf.size(flg) > 0:
            i = flg[0]
            idx = idx.write(p, i)
            p += 1
            y0, x0, r0 = y[i], x[i], r[i]
            j = flg[1:]
            y1, x1 = tf.gather(y, j), tf.gather(x, j)
            cond = tf.square(y1-y0) + tf.square(x1-x0) >= tf.square(self.thr_dist*r0)
            flg = tf.boolean_mask(flg[1:], cond)
        idx = idx.stack()
        self.ps.assign(tf.gather(self.ps, idx))
        self.rs.assign(tf.gather(self.rs, idx))
        self.gs.assign(tf.gather(self.gs, idx))
        tf.print(tf.size(self.rs))


    @tf.function(input_signature=[
        tf.TensorSpec([None,None,None], tf.float32),
        tf.TensorSpec((), tf.int32),
    ])
    def _append_find(self, imgs, s):
        if self.gauss > 0.0:
            imgs = gaussian(imgs, self.gauss)
        gl = gaussian_laplace_multi(imgs, self.radius)
        max_gl = tf.nn.max_pool3d(
            gl[...,tf.newaxis], [1,3,3,3,1], [1,1,1,1,1], padding='SAME')[...,0]
        bit = tf.equal(gl, max_gl) & (gl > self.thr_gl) & self.base.imgs.mask[...,tf.newaxis]
        posr = tf.cast(tf.where(bit), tf.int32)
        pos = tf.stack([posr[:,0]+s, posr[:,1], posr[:,2]], axis=1)
        if tf.shape(posr)[0] > 0:
            self.ps.assign(tf.concat([self.ps, pos], axis=0))
            self.rs.assign(tf.concat([self.rs, tf.gather(self.radius, posr[:,3])], axis=0))
            self.gs.assign(tf.concat([self.gs, tf.gather_nd(gl, posr)], axis=0))
