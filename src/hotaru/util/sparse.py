# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division

import tensorflow.compat.v2 as tf
import numpy as np

from .module import Module


class SparseBase(Module):

    def build(self, h, w):
        self.h = h
        self.w = w
        self.add_variable('size', (None,), tf.int32)
        self.add_variable('pos', (None,2), tf.int32)
        self.add_variable('val', (None,))

    #@tf.function(input_signature=[])
    def to_sparse(self):
        nk = tf.size(self.size)
        ones = tf.ones(tf.size(self.val), tf.int64)
        idx = tf.RaggedTensor.from_row_lengths(ones, self.size)
        idx *= tf.range(tf.cast(nk, tf.int64))[:,tf.newaxis]
        idx = idx.flat_values
        y, x = tf.cast(self.pos[:,0], tf.int64), tf.cast(self.pos[:,1], tf.int64)
        pos = tf.stack([idx, y, x], axis=1)
        out = tf.sparse.SparseTensor(pos, self.val, (nk,self.h,self.w))
        return tf.sparse.reorder(out)

    #@tf.function
    def clear(self):
        self.size.assign(tf.zeros([0], tf.int32))
        self.pos.assign(tf.zeros([0,2], tf.int32))
        self.val.assign(tf.zeros([0], tf.float32))
