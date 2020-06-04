# -*- coding: utf-8 -*- 
from __future__ import absolute_import
from __future__ import division

import tensorflow.compat.v2 as tf
import numpy as np

from ..util import Timer, tictoc, Dict
#from ..image.driver import segment_imgs
#from .clean import clean_imgs


class BlockBase(object):

    @staticmethod
    def _matmul(index, val, data):
        nd = tf.shape(index)[0]
        nk = tf.reduce_max(index[:,0]) + 1
        bs = tf.shape(val)[1]
        nt = tf.shape(data)[0]

        h, w = tf.shape(data)[1], tf.shape(data)[2]
        hh, ww = ((h + bs - 1) // bs) * bs, ((w + bs - 1) // bs) * bs
        data = tf.pad(data, [[0,0],[0,hh-h],[0,ww-w]])

        # TODO
        # - loop (x, y) and bind k
        out = tf.zeros((nk, nt))
        for i in tf.range(nd):
            k, y, x = index[i,0], index[i,1], index[i,2]
            dk = tf.slice(data, [0,y*bs,x*bs], [nt,bs,bs])
            update = tf.tensordot(val[i], dk, [[0,1], [1,2]])
            out = tf.tensor_scatter_nd_add(out, [[k]], update[tf.newaxis,:])
        return out

    @staticmethod
    def _sum(index, data):
        nk = tf.reduce_max(index[:,0]) + 1
        return tf.math.segment_sum(tf.reduce_sum(data, axis=(1,2)), index[:,0])

    @staticmethod
    def _out(index, data):
        s = BlockBase._sum(index, data)
        return s * s[:,tf.newaxis]

    @staticmethod
    def _cov(index, data):
        nk = tf.reduce_max(index[:,0]) + 1
        cov = tf.zeros((nk, nk))
        ny = tf.reduce_max(index[:,1]) + 1
        nx = tf.reduce_max(index[:,2]) + 1
        for y in tf.range(ny):
            for x in tf.range(nx):
                cond = tf.equal(index[:,1], y) & tf.equal(index[:,2], x)
                dyx = tf.boolean_mask(data, cond)
                cyx = tf.reshape(tf.tensordot(dyx, dyx, [[1,2], [1,2]]), [-1])
                idx = tf.boolean_mask(index[:,0], cond)
                n = tf.size(idx)
                idx = tf.stack([
                    tf.reshape(tf.tile(idx[:,tf.newaxis], [1, n]), [-1]),
                    tf.tile(idx, [n]),
                ], axis=1)
                cov = tf.tensor_scatter_nd_add(cov, idx, cyx)
        return cov


class BlockTensor(BlockBase):

    def __init__(self, imgs, bs):
        super().__init__()
        size = imgs.size.read_value()
        pos = imgs.pos.read_value()
        val = imgs.val.read_value()

        nk = tf.size(size)
        shape = tf.reduce_max(pos, axis=0) + 1
        block_shape = (shape + bs - 1) // bs
        shape = bs * block_shape
        nh, nw = block_shape[0], block_shape[1]
        nb = nh * nw

        rpos = tf.RaggedTensor.from_row_lengths(pos, size)
        rval = tf.RaggedTensor.from_row_lengths(val, size)
        index = tf.TensorArray(tf.int32, size=0, dynamic_size=True)
        data = tf.TensorArray(tf.float32, size=0, dynamic_size=True)
        i = tf.constant(0)
        for k in tf.range(nk):
            s = tf.reduce_min(rpos[k], axis=0) // bs
            e = tf.reduce_max(rpos[k], axis=0) // bs - s + 1
            d = tf.scatter_nd(rpos[k] - s*bs, rval[k], e*bs)
            for y in tf.range(e[0]):
                for x in tf.range(e[1]):
                    index = index.write(i, tf.stack([k, s[0]+y, s[1]+x]))
                    data = data.write(i, d[y*bs:(y+1)*bs,x*bs:(x+1)*bs])
                    i += 1
        self.index = index.stack()
        self.data = data.stack()

    def matmul(self, data):
        return self._matmul(self.index, self.data, data)

    def sum(self):
        return self._sum(self.index, self.data)

    def out(self):
        return self._out(self.index, self.data)

    def cov(self):
        return self._cov(self.index, self.data)
