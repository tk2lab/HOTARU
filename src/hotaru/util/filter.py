# -*- coding: utf-8 -*-

"""Image Filters."""

from __future__ import absolute_import
from __future__ import division

import tensorflow.compat.v2 as tf
import numpy as np


def neighbor(imgs):
    imgs = imgs[...,tf.newaxis]
    fil = tf.convert_to_tensor([[1,1,1],[1,0,1],[1,1,1]], tf.float32) / 8.0
    fil = tf.reshape(fil, [3,3,1,1])
    return tf.nn.conv2d(imgs, fil, [1,1,1,1], padding='SAME')[...,0]


def gaussian(imgs, r):
    r = tf.convert_to_tensor(r, tf.float32)
    mr = tf.math.ceil(r)
    d = tf.square(tf.range(-4.0*mr, 4.0*mr+1.0, 1.0))
    r2 = tf.square(r)
    o0 = tf.exp(-d/(2.0*r2)) / np.sqrt(2.0*np.pi) / r
    tmp = imgs[...,tf.newaxis]
    tmp = tf.nn.conv2d(tmp, tf.reshape(o0, (1, -1, 1, 1)), (1,1), 'SAME')
    tmp = tf.nn.conv2d(tmp, tf.reshape(o0, (-1, 1, 1, 1)), (1,1), 'SAME')
    return tmp[...,0]


def gaussian_laplace(imgs, r):
    mr = tf.math.ceil(r)
    d = tf.square(tf.range(-4.0*mr, 4.0*mr+1.0, 1.0))
    r2 = tf.square(r)
    o0 = tf.exp(-d/(2.0*r2)) / tf.sqrt(2.0*np.pi) / r
    o2 = (1.0 - d/r2) * o0
    tmp = imgs[...,tf.newaxis]
    gl1 = tf.nn.conv2d(tmp, tf.reshape(o2, (1, -1, 1, 1)), (1,1), 'SAME')
    gl1 = tf.nn.conv2d(gl1, tf.reshape(o0, (-1, 1, 1, 1)), (1,1), 'SAME')
    gl2 = tf.nn.conv2d(tmp, tf.reshape(o2, (-1, 1, 1, 1)), (1,1), 'SAME')
    gl2 = tf.nn.conv2d(gl2, tf.reshape(o0, (1, -1, 1, 1)), (1,1), 'SAME')
    return (gl1 + gl2)[...,0]


def gaussian_laplace_multi(imgs, radius):
    tmp = tf.map_fn(lambda r: gaussian_laplace(imgs, r), radius)
    return tf.transpose(tmp, (1, 2, 3, 0))


def max_pos(gl, mask, min_gl=0.0):
    max_gl = tf.nn.max_pool3d(gl[...,tf.newaxis], [1,3,3,3,1], [1,1,1,1,1], padding='SAME')[...,0]
    gl = gl[...,1:-1]
    max_gl = max_gl[...,1:-1]
    bit = tf.equal(gl, max_gl) & (gl > min_gl) & mask[...,tf.newaxis]
    return gl, tf.cast(tf.where(bit), tf.int32)


@tf.function(input_signature=[
    tf.TensorSpec((None,None,None), tf.float32),
    tf.TensorSpec((None,None), tf.bool),
    tf.TensorSpec((), tf.float32),
    tf.TensorSpec((None,), tf.float32),
    tf.TensorSpec((), tf.float32),
])
def find_peak_local(imgs, mask, gauss, radius, min_gl):
    if gauss > 0.0:
        imgs = gaussian(imgs, gauss)
    gl = gaussian_laplace_multi(imgs, radius)
    max_gl = tf.nn.max_pool3d(
        gl[...,tf.newaxis], [1,3,3,3,1], [1,1,1,1,1], padding='SAME')[...,0]
    bit = tf.equal(gl, max_gl) & (gl > min_gl) & mask[...,tf.newaxis]
    posr = tf.cast(tf.where(bit), tf.int32)
    pos = posr[:,:3]
    g = tf.gather_nd(gl, posr)
    r = tf.gather(radius, posr[:,3])
    '''
    def gen_mean_max(p):
        t, y, x, r = p
        img = imgs[t, y-2:y+3, x-2:x+3]
        return tf.reduce_mean(img), tf.reduce_max(img)
    v, m = tf.map_fn(gen_mean_max, pos, (tf.float32, tf.float32))
    return t, y, x, r, g, v, m
    '''
    return pos, r, g

#@tf.function
def _find_peak_local(imgs, mask, radius, min_gl=0.0):

    def gen_mean_max(p):
        t, y, x, r = p
        img = imgs[t, y-2:y+3, x-2:x+3]
        return tf.reduce_mean(img), tf.reduce_max(img)

    gl = gaussian_laplace_multi(imgs, radius)
    gl, pos = max_pos(gl, mask, min_gl)
    t, y, x = pos[:,0], pos[:,1], pos[:,2]
    r = tf.gather(radius, pos[:,3] + 1)
    g = tf.gather_nd(gl, pos)
    v, m = tf.map_fn(gen_mean_max, pos, (tf.float32, tf.float32))
    return t, y, x, r, g, v, m
