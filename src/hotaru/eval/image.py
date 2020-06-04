# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division

import tensorflow.compat.v2 as tf
import numpy as np

from ..util.timer import Timer, tictoc
from ..util.filter import neighbor


@tf.function(input_signature=[
    tf.TensorSpec((), tf.variant), tf.TensorSpec((), tf.int32),
    tf.TensorSpec((None,None), tf.bool),
])
def calc_stats(imgs, nt, mask):
    imgs = tf.data.experimental.from_variant(imgs, tf.TensorSpec([None,None,None], tf.float32))

    avg_t = tf.zeros((nt,))
    avg_x = tf.zeros_like(mask, tf.float32)
    std = tf.constant(0.0)
    e = tf.constant(0)
    for img in imgs:
        #tf.autograph.experimental.set_loop_options(
        #    parallel_iterations=1,
        #    swap_memory=True,
        #)
        s, e = e, e + tf.shape(img)[0]
        masked = tf.boolean_mask(img, mask, axis=1)
        mean = tf.reduce_mean(masked, axis=1)
        avg_t = tf.tensor_scatter_nd_update(avg_t, tf.range(s, e)[:,tf.newaxis], mean)
        avg_x += tf.reduce_sum(img, axis=0)
        std += tf.reduce_sum(tf.square(masked - mean[:,tf.newaxis]))
        tf.print('*', end='')
    tf.print()

    nt = tf.cast(nt, tf.float32)
    nx = tf.cast(tf.math.count_nonzero(mask), tf.float32)
    avg_x /= nt
    avg0 = tf.reduce_mean(avg_t)
    avg_t -= avg0
    avgxm = tf.boolean_mask(avg_x - avg0, mask)
    std = tf.sqrt(std / (nt * nx) - tf.reduce_mean(tf.square(avgxm)))
    tf.print(tf.reduce_min(avg_t), tf.reduce_max(avg_t))
    tf.print(tf.reduce_min(avg_x), tf.reduce_max(avg_x))
    tf.print(std)
    return avg_t, avg_x, std

#@tf.function(input_signature=[
#    tf.TensorSpec((), tf.variant),
#    tf.TensorSpec((None,None), tf.bool),
#])
def _calc_stats(imgs, mask):
    #imgs = tf.data.experimental.from_variant(imgs, tf.TensorSpec([None,None,None], tf.float32))
    nx = tf.cast(tf.math.count_nonzero(mask), tf.float32)

    avg_x = tf.zeros_like(mask, tf.float32)
    nt = tf.constant(0.0)
    for img in imgs:
        #tf.autograph.experimental.set_loop_options(
            #parallel_iterations=1,
            #swap_memory=True,
        #)
        avg_x += tf.reduce_sum(img, axis=0)
        nt += tf.cast(tf.shape(img)[0], tf.float32)
        tf.print('*', end='')
    tf.print()
    avg_x /= nt

    avg_t = tf.zeros([0])
    std = tf.constant(0.0)
    for img in imgs:
        #tf.autograph.experimental.set_loop_options(
        #    parallel_iterations=1,
        #    swap_memory=True,
        #)
        masked = tf.boolean_mask(img - avg_x, mask, axis=1)
        mean = tf.reduce_mean(masked, axis=1)
        avg_t = tf.concat([avg_t, mean], axis=0)
        std += tf.reduce_sum(tf.square(masked - mean[:,tf.newaxis]))
        tf.print('*', end='')
    tf.print()
    std = tf.sqrt(std / (nt * nx))

    return avg_t, avg_x, std


@tf.function
def calc_max(imgs):
    mxx = tf.constant(-np.Inf)
    for img in imgs:
        tmp = tf.reduce_max(img, axis=0)
        mxx = tf.where(mxx > tmp, mxx, tmp)
    return mxx


@tf.function
def calc_cor(imgs):
    sx1, sx2, sy1, sy2, sxy = (tf.constant(0.0) for _ in range(5))
    count = tf.constant(0, tf.int32)
    for img in imgs:
        nei = neighbor(img)
        sx1 += tf.reduce_sum(img, axis=0)
        sy1 += tf.reduce_sum(nei, axis=0)
        sx2 += tf.reduce_sum(tf.square(img), axis=0)
        sxy += tf.reduce_sum(img * nei, axis=0)
        sy2 += tf.reduce_sum(tf.square(nei), axis=0)
        count += tf.shape(img)[0]
    ntf = tf.cast(count, tf.float32)
    avg_x = sx1 / ntf
    avg_y = sy1 / ntf
    cov_xx = sx2 / ntf - tf.square(avg_x)
    cov_xy = sxy / ntf - avg_x * avg_y
    cov_yy = sy2 / ntf - tf.square(avg_y)
    return cov_xy / tf.sqrt(cov_xx * cov_yy)    
