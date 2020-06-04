# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division

import tensorflow.compat.v2 as tf
import numpy as np
import heapq as hq


def get_segment_index(gl, y, x, mask):
    #return get_segment_index_tf(gl, y, x, mask)
    return tf.numpy_function(
        get_segment_index_py, [gl, y, x, mask], tf.int32
    )

def get_segment_index_py(img, y0, x0, mask):

    def push(y, x, dy, dx):
        yn, xn = y+dy, x+dx
        if mask[yn, xn] and (img[yn,xn] <= img[y,x]):
            q.append([yn, xn, dy, dx])

    pos = [[y0,x0]]
    q = []

    start = [[1,0],[1,1],[0,1],[-1,1],[-1,0],[-1,-1],[0,-1],[1,-1]]
    for dy, dx in start:
        push(y0, x0, dy, dx)

    while len(q) > 0:
        y, x, dy, dx = q[0]
        pos.append([y, x])
        q = q[1:]
        push(y, x, dy, dx)
        if dx == 0:
            push(y, x, dy, +1)
            push(y, x, dy, -1)
        elif dy == 0:
            push(y, x, +1, dx)
            push(y, x, -1, dx)

    return np.array(pos, np.int32)

@tf.function(input_signature=[
    tf.TensorSpec((None,None), tf.float32),
    tf.TensorSpec((), tf.int32), tf.TensorSpec((), tf.int32),
    tf.TensorSpec((None, None), tf.bool),
])
def get_segment_index_tf(img, y0, x0, mask):

    def push(y, x, dy, dx):
        yn, xn = y+dy, x+dx
        if mask[yn, xn]:
            if img[yn,xn] <= img[y,x]:
                q.enqueue([yn, xn, dy, dx])

    #mask = tf.pad(mask, [[0,1], [0,1]])
    pos = tf.TensorArray(tf.int32, size=0, dynamic_size=True)

    q = tf.queue.FIFOQueue(2000, [tf.int32]*4, [()]*4)
    i = tf.constant(0)
    pos = pos.write(i, [y0,x0])
    for dy in tf.constant([-1,0,1]):
        for dx in tf.constant([-1,0,1]):
            if tf.not_equal(dx, 0) & tf.not_equal(dy, 0):
                push(y0, x0, dy, dx)

    while q.size() > 0:
        y, x, dy, dx = q.dequeue()
        i += 1
        pos = pos.write(i, [y, x])
        push(y, x, dy, dx)
        if dx == 0:
            push(y, x, dy, +1)
            push(y, x, dy, -1)
        elif dy == 0:
            push(y, x, +1, dx)
            push(y, x, -1, dx)

    return pos.stack()


def get_segment_index_old(img, mask, y0, x0, min_shape=-np.Inf, max_size=10000):

    def push(y, x):
        if rest[y,x]:
            hq.heappush(q, (-img[y,x],y,x))
            rest[y,x] = False

    def pop():
        p, y, x = hq.heappop(q)
        pos.append([y, x])
        return y, x

    mask = tf.pad(mask, [[0,1],[0,1]])
    #dpos = [[-1,0],[1,0],[0,-1],[0,1],[-1,-1],[-1,1],[1,-1],[1,1]]
    dpos = [[-1,0],[1,0],[0,-1],[0,1]]
    h, w = img.shape

    rest = np.ones(img.shape, bool)
    pos = []
    q = []

    push(y0, x0)
    while len(q) > 0:
        y, x = pop()
        for dy, dx in dpos:
            yn, xn = y + dy, x + dx
            if mask[yn, xn] and (img[yn,xn] <= img[y,x]):
                push(yn, xn)

    return np.array(pos, np.int32)
