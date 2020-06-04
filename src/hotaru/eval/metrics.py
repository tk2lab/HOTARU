# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division

import tensorflow.compat.v2 as tf

import tensorflow.keras.backend as K
from tensorflow.python.keras.metrics import Metric


def min_peak(y_true, y_pred):
    return K.min(K.max(y_pred, axis=1))

def max_peak(y_true, y_pred):
    return K.max(K.max(y_pred, axis=1))

def sim_mat(y_true, y_pred):
    yt2 = K.sum(K.square(y_true), axis=1, keepdims=True)
    yp2 = K.sum(K.square(y_pred), axis=1, keepdims=True)
    y_true = y_true / K.sqrt(yt2)
    y_pred = y_pred / K.sqrt(yp2)
    return tf.matmul(y_true, tf.transpose(y_pred))

def sim_prec(y_true, y_pred):
    sim = sim_mat(y_true, y_pred)
    uni, idx = tf.unique(tf.argmax(sim, axis=1, output_type=tf.int32))
    ok = tf.cast(tf.size(uni), tf.float32)
    m = tf.cast(tf.shape(y_pred)[0], tf.float32)
    return ok / m

def sim_recall(y_true, y_pred):
    sim = sim_mat(y_true, y_pred)
    uni, idx = tf.unique(tf.argmax(sim, axis=0, output_type=tf.int32))
    ok = tf.cast(tf.size(uni), tf.float32)
    n = tf.cast(tf.shape(y_true)[0], tf.float32)
    return ok / n

def sim_f_measure(y_true, y_pred):
    prec = sim_prec(y_true, y_pred)
    recall = sim_recall(y_true, y_pred)
    return 2.0 * prec * recall / (prec + recall)

def min_sim(y_true, y_pred):
    sim = sim_mat(y_true, y_pred)
    return K.min(K.max(sim, axis=1))
