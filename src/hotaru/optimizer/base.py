# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division

import tensorflow.compat.v2 as tf
import numpy as np


class ProxOptimizerBase(tf.Module):

    def __init__(self, loss_fn, xs, lr=1.0, scale=20.0, fac_lr=0.7):
        self.scale = tf.Variable(scale)
        self.fac_lr = tf.Variable(fac_lr)

        self.iterator = tf.Variable(0.0)
        self.lr = tf.Variable(lr)

        self.loss_fn = loss_fn
        self.xs = xs
        self.reset()

    def train(self, steps, tol=None,
              logdir=None, step_hook=None, check_cond=True, verbose=1):
        if logdir:
            writer = tf.summary.create_file_writer(logdir)
        else:
            writer = tf.summary.create_noop_writer()

        with writer.as_default():
            pb = tf.keras.utils.Progbar(
                steps, stateful_metrics=['loss','diff','retry','lr'],
                verbose=verbose,
            )
            old_loss = np.Inf
            total_retry = 0
            for step in range(steps):
                if check_cond:
                    loss, retry = self.minimize_with_check()
                else:
                    loss, retry = self.minimize()
                total_retry += retry

                tf.summary.scalar('loss', loss, step=step)
                if step_hook:
                    step_hook(step)

                diff = old_loss - loss
                old_loss = loss

                pb.add(1, [
                    ('loss', loss),
                    ('diff', diff),
                    ('retry', total_retry),
                    ('lr', self.lr.numpy()),
                ])

                if tol is not None and np.abs(diff) < tol:
                    pb.target = step
                    pb.update(step, [
                        ('loss', loss),
                        ('diff', diff),
                        ('retry', total_retry),
                        ('lr', self.lr.numpy()),
                    ])
                    break
        writer.close()
        return loss

    @tf.function
    def minimize(self):
        loss_y, gs_y = self._minimize()
        loss_new = self.loss_fn(*self.xs)
        penalty = self.calc_penalty(self.xs)
        self.iterator.assign(self.iterator + 1.0)
        return loss_new + penalty, 0

    @tf.function
    def minimize_with_check(self):
        loss_y, gs_y = self._minimize()
        loss_est, loss_new = self._losses(loss_y, gs_y)
        retry = tf.constant(0, tf.int32)
        while loss_new > loss_est:
            self._retry(gs_y)
            retry += 1
            loss_est, loss_new = self._losses(loss_y, gs_y)
        penalty = self.calc_penalty(self.xs)
        self.iterator.assign(self.iterator + 1.0)
        return loss_new + penalty, retry

    def _losses(self, loss_y, gs):
        loss_est = loss_y
        for x, y, g in zip(self.xs, self.ys, gs):
            d = x - y
            loss_est += tf.reduce_sum(d * g)
            loss_est += (0.5 / self.lr) * tf.reduce_sum(d * d)
        loss_new = self.loss_fn(*self.xs)
        return loss_est, loss_new

    @staticmethod
    def calc_penalty(xs):
        penalty = 0.0
        for x in xs:
            if hasattr(x, 'regularizer'):
                penalty += x.regularizer(x)
        return penalty

    @staticmethod
    def apply_mix(x, y, t):
        return (1.0 - t) * x + t * y

    @staticmethod
    def apply_prox(x, y, g, lr):
        ret = y - lr * g
        if hasattr(x, 'regularizer'):
            ret = x.regularizer.prox(ret, lr)
        return ret
