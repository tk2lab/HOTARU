# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division

import tensorflow.compat.v2 as tf
import numpy as np

from ..util.module import Module
from ..util import Timer


class ComponentCommon(object):

    def lr0(self):
        return 1.0 / self.lipschitz()

    def loss(self, inputs):
        x = inputs[0]
        return 0.5 * tf.math.log(self.var(x)) + self.pena / self.base._nm()

    def penalty(self, inputs):
        x = inputs[0]
        return self._regu.loss(x) / self.base._nm()

    def prox(self, inputs, eta):
        x = inputs[0]
        return [self._regu.prox(x, eta / self.base._nm())]

    def reset(self):
        imgs = self.base.imgs.to_dataset()
        self._reset(imgs)
        self.optimizer.lr.assign(self.lr0())
        print()

    def train(self):
        score = None
        for epoch in tf.range(self.epochs):
            old_score, score = score, np.mean(self.optimizer.train())

            lr = self.optimizer.lr.read_value()
            x = self.optimizer.xs[0]
            nk = tf.shape(x)[0]
            var = self.var(x)
            pena = self._regu.loss(x) / self.base._nm()
            peak = tf.reduce_max(x, axis=1)
            pmin = tf.reduce_min(peak)
            pmax = tf.reduce_max(peak)
            if old_score is None:
                print(f'epoch {epoch:>3d}, {score: .6f}, -----------, {lr:.3e}', end='')
                print(f' {nk:>3d} {var:.3f} {pena:.5f} {pmin:.3f} {pmax:.3f}')
                self.remove_weak()
            else:
                diff = score - old_score
                print(f'epoch {epoch:>3d}, {score: .6f}, {diff: .8f}, {lr:.3e}', end='')
                print(f' {nk:>3d} {var:.3f} {pena:.5f} {pmin:.3f} {pmax:.3f}')
                if not self.remove_weak():
                    if np.abs(diff) < self.tol.numpy():
                        break

    def remove_weak(self):
        thr = self.base.temporal.thr_weak
        peak = tf.reduce_max(self.optimizer.xs[0], axis=1)
        sid = tf.argsort(peak)
        '''
        count = tf.constant(0)
        for i, k in enumerate(sid):
            print(f'[{self.ids[k]:>4d}] {peak[k]:.3f}', end='')
            if peak[k] < thr:
                print(' ... remove')
            else:
                print()
                count += 1
                if count >= 10:
                    break
        '''
        cond = tf.gather(peak, sid) >= thr
        ret = tf.reduce_all(cond)
        if ret:
            return False
        else:
            sid = tf.boolean_mask(sid, cond)[::-1]
            self.select(sid)
            #tf.print(tf.size(self.ids))
            return True

    def _var(self, xval):
        xcov = tf.matmul(xval, xval, False, True)
        xsum = tf.reduce_sum(xval, axis=1)
        xout = xsum[:, tf.newaxis] * xsum
        return (
            self.base._nn()
            + tf.reduce_sum(self.ycov * xcov)
            + tf.reduce_sum(self.yout * xout)
            - 2.0 * tf.reduce_sum(self.ydat * xval)
        ) / self.base._nm()

    def _lipschitz(self):
        return 0.5 * tf.reduce_max(tf.linalg.eigvalsh(self.ycov)) / self.base._nm()


class Penalty(Module):

    def build(self, base, la=0.0, lu=0.0, sa=100.0, ru=1.0, bx=0.0, bt=0.0):
        self.base = base
        self.add_variable('la', (), init=la)
        self.add_variable('lu', (), init=lu)
        self.add_variable('sa', (), init=sa)
        self.add_variable('ru', (), init=ru)
        self.add_variable('bx', (), init=bx)
        self.add_variable('bt', (), init=bt)

    def set(self, **args):
        for k, v in args.items():
            getattr(self, k).assign(v)

    def _common(self, val):
        nkf = tf.cast(tf.shape(val)[0], tf.float32)
        ntf = tf.cast(self.base.imgs.nt, tf.float32)
        nxf = tf.cast(self.base.imgs.nx, tf.float32)
        bx, bt = self.bx, self.bt

        xcov = tf.matmul(val, val, False, True)
        xsum = tf.reduce_sum(val, axis=1)
        xout = xsum * xsum[:,tf.newaxis]
        cx = (1.0 - tf.square(self.bt)) / nxf
        cy = (1.0 - tf.square(self.bx)) / ntf
        if tf.equal(tf.shape(val)[1], self.base.imgs.nt):
            cx, cy = cy, cx
        cov =   xcov - cx * xout
        out = - cy * xcov + xout / (ntf * nxf)
        return xsum, cov, out, self._pena(nkf)

    def _pena(self, nkf):

        def slin(x):
            if x > 0.0:
                return tf.math.log(tf.math.expm1(x)/x) - x
            else:
                return tf.constant(0.0)
        
        ntf = tf.cast(self.base.imgs.nt, tf.float32)
        nxf = tf.cast(self.base.imgs.nx, tf.float32)
        c = 0.5 * tf.math.log(2.0 * np.pi * np.e)
        pena = tf.constant(0.0)
        pena += ntf*nxf * c
        if self.bx > 0.0:
            pena += nxf * (c - 0.5 * tf.math.log(1.0 - self.bx))
        if self.bt > 0.0:
            pena += ntf * (c - 0.5 * tf.math.log(1.0 - self.bt))
        pena += nkf*nxf * (slin(self.la) - tf.math.log(self.sa/nxf))
        pena += nkf*ntf * (slin(self.lu) - tf.math.log(self.ru/self.base.gamma.hz))
        return pena
