# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np

from .common import ComponentCommon
from ..util.module import Module
from ..optimizer import MaxNormL1
from ..optimizer import FastProxOptimizer as Optimizer
#from ..optimizer import FeasibleProxOptimizer as Optimizer


class SpatialComponent(Module, ComponentCommon):

    def build(self, base):
        self.base = base
        self.add_variable('epochs', (), tf.int32, 100)
        self.add_variable('tol', (), init=1e-3)

        self.add_variable('ids', (None,), tf.int32)
        self.add_variable('aval', (None,None))

        self.add_variable('ydat', (None,None))
        self.add_variable('ycov', (None,None))
        self.add_variable('yout', (None,None))
        self.add_variable('pena', ())

        with self.name_scope:
            self._regu = MaxNormL1(self.base.penalty.la)
            self.optimizer = Optimizer(self, [self.aval])
        self.steps = self.optimizer.steps
        self.scale = self.optimizer.scale
        self.fac_lr = self.optimizer.fac_lr

    def var(self, aval):
        return self._var(aval)

    def lipschitz(self):
        return self._lipschitz()

    def _reset(self, imgs):
        uval = self.base.temporal.uval.read_value()
        uval /= tf.reduce_max(uval, axis=1, keepdims=True)
        vval = self.base.gamma.u_to_v(uval)

        nk, nx = tf.shape(uval)[0], self.base.imgs.nx.read_value()
        vdat = tf.zeros((nk,nx))
        e = tf.constant(0)
        for d in imgs:
            s, e = e, e + tf.shape(d)[0]
            vdat += tf.matmul(vval[:,s:e], d)
            tf.print('*', end='')
        self.ydat.assign(vdat)

        _, cov, out, pena = self.base.penalty._common(vval)
        self.ycov.assign(cov)
        self.yout.assign(out)
        self.pena.assign(pena + self.base.penalty.lu * tf.reduce_sum(uval))

        self.ids.assign(self.base.temporal.ids)
        self.aval.assign(tf.zeros((nk, nx)))

    def select(self, sid):
        self.ids.assign(tf.gather(self.ids, sid))
        self.aval.assign(tf.gather(self.aval, sid))
        self.ydat.assign(tf.gather(self.ydat, sid))
        self.ycov.assign(tf.gather(tf.transpose(tf.gather(self.ycov, sid)), sid))
        self.yout.assign(tf.gather(tf.transpose(tf.gather(self.yout, sid)), sid))
