# -*- coding: utf-8 -*-

import tensorflow as tf
import pandas as pd
import numpy as np

from .common import ComponentCommon
from ..util.module import Module
from ..optimizer import MaxNormL1
from ..optimizer import FastProxOptimizer as Optimizer
#from ..optimizer import FeasibleProxOptimizer as Optimizer


class TemporalComponent(Module, ComponentCommon):
    
    def build(self, base):
        self.base = base
        self.add_variable('epochs', (), tf.int32, 100)
        self.add_variable('tol', (), init=1e-3)
        self.add_variable('thr_weak', (), init=0.0)
        self.add_variable('thr_usim', (), init=0.0)

        self.add_variable('ids', (None,), tf.int32)
        self.add_variable('uval', (None,None))
        self.add_variable('vobs', (None,None))

        self.add_variable('ydat', (None,None))
        self.add_variable('ycov', (None,None))
        self.add_variable('yout', (None,None))
        self.add_variable('pena', ())

        with self.name_scope:
            self._regu = MaxNormL1(self.base.penalty.lu)
            self.optimizer = Optimizer(self, [self.uval])
        self.steps = self.optimizer.steps
        self.scale = self.optimizer.scale
        self.fac_lr = self.optimizer.fac_lr

    def var(self, uval):
        return self._var(self.base.gamma.u_to_v(uval))

    def lipschitz(self):
        gsum2 = tf.square(tf.reduce_sum(self.base.gamma.uvkernel))
        return gsum2 * self._lipschitz()

    def _reset(self, imgs):
        aval = self.base.footprint.to_sparse()
        aval = tf.sparse.to_dense(aval)
        aval = tf.boolean_mask(aval, self.base.imgs.mask, axis=1)

        nk, nt = tf.shape(aval)[0], self.base.imgs.nt.read_value()
        adat = []
        for d in imgs:
            adat.append(tf.matmul(aval, d, False, True))
            tf.print('*', end='')
        self.ydat.assign(tf.concat(adat, axis=1))

        asum, cov, out, pena = self.base.penalty._common(aval)
        self.ycov.assign(cov)
        self.yout.assign(out)
        self.pena.assign(pena + self.base.penalty.la * tf.reduce_sum(asum))

        nu = nt + self.base.gamma.pad
        self.ids.assign(self.base.footprint.ids)
        self.uval.assign(tf.zeros((nk, nu)))
        self.vobs.assign(self.ydat / asum[:,tf.newaxis])

    def remove_sim(self):
        _usim = self.calc_usim()
        _asim = self.calc_asim()

        cond = (_usim > 0.1) & (_asim > 0.5 * self.base.footprint.thr_asim) #self.thr_asim
        _pair = tf.cast(tf.where(cond), tf.int32)
        _pair = tf.boolean_mask(_pair, _pair[:,0] < _pair[:,1])
        sid = tf.argsort(tf.gather_nd(_usim, _pair))[::-1]

        pair = tf.gather(_pair, sid)
        asim = tf.gather_nd(_asim, pair)
        usim = tf.gather_nd(_usim, pair)

        nk = tf.size(self.ids)
        remove = tf.zeros((nk,), tf.bool)
        nsim = tf.size(usim)
        count = tf.constant(0)
        for p in tf.range(nsim):
            pk = pair[p]
            i, j = pk[0], pk[1]
            tf.print(f'similar pair [{self.ids[i]:>4d},{self.ids[j]:>4d}]: {usim[p]:.3f}, {asim[p]:.3f}', end='')
            if (usim[p] <= self.thr_usim):
                tf.print()
                count += 1
                if count >= 10:
                    break
            elif remove[i] | remove[j]:
                tf.print(' ... skip')
            else:
                tf.print(f' ... remove {self.ids[j]}')
                remove = tf.tensor_scatter_nd_update(remove, [[j]], [tf.constant(True)])
        remove_id = tf.where(remove)

        nr = tf.size(remove_id)
        cond = tf.scatter_nd(remove_id, tf.ones((nr,), tf.bool), [tf.size(self.ids)])
        sid = tf.where(tf.logical_not(cond))[:,0]
        if not tf.math.reduce_all(cond):
            self.select(sid)
        tf.print(tf.size(sid))

    def save(self, filename):
        idname = [f'uid{i}' for i in self.ids.numpy()]
        utime = np.arange(-self.base.gamma.pad.numpy(), self.base.imgs.nt.numpy()) / self.base.gamma.hz.numpy()
        vtime = np.arange(self.base.imgs.nt.numpy()) / self.base.gamma.hz.numpy()
        uval, vobs = self.uval.numpy(), self.vobs.numpy()
        vval = self.base.gamma.u_to_v(self.uval).numpy()
        pd.DataFrame(uval.T, index=utime, columns=idname).to_csv(filename + 'u.csv')
        pd.DataFrame(vval.T, index=vtime, columns=idname).to_csv(filename + 'v.csv')
        pd.DataFrame(vobs.T, index=vtime, columns=idname).to_csv(filename + 'vobs.csv')


    def select(self, sid):
        self.ids.assign(tf.gather(self.ids, sid))
        self.uval.assign(tf.gather(self.uval, sid))
        self.vobs.assign(tf.gather(self.vobs, sid))
        self.ydat.assign(tf.gather(self.ydat, sid))
        self.ycov.assign(tf.gather(tf.transpose(tf.gather(self.ycov, sid)), sid))
        self.yout.assign(tf.gather(tf.transpose(tf.gather(self.yout, sid)), sid))

    def calc_usim(self):
        u = self.uval.read_value()
        u /= tf.sqrt(tf.reduce_sum(tf.square(u), axis=1, keepdims=True))
        return tf.matmul(u, u, False, True)

    def calc_vsim(self):
        v = self.base.gamma.u_to_v(self.uval)
        v -= tf.reduce_mean(v, axis=1, keepdims=True)
        v /= tf.sqrt(tf.reduce_sum(tf.square(v), axis=1, keepdims=True))
        return tf.matmul(v, v, False, True)

    def calc_adup(self):
        return self.change_cov(self.base.footprint.calc_adup())

    def calc_asim(self):
        return self.change_cov(self.base.footprint.calc_asim())

    def change_cov(self, cov):
        aid = self.base.uid_to_aid(self.base.footprint.ids, self.ids)
        cov = tf.gather(cov, aid)
        cov = tf.transpose(cov)
        cov = tf.gather(cov, aid)
        return cov
