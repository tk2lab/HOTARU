# -*- coding: utf-8 -*-

import tensorflow as tf
import pandas as pd
import numpy as np
import tifffile

from ..util.sparse import SparseBase
from ..util.filter import gaussian, gaussian_laplace_multi
from ..util.segment import get_segment_index, get_segment_index_tf


class Footprint(SparseBase):

    def build(self, base=None):
        super().build(base.imgs.h, base.imgs.w)
        self.base = base
        self.add_variable('thr_blur', (), init=0.0)
        self.add_variable('thr_asim', (), init=0.0)
        self.add_variable('ids', (None,), tf.int32)
        self.add_variable('ps', (None, 2), tf.int32)
        self.add_variable('rs', (None,))
        self.add_variable('gs', (None,))

    def save(self, filename, ids=None):
        data = tf.sparse.to_dense(self.to_sparse())
        if ids is not None:
            idx = self.base.uid_to_aid(self.ids, ids)
        else:
            idx = tf.range(tf.shape(data)[0], dtype=tf.int32)
        tifffile.imsave(filename+'a.tif', tf.gather(data, idx).numpy())
        pd.DataFrame(index=tf.gather(self.ids, idx).numpy(), data=dict(
            y=tf.gather(self.ps[:,0], idx).numpy(),
            x=tf.gather(self.ps[:,1], idx).numpy(),
            radius=tf.gather(self.rs, idx).numpy(),
            shape=tf.gather(self.gs, idx).numpy(),
        )).to_csv(filename+'a.csv')

    def segment(self):
        self.clear()
        imgs = self.base.imgs.to_dataset()
        pos = tf.cast(tf.where(self.base.imgs.mask), tf.int32)
        shape = tf.shape(self.base.imgs.mask)
        e = tf.constant(0)
        for img in imgs:
            s, e = e, e + tf.shape(img)[0]
            img = tf.stack([tf.scatter_nd(pos, x, shape) for x in img])
            self._append_segment(img, s, e)
            tf.print('*', end='')
        tf.print()
        tf.print(tf.size(self.gs))

    def clean(self, batch=100):
        pos = tf.where(self.base.imgs.mask)
        shape = self.base.imgs.h, self.base.imgs.w
        imgs = tf.data.Dataset.from_tensor_slices(self.base.spatial.aval)
        imgs = imgs.map(lambda x: tf.scatter_nd(pos, x, shape) / tf.reduce_max(x))
        imgs = imgs.batch(batch)
        e = tf.constant(0)
        self.clear()
        self.ids.assign(self.base.spatial.ids)
        for img in imgs:
            s, e = e, e + tf.shape(img)[0]
            self._append_clean(img)
            tf.print('*', end='')
        tf.print()
        tf.print(e)

    def remove_blur(self):
        sid = tf.argsort(self.gs)
        count = tf.constant(0)
        for i, k in enumerate(sid):
            print(f'[{self.ids[k]:>4d}] {self.gs[k]:.3f}', end='')
            if self.gs[k] < self.thr_blur:
                print('... remove')
            else:
                print()
                count += 1
                if count >= 10:
                    break
        cond = tf.gather(self.gs, sid) >= self.thr_blur
        sid = tf.boolean_mask(sid, cond)[::-1]
        self.select(sid)
        tf.print(tf.size(sid))

    def remove_sim(self):
        _usim = self.calc_usim()
        _asim = self.calc_asim()

        cond = _asim > 0.1 #self.thr_asim
        _pair = tf.cast(tf.where(cond), tf.int32)
        _pair = tf.boolean_mask(_pair, _pair[:,0] < _pair[:,1])
        sid = tf.argsort(tf.gather_nd(_asim, _pair))[::-1]

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
            if (asim[p] <= self.thr_asim):
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


    #@tf.function(input_signature=[])
    def clear(self):
        super().clear()
        self.ids.assign(tf.zeros([0], tf.int32))
        self.ps.assign(tf.zeros([0,2], tf.int32))
        self.rs.assign(tf.zeros([0]))
        self.gs.assign(tf.zeros([0]))

    def select(self, sid):
        self.ids.assign(tf.gather(self.ids, sid))
        self.ps.assign(tf.gather(self.ps, sid))
        self.rs.assign(tf.gather(self.rs, sid))
        self.gs.assign(tf.gather(self.gs, sid))

        pos = tf.RaggedTensor.from_row_lengths(self.pos, self.size)
        val = tf.RaggedTensor.from_row_lengths(self.val, self.size)
        self.size.assign(tf.gather(self.size, sid))
        self.pos.assign(tf.gather(pos, sid).flat_values)
        self.val.assign(tf.gather(val, sid).flat_values)


    def calc_asim(self):
        sa = self.base.footprint.to_sparse()
        sa = tf.sparse.reshape(sa, [sa.dense_shape[0], -1])
        a = tf.sparse.to_dense(sa)
        asim = tf.matmul(a, a, False, True)
        sig_a = tf.sqrt(tf.linalg.tensor_diag_part(asim))
        asim /= sig_a * sig_a[:,tf.newaxis]
        return asim

    def calc_adup(self):
        sa = self.to_sparse()
        a = tf.cast(tf.sparse.to_dense(sa) > 0.01, tf.float32)
        anz = tf.reduce_sum(a, axis=(1,2))
        asim = tf.tensordot(a, a, ((1,2), (1,2)))
        asim /= anz + anz[:,tf.newaxis]
        return asim

    def calc_vsim(self):
        return self.change_cov(self.base.temporal.calc_vsim())

    def calc_usim(self):
        return self.change_cov(self.base.temporal.calc_usim())

    def change_cov(self, cov):
        aid = self.base.uid_to_aid(self.base.temporal.ids, self.ids)
        cov = tf.gather(cov, aid)
        cov = tf.transpose(cov)
        cov = tf.gather(cov, aid)
        return cov

    #@tf.function(input_signature=[
    #    tf.TensorSpec((None,None,None), tf.float32),
    #    tf.TensorSpec((), tf.int32), tf.TensorSpec((), tf.int32),
    #])
    def _append_segment(self, imgs, s, e):
        ps, rs = self.base.peak.ps, self.base.peak.rs
        cond = (ps[:,0] >= s) & (ps[:,0] < e)
        ids = tf.cast(tf.where(cond)[:,0], tf.int32)
        if tf.size(ids) > 0:
            ps = tf.gather(ps, ids)
            rs = tf.gather(rs, ids)
            ts, ys, xs = ps[:,0] - s, ps[:,1], ps[:,2]
            self.ids.assign(tf.concat([self.ids, ids], axis=0))

            gauss = self.base.peak.gauss
            if gauss > 0.0:
                imgs = gaussian(imgs, gauss)

            radius, rs = tf.unique(rs)
            gls = gaussian_laplace_multi(imgs, radius) 
            mask = tf.pad(self.base.imgs.mask, [[0,1],[0,1]])

            nk = tf.shape(ts)[0]
            for k in tf.range(nk):
                tf.autograph.experimental.set_loop_options(
                    parallel_iterations=1,
                )
                t, r, y, x = ts[k], rs[k], ys[k], xs[k]
                gl = gls[t, :, :, r]
                po = get_segment_index(gl, y, x, mask)
                gv = tf.gather_nd(gl, po)
                gv -= tf.reduce_min(gv)
                mgv = tf.reduce_max(gv)
                gv /= mgv

                self.ps.assign(tf.concat([self.ps, [[y,x]]], axis=0))
                self.rs.assign(tf.concat([self.rs, [radius[r]]], axis=0))
                self.gs.assign(tf.concat([self.gs, [mgv]], axis=0))
                self.size.assign(tf.concat([self.size, [tf.size(gv)]], axis=0))
                self.pos.assign(tf.concat([self.pos, po], axis=0))
                self.val.assign(tf.concat([self.val, gv], axis=0))

    #@tf.function(input_signature=[
    #    tf.TensorSpec((None,None,None), tf.float32),
    #])
    def _append_clean(self, imgs):
        gauss = self.base.peak.gauss
        radius = self.base.peak.radius
        mask = self.base.imgs.mask

        if gauss > 0.0:
            imgs = gaussian(imgs, gauss)
        gls = gaussian_laplace_multi(imgs, radius)
        max_gls = tf.reduce_max(gls, axis=(1,2,3), keepdims=True)
        p = tf.cast(tf.where(tf.equal(gls, max_gls) & mask[...,tf.newaxis]), tf.int32)
        mask = tf.pad(mask, [[0,1],[0,1]])

        shape = tf.shape(imgs)
        nk, h, w = shape[0], shape[1], shape[2]
        for k in tf.range(nk):
            tf.autograph.experimental.set_loop_options(
                parallel_iterations=1,
            )
            t, y, x, r = p[k,0], p[k,1], p[k,2], p[k,3]
            gl = gls[t,:,:,r]
            po = get_segment_index(gl, y, x, mask)
            sc = tf.reduce_max(tf.gather_nd(gl, po))
            va = tf.gather_nd(imgs[t], po)
            va -= tf.reduce_min(va)
            mva = tf.reduce_max(va)
            va /= mva

            self.ps.assign(tf.concat([self.ps, [[y,x]]], axis=0))
            self.rs.assign(tf.concat([self.rs, [radius[r]]], axis=0))
            self.gs.assign(tf.concat([self.gs, [sc]], axis=0))
            self.size.assign(tf.concat([self.size, [tf.size(va)]], axis=0))
            self.pos.assign(tf.concat([self.pos, po], axis=0))
            self.val.assign(tf.concat([self.val, va], axis=0))
