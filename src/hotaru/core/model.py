# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np

from ..util.module import Module
from .common import Penalty
from .dynamics import DoubleExp
from .imgs import Imgs
from .peak import Peak
from .footprint import Footprint
from .ustep import TemporalComponent
from .astep import SpatialComponent
from ..util.timer import Timer, tictoc


class HotaruModel(Module):

    @staticmethod
    def load_or_build(job_dir, imgs_file=None, hz=None, mask_type=None,
                      tau1=None, tau2=None, ltau=None, batch=100):
        if tf.saved_model.contains_saved_model(job_dir):
            model = tf.saved_model.load(job_dir)
        else:
            model = HotaruModel('Hotaru')
            model._build()
        if imgs_file is not None:
            data_file = job_dir + '/assets/data.tfrecord'
            model.imgs.set(imgs_file, mask_type, data_file, batch)
        model.gamma.set(hz=hz, tau1=tau1, tau2=tau2, ltau=ltau)
        return model

    @staticmethod
    def build(job_dir, imgs_file, hz, mask_type, tau1, tau2, ltau, batch=100):
        model = HotaruModel('Hotaru')
        model._build()
        data_file = job_dir + '/assets/data.tfrecord'
        model.imgs.set(imgs_file, mask_type, data_file, batch)
        model.gamma.set(hz=hz, tau1=tau1, tau2=tau2, ltau=ltau)
        return model

    def set_gamma(self, **args):
        self.gamma.set(**args)

    def set_penalty(self, **args):
        self.penalty.set(**args)

    def save(self, filename):
        self.footprint.save(filename, self.temporal.ids)
        self.temporal.save(filename)

    def get_peak(self, **args):
        if tf.reduce_all([tf.equal(getattr(self.peak, k), v) for k, v in args.items()]):
            print()
            print('===== use cache for peak =====')
            print(f'{tf.size(self.peak.rs)}')
            return
        for k, v in args.items():
            getattr(self.peak, k).assign(v)
        with Timer('find peak'):
            self.peak.find()
        with Timer('reduce peak'):
            self.peak.reduce()
        self.history = []

    def get_segment(self, start=-1):
        if hasattr(self, 'history') and len(self.history) > max(0, start):
            if start == -1:
                start = len(self.history) - 1
            for k, v in self.history[start].items():
                getattr(self.footprint, k).assign(v)
            self.history = self.history[:start+1]
            print()
            print(f'===== use cache for segment: epoch={start} =====')
            print(f'{tf.size(self.footprint.size)}')
            return
        with Timer('get segment'):
            self.footprint.segment()
        keys = ['ids', 'ps', 'rs', 'gs', 'size', 'pos', 'val']
        self.history = [{
            k: tf.Variable(getattr(self.footprint, k)) for k in keys
        }]

    def ustep(self, **args):
        for k, v in args.items():
            getattr(self.temporal, k).assign(v)
        with Timer('temporal reset'):
            self.temporal.reset()
        with Timer('temporal train'):
            self.temporal.train()
        #with Timer('remove weak'):
        #    self.temporal.remove_weak()
        with Timer('remove sim u'):
            self.temporal.remove_sim()

    def astep(self, **args):
        for k, v in args.items():
            getattr(self.spatial, k).assign(v)
        with Timer('spatial reset'):
            self.spatial.reset()
        with Timer('spatial train'):
            self.spatial.train()

    def clean(self, batch=100, **args):
        for k, v in args.items():
            getattr(self.footprint, k).assign(v)
        with Timer('clean'):
            self.footprint.clean(batch)
        with Timer('remove blur'):
            self.footprint.remove_blur()
        with Timer('remove sim a'):
            self.footprint.remove_sim()
        keys = ['ids', 'ps', 'rs', 'gs', 'size', 'pos', 'val']
        self.history += [{
            k: tf.Variable(getattr(self.footprint, k)) for k in keys
        }]

    
    def _build(self):
        self.add_module('imgs', Imgs('Imgs'))
        self.add_module('gamma', DoubleExp('Gamma'))
        self.add_module('penalty', Penalty('Penalty'))
        self.add_module('peak', Peak('Peak'))
        self.add_module('footprint', Footprint('Footprint'))
        self.add_module('temporal', TemporalComponent('Temporal'))
        self.add_module('spatial', SpatialComponent('Spatial'))
        self.imgs.build()
        self.gamma.build()
        self.penalty.build(self)
        self.peak.build(self)
        self.footprint.build(self)
        self.temporal.build(self)
        self.spatial.build(self)
        self.history = []

    @staticmethod
    def uid_to_aid(ida, ids):
        nk = tf.size(ida)
        mid = tf.reduce_max(ida) + 1
        rid = tf.scatter_nd(ida[:,tf.newaxis], tf.range(nk), [mid])
        return tf.gather(rid, ids)

    def _nn(self):
        return tf.cast(self.imgs.nt, tf.float32) * tf.cast(self.imgs.nx, tf.float32)

    def _nm(self):
        ret = self._nn()
        if self.penalty.bt > 0.0:
            ret += tf.cast(self.imgs.nt, tf.float32)
        if self.penalty.bx > 0.0:
            ret += tf.cast(self.imgs.nx, tf.float32)
        return ret
