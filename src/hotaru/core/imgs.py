# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
import os

from ..util.module import Module
from ..util.imgs import load_imgs, get_mask
from ..eval.image import calc_stats, calc_max, calc_cor
from ..util.timer import Timer


class Imgs(Module):

    def to_dataset(self):
        def parse(ex):
            return tf.io.parse_tensor(ex, tf.float32)
        data = tf.data.TFRecordDataset(self.data_file)
        data = data.map(parse)
        return data

    def build(self):
        # original data properies
        self.add_variable('imgs_file', (), tf.string)
        self.add_variable('mask_type', (), tf.string)
        self.add_variable('origin', (2,), tf.int32)
        # statistics of clipped data
        self.add_variable('avgt', (None,))
        self.add_variable('avgx', (None, None))
        self.add_variable('std', ())
        # for output dataset
        self.add_variable('nt', (), tf.int32)
        self.add_variable('mask', (None, None), tf.bool)
        self.add_variable('nx', (), tf.int32)
        self.add_variable('h', (), tf.int32)
        self.add_variable('w', (), tf.int32)
        self.add_variable('ft', (None,))
        self.add_variable('fx', (None,))
        self.add_variable('f0', ())

    def set(self, imgs_file, mask_type, data_file, batch=100):

        def gen():
            b = batch
            for s in range(0, nt, b):
                e = min(s + b, nt)
                yield tf.convert_to_tensor(imgs[s:e], tf.float32)

        if hasattr(self, 'imgs_file'):
            if self.imgs_file.numpy().decode() == imgs_file:
                if self.mask_type.numpy().decode() == mask_type:
                    print()
                    print('===== use cahce data =====')
                    print(f'{self.nt.read_value()} {self.h.read_value()} {self.w.read_value()} {self.nx.read_value()}')
                    return

        with Timer('load'):
            imgs = load_imgs(imgs_file)
            nt, h, w = imgs.shape
            print(nt, h, w)

            mask = get_mask(mask_type, h, w)
            my = np.where(np.any(mask, axis=1))[0]
            mx = np.where(np.any(mask, axis=0))[0]
            y0, y1, x0, x1 = my[0], my[-1] + 1, mx[0], mx[-1] + 1
            imgs = imgs[:,y0:y1, x0:x1]
            mask = mask[y0:y1, x0:x1]
            (h, w), nx = mask.shape, np.count_nonzero(mask)
            print(nt, h, w, nx)

        self.imgs_file.assign(imgs_file)
        self.mask_type.assign(mask_type)
        self.origin.assign((y0, x0))
        self.nt.assign(nt)
        self.h.assign(h)
        self.w.assign(w)

        with Timer('stat'):
            data = tf.data.Dataset.from_generator(gen, tf.float32)
            data = tf.data.experimental.to_variant(data)
            mask = tf.convert_to_tensor(mask, tf.bool)
            avgt, avgx, std = calc_stats(data, nt, mask)

        self.avgt.assign(avgt)
        self.avgx.assign(avgx)
        self.std.assign(std)

        tf.io.gfile.makedirs(os.path.dirname(data_file))
        with Timer('tfrecord'), tf.io.TFRecordWriter(data_file) as writer:
            data = tf.data.Dataset.from_generator(gen, tf.float32)
            e = tf.constant(0)
            ft = []
            fx = tf.zeros((nx,))
            for img in data:
                s, e = e, e + tf.shape(img)[0]
                img = tf.convert_to_tensor(img, tf.float32)
                img -= avgt[s:e,tf.newaxis,tf.newaxis]
                img -= avgx
                img /= std
                #img = tf.where(mask, img, tf.zeros((h, w)))
                img = tf.boolean_mask(img, mask, axis=1)
                ft.append(tf.reduce_mean(img, axis=1))
                fx += tf.reduce_sum(img, axis=0)
                writer.write(tf.io.serialize_tensor(img).numpy())
                tf.print('*', end='')
            tf.print()

        self.data_file = tf.saved_model.Asset(data_file)
        self.mask.assign(mask)
        self.nx.assign(nx)
        self.ft.assign(tf.concat(ft, axis=0))
        self.fx.assign(fx / tf.cast(nt, tf.float32))
        self.f0.assign(tf.reduce_mean(self.fx))

    def calc_max(self):
        imgs = self.to_dataset()
        imgs = tf.data.experimental.to_variant(imgs)
        h, w = self.h.numpy(), self.w.numpy()
        self.add_variable('max', (h, w), init=calc_max(imgs))

    def calc_cor(self):
        imgs = self.to_dataset()
        imgs = tf.data.experimental.to_variant(imgs)
        h, w = self.h.numpy(), self.w.numpy()
        self.add_variable('cor', (h, w), init=calc_cor(imgs))
