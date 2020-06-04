 # -*- coding: utf-8 -*-

import tensorflow as tf
import tifffile
import numpy as np
import tempfile
import os


def load_imgs(filename):
    in_type = filename.split('.')[-1]
    if in_type[:3] == 'raw':
        return load_imgs_raw(filename)
    elif in_type == 'tif':
        return load_imgs_tif(filename)
    elif in_type == 'npy':
        return load_imgs_npy(filename)


def get_mask(masktype, h, w):
    if masktype[-4:] == '.tif':
        with GFile(masktype, 'rb') as fp:
            tif = tifffile.imread(fp)
        mask = tif > 0
    elif masktype[-4:] == '.npy':
        with GFile(masktype, 'rb') as fp:
            mask = np.load(fp)
    elif masktype[-4:] == '.pad':
        pad = int(masktype.split('.')[-2])
        mask = np.zeros([h, w], bool)
        mask[pad:h-pad,pad:w-pad] = True
    return mask


class GFile(tf.io.gfile.GFile):

    def readinto(self, buff):
        result = self.read(buff.data.nbytes)
        buff.data.cast('B')[:] = memoryview(result)
        return len(result)


def load_imgs_npy(filename):
    if filename.startswith('gs://'):
        origfile = filename
        d = tempfile.TemporaryDirectory()
        filename = d.name + '/' + os.path.basename(origfile)
        tf.io.gfile.copy(origfile, filename)
    return np.load(filename, mmap_mode='r')


def load_imgs_raw(filename):
    if filename.startswith('gs://'):
        origfile = filename
        d = tempfile.TemporaryDirectory()
        filename = d.name + '/' + os.path.basename(origfile)
        tf.io.gfile.copy(origfile, filename)
    in_type = infile.split('.')[-1]
    _, dtype, h, w, endian = in_type.split('_')
    h, w = int(h), int(w)
    dtype = np.dtype(dtype).newbyteorder('<' if endian == 'l' else '>')
    return np.memmap(filename, dtype, 'r', shape=(-1, h, w))


def load_imgs_tif(filename):
    if filename.startswith('gs://'):
        origfile = filename
        d = tempfile.TemporaryDirectory()
        filename = d.name + '/' + os.path.basename(origfile)
        tf.io.gfile.copy(origfile, filename)
    #if filename.startswith('gs://') or tif.series[0].offset is None:
    tif = tifffile.TiffFile(filename)
    if tif.series[0].offset is None:
        #print('copy')
        return tif.series[0].asarray('memmap')
    #print('memmap')
    return tifffile.memmap(filename)
