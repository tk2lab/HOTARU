import tensorflow as tf
import numpy as np
import tifffile

from ..util.gs import ensure_local_file


def load_data(filename):
    filename = ensure_local_file(filename)
    in_type = filename.split('.')[-1]
    if in_type == 'npy':
        (nt, h, w), gen = load_data_npy(filename)
    elif in_type == 'tif':
        (nt, h, w), gen = load_data_tif(filename)
    elif in_type[:3] == 'raw':
        (nt, h, w), gen = load_data_raw(filename)
    return gen, nt, h, w


def load_data_npy(filename):
    mmap = np.load(filename, mmap_mode='r')
    def gen():
        for m in mmap:
            yield m
    return mmap.shape, gen


def load_data_tif(filename):
    tif = tifffile.TiffFile(filename)
    if tif.series[0].offset:
        mmap = tif.series[0].asarray(out='memmap')
        def gen():
            for m in mmap:
                yield m
    else:
        def gen():
            for m in tif.series[0]:
                yield m.asarray()
    return tif.series[0].shape, gen


def load_data_raw(filename):
    in_type = infile.split('.')[-1]
    _, dtype, h, w, endian = in_type.split('_')
    h, w = int(h), int(w)
    dtype = np.dtype(dtype).newbyteorder('<' if endian == 'l' else '>')
    mmap = np.memmap(filename, dtype, 'r', shape=(-1, h, w))
    def gen():
        for m in mmap:
            yield m
    return mmap.shape, gen
