import tensorflow as tf
import numpy as np
import tifffile

from ..util.gs import ensure_local_file


def get_shape(filename):
    filename = ensure_local_file(filename)
    in_type = filename.split('.')[-1]
    if in_type == 'npy':
        return np.load(filename, mmap_mode='r').shape
    elif in_type == 'tif':
        return tifffile.TiffFile(filename).series[0].shape
    elif in_type[:3] == 'raw':
        _, dtype, h, w, endian = in_type.split('_')
        h, w = int(h), int(w)
        dtype = np.dtype(dtype).newbyteorder('<' if endian == 'l' else '>')
        return np.memmap(filename, dtype, 'r', shape=(-1, h, w)).shape
    raise RuntimeError(f'{filename} is not imgs file')


def load_data(filename):
    filename = ensure_local_file(filename)
    in_type = filename.split('.')[-1]
    if in_type == 'npy':
        return load_data_npy(filename)
    elif in_type == 'tif':
        return load_data_tif(filename)
    elif in_type[:3] == 'raw':
        return load_data_raw(filename)
    raise RuntimeError(f'{filename} is not imgs file')


def load_data_npy(filename):
    mmap = np.load(filename, mmap_mode='r')
    return mmap, lambda x: x


def load_data_tif(filename):
    tif = tifffile.TiffFile(filename)
    if tif.series[0].offset:
        mmap = tif.series[0].asarray(out='memmap')
        return mmap, lambda x: x
    else:
        return tif.series[0], lambda x: x.asarray()


def load_data_raw(filename):
    in_type = infile.split('.')[-1]
    _, dtype, h, w, endian = in_type.split('_')
    h, w = int(h), int(w)
    dtype = np.dtype(dtype).newbyteorder('<' if endian == 'l' else '>')
    mmap = np.memmap(filename, dtype, 'r', shape=(-1, h, w))
    return mmap, lambda x: x
