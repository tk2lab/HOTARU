import tensorflow as tf
import numpy as np
import tifffile

from ..util.gs import ensure_local_file


def load_data(path, in_type=None):
    path = ensure_local_file(path)
    if in_type is None:
        in_type = path.split('.')[-1]
    if in_type == 'npy':
        return NpyData(path)
    elif in_type == 'tif' or in_type == 'tiff':
        return TifData(path)
    elif in_type[:3] == 'raw':
        return RawData(path)
    raise RuntimeError(f'{path} is not imgs file')


class Data:

    def __init__(self, path):
        self._path = path

    def shape(self):
        return self._load().shape
    
    def clipped_dataset(self, y0, y1, x0, x1):
        def gen_clipped_tensor():
            for x in self._load():
                clip = self._wrap(x)[y0:y1, x0:x1]
                yield tf.convert_to_tensor(clip, tf.float32)
        return tf.data.Dataset.from_generator(gen_clipped_tensor, tf.float32)

    def _load(self):
        NotImplemented

    def _wrap(self, x):
        return x


class NumpyData(Data):

    def _load(self):
        return np.load(self._filename, mmap_mode='r')


class TifData(Data):

    def _load(self):
        if not hasattr(self, '_imgs'):
            tif = tifffile.TiffFile(self._path)
            if tif.series[0].offset:
                self._imgs = tif.series[0].asarray(out='memmap')
            else:
                self._imgs = tif.series[0]
                self._wrap = lambda x: x.asarray()
        return self._imgs


class RawData(Data):

    def _load(self):
        with open(f'{self._path}.info', 'r') as fp:
            info = fp.readline().replace('\n', '')
        dtype, h, w, endian = info.split(',')
        h, w = int(h), int(w)
        dtype = np.dtype(dtype).newbyteorder('<' if endian == 'l' else '>')
        return np.memmap(self._path, dtype, 'r').reshape(-1, h, w)
