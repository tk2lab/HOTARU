import tempfile

import tensorflow as tf
import tifffile


def load_tiff(base):
    with tf.io.gfile.GFile(f'{base}.tif', 'rb') as fp:
        return tifffile.imload(fp)


def save_tiff(base, val):
    if base.startswith('gs://'):
        dirname = os.path.dirname(base[6:])
        dirname = os.path.join(tempfile.gettempdir(), 'hotaru-gs', dirname)
        tf.io.gfile.makedirs(dirname)
        basename = os.path.basename(base)
        localfile = os.path.join(dirname, f'{basename}.tif')
        tifffile.imwrite(localfile, val)
        tf.io.gfile.copy(localfile, f'{base}.tif')
    else:
        tifffile.imwrite(f'{base}.tif', val)
