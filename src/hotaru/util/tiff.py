import tempfile

import tensorflow as tf
import tifffile


def load_tiff(path):
    with tf.io.gfile.GFile(path, 'rb') as fp:
        return tifffile.imload(fp)


def save_tiff(path, val):
    if path.startswith('gs://'):
        dirname = os.path.dirname(path[6:])
        dirname = os.path.join(tempfile.gettempdir(), 'hotaru-gs', dirname)
        tf.io.gfile.makedirs(dirname)
        basename = os.path.basename(path)
        localfile = os.path.join(dirname, f'{basename}.tif')
        tifffile.imwrite(localfile, val)
        tf.io.gfile.copy(localfile, path)
    else:
        tifffile.imwrite(path, val)
