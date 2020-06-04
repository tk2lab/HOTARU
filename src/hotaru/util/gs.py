import tempfile
import os
from urllib.parse import urlparse, urlunparse

import tensorflow as tf


def abspath(filename):
    scheme, netloc, path, params, query, fragment = urlparse(filename)
    if scheme == '':
        scheme = 'file'
        path = os.path.abspath(path)
    return urlunparse((scheme, netloc, path, params, query, fragment))


def ensure_local_file(filename):
    if not filename.startswith('gs://'):
        return filename
    dirname = os.path.dirname(filename[6:])
    dirname = os.path.join(tempfile.gettempdir(), 'hotaru-gs', dirname)
    tf.io.gfile.makedirs(dirname)
    basename = os.path.basename(filename)
    newfilename = os.path.join(dirname, basename)
    tf.io.gfile.copy(filename, newfilename)
    return newfilename
