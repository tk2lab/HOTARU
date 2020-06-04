# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division

import contextlib
import time


@contextlib.contextmanager
def Timer(msg):
    print()
    print('===== {} start ====='.format(msg))
    start = time.time()
    yield
    print('{} end: {} sec'.format(msg, time.time() - start))


def tictoc(fn):

    @Timer(fn.__name__)
    def wrap(*args, **kwargs):
        return fn(*args, **kwargs)

    return wrap
