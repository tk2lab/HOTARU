import tensorflow as tf

from ..util.distribute import (
    ReduceOp,
    distributed,
)
from ..util.progress import Progress


def distributed_matmul(val, dat, trans=False, batch=None):
    """"""

    @distributed(ReduceOp.CONCAT)
    def _matmul_trans(dat, val):
        return tf.matmul(dat, val, False, True)

    @distributed(ReduceOp.SUM)
    def _matmul(tdat, val):
        t, dat = tdat
        val = tf.gather(val, t, axis=1)
        return tf.linalg.matmul(val, dat)

    nt = dat.shape[0]
    if trans:
        dat = Progress(
            dat,
            "mutmul",
            nt,
            unit="frame",
            batch=batch,
        )
        return tf.transpose(_matmul_trans(dat, val))
    else:
        dat = Progress(dat.enumerate(), "mutmul", nt, unit="frame", batch=batch)
        return _matmul(dat, val)
