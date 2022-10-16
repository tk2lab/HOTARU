from collections import namedtuple

import tensorflow as tf

from ..util.distribute import ReduceOp
from ..util.distribute import distributed
from ..util.progress import Progress


def calc_stats(data, batch):
    """"""

    @distributed(
        ReduceOp.CONCAT,
        ReduceOp.SUM,
        ReduceOp.SUM,
    )
    def _calc(data):
        avgt = tf.math.reduce_mean(data, axis=1)
        sumx = tf.math.reduce_sum(data, axis=0)
        sq = tf.math.reduce_sum(tf.math.square(data - avgt[:, None]))
        return avgt, sumx, sq

    nt, nx = data.shape

    data = Progress(data, "calc stats", nt, unit="frame", batch=batch)
    avgt, sumx, sq = _calc(data)
    avgx = sumx / nt
    avg0 = tf.math.reduce_mean(avgt)
    avgx -= avg0
    var = sq / nt / nx - tf.math.reduce_mean(tf.math.square(avgx))
    std = tf.math.sqrt(var)

    Stats = namedtuple("Stats", ["avgx", "avgt", "std"])
    stats = (o.numpy() for o in (avgx, avgt, std))
    return Stats(*stats)
