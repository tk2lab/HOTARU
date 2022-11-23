from collections import namedtuple

import tensorflow as tf

from ..util.distribute import ReduceOp
from ..util.distribute import distributed
from ..util.progress import Progress
from .neighbor import neighbor


def calc_mean_std(data, batch, verbose=1):
    disable = verbose < 1
    nt, nx = data.shape

    data = Progress(
        data, "mean std", nt, unit="frame", batch=batch, disable=disable
    )
    avgt, sumx, sq = _mean_std(data)
    avgx = sumx / nt
    avg0 = tf.math.reduce_mean(avgt)
    avgx -= avg0
    var = sq / nt / nx - tf.math.reduce_mean(tf.math.square(avgx))
    std = tf.math.sqrt(var)

    Stats = namedtuple("Stats", ["avgx", "avgt", "std"])
    stats = (o.numpy() for o in (avgx, avgt, std))
    return Stats(*stats)


def calc_max(data, batch, verbose=1):
    disable = verbose < 1
    nt = data.shape[0]

    data = Progress(
        data, "max", nt, unit="frame", batch=batch, disable=disable
    )
    imax = _max(data)
    return imax.numpy()


def calc_std(data, batch, verbose=1):
    disable = verbose < 1
    nt = data.shape[0]

    data = Progress(
        data, "std", nt, unit="frame", batch=batch, disable=disable
    )
    s, n = _std(data)
    std = tf.math.sqrt(s / n)
    return std.numpy()


def calc_cor(imgs, batch, verbose=1):
    disable = verbose < 1
    nt = data.shape[0]

    data = Progress(
        data, "cor", nt, unit="frame", batch=batch, disable=disable
    )
    sx1, sy1, sx2, sxy, sy2, ntf = _cor(imgs)
    avg_x = sx1 / ntf
    avg_y = sy1 / ntf
    cov_xx = sx2 / ntf - tf.math.square(avg_x)
    cov_xy = sxy / ntf - avg_x * avg_y
    cov_yy = sy2 / ntf - tf.math.square(avg_y)
    cor = cov_xy / tf.math.sqrt(cov_xx * cov_yy)
    return cor.numpy()


@distributed(ReduceOp.CONCAT, ReduceOp.SUM, ReduceOp.SUM)
@tf.function(input_signature=[tf.TensorSpec([None, None], tf.float32)])
def _mean_std(data):
    avgt = tf.math.reduce_mean(data, axis=1)
    sumx = tf.math.reduce_sum(data, axis=0)
    sq = tf.math.reduce_sum(tf.math.square(data - avgt[:, None]))
    return avgt, sumx, sq


@distributed(ReduceOp.SUM)
@tf.function
def _max(img):
    img = tf.cast(img, tf.float32)
    return tf.reduce_sum(img, axis=0)


@distributed(ReduceOp.SUM, ReduceOp.SUM)
@tf.function
def _std(img):
    img = tf.cast(img, tf.float32)
    d = img - tf.reduce_mean(img, axis=0)
    s = tf.reduce_sum(d**2, axis=0)
    n = tf.cast(tf.shape(img)[0], tf.float32)
    return s, n


@distributed(
    ReduceOp.SUM,
    ReduceOp.SUM,
    ReduceOp.SUM,
    ReduceOp.SUM,
    ReduceOp.SUM,
    ReduceOp.SUM,
)
@tf.function
def _cor(img):
    img = tf.cast(img, tf.float32)
    nei = neighbor(img)
    sx1 = tf.math.reduce_sum(img, axis=0)
    sy1 = tf.math.reduce_sum(nei, axis=0)
    sx2 = tf.math.reduce_sum(tf.math.square(img), axis=0)
    sxy = tf.math.reduce_sum(img * nei, axis=0)
    sy2 = tf.math.reduce_sum(tf.math.square(nei), axis=0)
    ntf = tf.cast(tf.shape(img)[0], tf.float32)
    return sx1, sy1, sx2, sxy, sy2, ntf
