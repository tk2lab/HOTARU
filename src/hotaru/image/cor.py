import tensorflow.keras.backend as K
import tensorflow as tf
import click

from ..util.distribute import distributed, ReduceOp
from .filter.neighbor import neighbor


def calc_cor(imgs, nt=None, verbose=1):

    @distributed(*[ReduceOp.SUM for _ in range(6)])
    def _calc(img):
        img = tf.cast(img, tf.float32)
        nei = neighbor(img)
        sx1 = K.sum(img, axis=0)
        sy1 = K.sum(nei, axis=0)
        sx2 = K.sum(K.square(img), axis=0)
        sxy = K.sum(img * nei, axis=0)
        sy2 = K.sum(K.square(nei), axis=0)
        ntf = K.cast_to_floatx(K.shape(img)[0])
        return sx1, sy1, sx2, sxy, sy2, ntf

    with click.progressbar(length=nt, label='Calc Cor') as prog:
        sx1, sy1, sx2, sxy, sy2, ntf = _calc(imgs, prog=prog)
    avg_x = sx1 / ntf
    avg_y = sy1 / ntf
    cov_xx = sx2 / ntf - K.square(avg_x)
    cov_xy = sxy / ntf - avg_x * avg_y
    cov_yy = sy2 / ntf - K.square(avg_y)
    return cov_xy / K.sqrt(cov_xx * cov_yy)
