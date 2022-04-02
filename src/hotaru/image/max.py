import tensorflow as tf
import click

from ..util.distribute import ReduceOp
from ..util.distribute import distributed


def calc_max(imgs, nt=None, verbose=1):

    @distributed(ReduceOp.SUM)
    def _calc(img):
        img = tf.cast(img, tf.float32)
        return tf.reduce_sum(img, axis=0),

    with click.progressbar(length=nt, label='Calc Max') as prog:
        imax, = _calc(imgs, prog=prog)
    return imax
