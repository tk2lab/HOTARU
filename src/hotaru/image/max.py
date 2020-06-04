import tensorflow as tf

from ..util.distributed import distributed, ReduceOp


def calc_max(data):

    @distributed(ReduceOp.MAX)
    def _calc(imgs):
        return tf.reduce_max(imgs, axis=0),

    return _calc(data)[0]
