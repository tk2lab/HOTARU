import tensorflow.keras.backend as K

from ..util.distributed import distributed, ReduceOp


@distributed(ReduceOp.MAX)
def calc_max(imgs):
    return K.max(imgs, axis=0)
