import tensorflow.keras.backend as K
import tensorflow as tf
from tqdm import trange

from ..util.distribute import distributed, ReduceOp


def calc_max(data, nt=None, verbose=1):

    @distributed(ReduceOp.SUM)
    def _calc(imgs):
        return K.sum(imgs, axis=0)

    with trange(nt, desc='Calc Stats', disable=verbose == 0) as prog:
        max_t = _calc(data, prog=prog)
    return max_t.numpy()
