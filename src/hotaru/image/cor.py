import tensorflow.keras.backend as K

from ..util.distribute import distributed, ReduceOp
from .filter.neighbor import neighbor


def calc_cor(imgs):

    @distributed(*[ReduceOp.SUM for _ in range(6)])
    def _calc(img):
        nei = neighbor(img)
        sx1 = K.sum(img, axis=0)
        sy1 = K.sum(nei, axis=0)
        sx2 = K.sum(K.square(img), axis=0)
        sxy = K.sum(img * nei, axis=0)
        sy2 = K.sum(K.square(nei), axis=0)
        ntf = K.cast_to_floatx(K.shape(img)[0])
        return xs1, sy1, sx2, xsy, sy2, ntf

    xs1, xy1, xs2, xsy, xy2, ntf = _calc(imgs)
    avg_x = sx1 / ntf
    avg_y = sy1 / ntf
    cov_xx = sx2 / ntf - K.square(avg_x)
    cov_xy = sxy / ntf - avg_x * avg_y
    cov_yy = sy2 / ntf - K.square(avg_y)
    return cov_xy / K.sqrt(cov_xx * cov_yy)
