import tensorflow.keras.backend as K
import tensorflow as tf
import numpy as np
from tqdm import trange

from ..util.distribute import distributed, ReduceOp
from ..util.dataset import unmasked
from ..image.filter.gaussian import gaussian
from ..image.filter.laplace import gaussian_laplace_multi
from .segment import get_segment_index_py


def clean_footprint(data, mask, gauss, radius, batch, verbose):
    dataset = tf.data.Dataset.from_tensor_slices(data)
    dataset = unmasked(dataset, mask)
    dataset = dataset.batch(batch)
    #to_dense = ToDense(mask)

    _mask = mask
    mask = K.constant(mask, tf.bool)
    gauss = K.constant(gauss)
    radius = K.constant(radius)

    thr = 0.01
    nk = data.shape[0]
    i = 0
    ss = []
    ps = []
    fs = []
    #i = tf.constant(0)
    with trange(nk, desc='Clean', disable=verbose==0) as prog:
        for data in dataset:
            gl, ll, rl, yl, xl = _prepare(data, mask, gauss, radius)
            _gl = gl.numpy()
            _ll = ll.numpy()
            _rl = rl.numpy()
            _yl = yl.numpy()
            _xl = xl.numpy()
            nx = _rl.size
            for k in range(nx):
                img, log, r, y, x = _gl[k], _ll[k], _rl[k], _yl[k], _xl[k]
                pos = get_segment_index_py(log, y, x, _mask)
                slog = log[pos[:, 0], pos[:, 1]]
                simg = img[pos[:, 0], pos[:, 1]]
                firmness = (slog.max() - slog.min()) / (simg.max() - simg.min())
                img = simg.min() * np.ones_like(img)
                img[pos[:, 0], pos[:, 1]] = simg
                img -= img.min()
                if img.max() > 0.0:
                    img /= img.max()
                ss.append(img)
                ps.append([r, y, x])
                fs.append(firmness)
                i += 1
                prog.update(1)
    return np.array(ss)[:, mask], np.array(ps), np.array(fs)


@distributed(*[ReduceOp.CONCAT for _ in range(5)], loop=False)
def _prepare(imgs, mask, gauss, radius):
    gs = gaussian(imgs, gauss) if gauss > 0.0 else imgs
    ls = gaussian_laplace_multi(gs, radius)
    nk, h, w = tf.shape(ls)[0], tf.shape(ls)[2], tf.shape(ls)[3]
    hw = h * w
    lsr = K.reshape(ls, (nk, -1))
    pos = tf.cast(K.argmax(lsr, axis=1), tf.int32)
    rs = pos // hw
    ys = (pos % hw) // w
    xs = (pos % hw) % w
    ls = tf.gather_nd(ls, tf.stack([tf.range(nk), rs], axis=1))
    return imgs, ls, rs, ys, xs
