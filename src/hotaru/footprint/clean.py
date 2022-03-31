import tensorflow.keras.backend as K
import tensorflow as tf
import numpy as np
import pandas as pd
from tqdm import trange

from ..util.distribute import distributed, ReduceOp
from ..util.dataset import unmasked
from ..image.filter.gaussian import gaussian
from ..image.filter.laplace import gaussian_laplace_multi
from ..eval.footprint import calc_sim_cos
from ..eval.footprint import calc_sim_area
from .segment import get_segment_index_py


def modify_footprint(footprint):
    i = np.arange(footprint.shape[0])
    j = np.argpartition(-footprint, 1)
    second = footprint[i, j[:, 1]]
    footprint[i, j[:, 0]] = second
    cond = second > 0.0
    return cond


def check_accept(footprint, peaks, radius, thr_abs, thr_rel, thr_sim):
    peaks['accept'] = 'yes'
    x = peaks['radius']
    cond1 = x == radius[0]
    cond2 = x == radius[-1]

    segment = (footprint > 0.5).astype(np.float32)
    area = np.sum(segment, axis=1)
    peaks['area'] = area
    cond3 = (area >= thr_abs + thr_rel * np.pi * x ** 2)

    sim = calc_sim_area(segment, ~(cond1 ^ cond2 ^ cond3))
    peaks['sim'] = sim
    cond4 = sim >= thr_sim

    peaks.loc[cond4, 'accept'] = 'large_sim'
    peaks.loc[cond3, 'accept'] = 'large_area'
    peaks.loc[cond2, 'accept'] = 'large_r'
    peaks.loc[cond1, 'accept'] = 'small_r'


def clean_footprint(data, index, mask, radius, batch, verbose):
    dataset = tf.data.Dataset.from_tensor_slices(data)
    dataset = unmasked(dataset, mask)
    dataset = dataset.batch(batch)
    #to_dense = ToDense(mask)

    _mask = mask
    mask = K.constant(mask, tf.bool)
    radius_ = K.constant(radius)

    thr = 0.01
    nk = data.shape[0]
    i = 0
    ss = []
    ps = []
    fs = []
    #i = tf.constant(0)
    with trange(nk, desc='Clean', disable=verbose==0) as prog:
        for data in dataset:
            gl, ll, rl, yl, xl = _prepare(data, mask, radius_)
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
    r, y, x = np.array(ps).T
    r = radius[r]
    f = np.array(fs)
    peaks = pd.DataFrame(dict(firmness=f, radius=r, x=x, y=y), index=index)
    return np.array(ss)[:, mask], peaks


@distributed(*[ReduceOp.CONCAT for _ in range(5)], loop=False)
def _prepare(imgs, mask, radius):
    ls = gaussian_laplace_multi(imgs, radius)
    nk, h, w = tf.shape(ls)[0], tf.shape(ls)[2], tf.shape(ls)[3]
    hw = h * w
    lsr = K.reshape(ls, (nk, -1))
    pos = tf.cast(K.argmax(lsr, axis=1), tf.int32)
    rs = pos // hw
    ys = (pos % hw) // w
    xs = (pos % hw) % w
    ls = tf.gather_nd(ls, tf.stack([tf.range(nk), rs], axis=1))
    return imgs, ls, rs, ys, xs
