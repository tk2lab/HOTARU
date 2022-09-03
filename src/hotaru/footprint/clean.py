import numpy as np
import pandas as pd
import tensorflow as tf

from ..evaluate.footprint import calc_sim_area
from ..filter.gaussian import gaussian
from ..filter.laplace import gaussian_laplace_multi
from ..util.dataset import unmasked
from ..util.distribute import ReduceOp
from ..util.distribute import distributed
from .segment import get_segment_index


def modify_footprint(footprint):
    i = np.arange(footprint.shape[0])
    j = np.argpartition(-footprint, 1)
    second = footprint[i, j[:, 1]]
    footprint[i, j[:, 0]] = second
    cond = second > 0.0
    return cond


def check_accept(footprint, peaks, radius, thr_abs, thr_rel, thr_sim):
    peaks["accept"] = "yes"
    x = peaks["radius"]
    cond1 = x == radius[0]
    cond2 = x == radius[-1]

    segment = (footprint > 0.5).astype(np.float32)
    area = np.sum(segment, axis=1)
    peaks["area"] = area
    cond3 = area >= thr_abs + thr_rel * np.pi * x**2

    sim = calc_sim_area(segment, ~(cond1 ^ cond2 ^ cond3))
    #sim = calc_sim_cos(segment)
    peaks["sim"] = sim
    cond4 = sim > thr_sim

    peaks.loc[cond4, "accept"] = "large_sim"
    peaks.loc[cond3, "accept"] = "large_area"
    peaks.loc[cond2, "accept"] = "large_r"
    peaks.loc[cond1, "accept"] = "small_r"


def clean_footprint(data, index, mask, gauss, radius, batch, prog=None):
    @distributed(
        ReduceOp.CONCAT,
        ReduceOp.CONCAT,
        ReduceOp.CONCAT,
        ReduceOp.CONCAT,
        ReduceOp.CONCAT,
    )
    def _clean(imgs, mask, gauss, radius):
        if gauss is not None:
            imgs = gaussian(imgs, gauss)
        logs = gaussian_laplace_multi(imgs, radius)
        nk, h, w = tf.shape(logs)[0], tf.shape(logs)[2], tf.shape(logs)[3]
        hw = h * w
        lsr = tf.reshape(logs, (nk, -1))
        pos = tf.cast(tf.argmax(lsr, axis=1), tf.int32)
        rs = pos // hw
        ys = (pos % hw) // w
        xs = (pos % hw) % w
        logs = tf.gather_nd(logs, tf.stack([tf.range(nk), rs], axis=1))

        out = tf.TensorArray(tf.float32, size=0, dynamic_size=True)
        firmness = tf.TensorArray(tf.float32, size=0, dynamic_size=True)
        nx = tf.size(rs)
        for k in tf.range(nx):
            img, log, y, x = imgs[k], logs[k], ys[k], xs[k]
            pos = get_segment_index(log, y, x, mask)
            slog = tf.gather_nd(log, pos)
            slogmin = tf.math.reduce_min(slog)
            slogmax = tf.math.reduce_max(slog)
            simg = tf.gather_nd(img, pos)
            simgmin = tf.math.reduce_min(simg)
            simgmax = tf.math.reduce_max(simg)
            val = (simg - simgmin) / (simgmax - simgmin)
            #hist = tf.histogram_fixed_width(val, [0.0, 1.0], 50)[1:]
            #thr = 0.02 + 0.02 * tf.cast(tf.math.argmax(hist), tf.float32)
            #val = tf.where(val >= thr, (val - thr) / (1 - thr), 0)
            img = tf.scatter_nd(pos, val, [h, w])
            out = out.write(k, tf.boolean_mask(img, mask))
            firmness = firmness.write(
                k, (slogmax - slogmin) / (simgmax - simgmin)
            )
        return out.stack(), firmness.stack(), rs, ys, xs

    nk = data.shape[0]
    dataset = tf.data.Dataset.from_tensor_slices(data)
    dataset = unmasked(dataset, mask)
    dataset = dataset.batch(batch)

    mask = tf.convert_to_tensor(mask, tf.bool)
    if gauss > 0.0:
        gauss = tf.convert_to_tensor(gauss, tf.float32)
    else:
        gauss = None
    radius = tf.convert_to_tensor(radius, tf.float32)

    footprint, f, r, y, x = _clean(dataset, mask, gauss, radius, prog=prog)
    r = tf.gather(radius, r)
    peaks = pd.DataFrame(
        dict(firmness=f.numpy(), radius=r.numpy(), x=x.numpy(), y=y.numpy()),
        index=index,
    )
    return footprint.numpy(), peaks
