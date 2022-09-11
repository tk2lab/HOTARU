import numpy as np
import pandas as pd
import tensorflow as tf

from ..filter.laplace import gaussian_laplace_multi
from ..util.dataset import unmasked
from ..util.distribute import ReduceOp
from ..util.distribute import distributed
from .segment import get_segment_index
from .segment import remove_noise


def modify_footprint(footprint):
    i = np.arange(footprint.shape[0])
    j = np.argpartition(-footprint, 1)
    second = footprint[i, j[:, 1]]
    footprint[i, j[:, 0]] = second
    cond = second > 0.0
    return cond


def check_overwrap(segment):
    index = [-1]
    overwrap = [np.nan]
    cov = segment @ segment.T
    for i in range(1, segment.shape[0]):
        val = cov[i, :i] / segment[i].sum()
        j = np.argmax(val)
        index.append(j)
        overwrap.append(val[j])
    return index, overwrap


def check_nearest(peaks):
    ys = peaks.y.values
    xs = peaks.x.values
    index = [-1]
    distance = [np.nan]
    for i in range(1, xs.size):
        dist = np.square(ys[i] - ys[:i]) + np.square(xs[i] - xs[:i])
        j = np.argmin(dist)
        index.append(j)
        distance.append(np.sqrt(dist[j]))
    return index, distance


def check_accept(
    segment,
    peaks,
    radius_min,
    radius_max,
    thr_area,
    thr_overwrap,
):
    cond_min = peaks.radius.values == radius_min
    cond_max = peaks.radius.values == radius_max

    binary = (segment > thr_area).astype(np.float32)
    peaks["area"] = area = binary.sum(axis=1)

    mask = ~(cond_min | cond_max)
    select = peaks.loc[mask].index

    peaks["next"] = -1
    peaks["overwrap"] = np.nan
    index, overwrap = check_overwrap(binary[mask])
    peaks.loc[select[1:], "next"] = index[1:]
    peaks.loc[select[1:], "overwrap"] = overwrap[1:]
    cond_sim = peaks.overwrap.values > thr_overwrap

    peaks["accept"] = "yes"
    peaks.loc[cond_min | cond_sim, "accept"] = "no"
    peaks.loc[cond_max, "accept"] = "localx"

    peaks["reason"] = "-"
    peaks.loc[cond_sim, "reason"] = "overwrap"
    peaks.loc[cond_max, "reason"] = "large_r"
    peaks.loc[cond_min, "reason"] = "small_r"


def clean_footprint(footprint, index, mask, radius, batch, prog=None):
    @distributed(
        ReduceOp.CONCAT,
        ReduceOp.CONCAT,
        ReduceOp.CONCAT,
        ReduceOp.CONCAT,
        ReduceOp.CONCAT,
    )
    def _clean(imgs, mask, radius):
        logs = gaussian_laplace_multi(imgs, radius)
        nk, h, w = tf.shape(logs)[0], tf.shape(logs)[2], tf.shape(logs)[3]
        hw = h * w
        lsr = tf.reshape(logs, (nk, -1))
        pos = tf.cast(tf.argmax(lsr, axis=1), tf.int32)
        rs = pos // hw
        ys = (pos % hw) // w
        xs = (pos % hw) % w
        logs = tf.gather_nd(logs, tf.stack([tf.range(nk), rs], axis=1))

        nx = tf.size(rs)
        out = tf.TensorArray(tf.float32, size=nk)
        firmness = tf.TensorArray(tf.float32, size=nk)
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
            val = remove_noise(val)
            img = tf.scatter_nd(pos, val, [h, w])
            out = out.write(k, tf.boolean_mask(img, mask))
            firmness = firmness.write(
                k, (slogmax - slogmin) / (simgmax - simgmin)
            )
        return out.stack(), firmness.stack(), rs, ys, xs

    nk = footprint.shape[0]
    dataset = tf.data.Dataset.from_tensor_slices(footprint)
    dataset = unmasked(dataset, mask)
    dataset = dataset.batch(batch)

    mask = tf.convert_to_tensor(mask, tf.bool)
    radius = tf.convert_to_tensor(radius, tf.float32)

    footprint, f, r, y, x = _clean(dataset, mask, radius, prog=prog)
    r = tf.gather(radius, r)
    peaks = pd.DataFrame(
        dict(firmness=f.numpy(), radius=r.numpy(), x=x.numpy(), y=y.numpy()),
        index=index,
    )
    return footprint.numpy(), peaks
