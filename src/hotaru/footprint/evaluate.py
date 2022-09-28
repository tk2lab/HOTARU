import numpy as np
import pandas as pd
import tensorflow as tf
import scipy.ndimage as ndi

from ..filter.laplace import gaussian_laplace_multi
from ..util.dataset import unmasked
from ..util.distribute import ReduceOp
from ..util.distribute import distributed
from .segment import get_segment_index
#from .segment import remove_noise


def normalize_footprint(footprint):
    i = np.arange(footprint.shape[0])
    j = np.argpartition(-footprint, 1)
    second = footprint[i, j[:, 1]]
    footprint[i, j[:, 0]] = second
    no_seg = second == 0.0
    return no_seg


def evaluate_footprint(footprint, mask, radius, batch, prog=None):
    @distributed(
        ReduceOp.CONCAT,
        ReduceOp.CONCAT,
        ReduceOp.CONCAT,
        ReduceOp.CONCAT,
        ReduceOp.CONCAT,
    )
    def _evaluate(imgs, mask, radius):
        logs = gaussian_laplace_multi(imgs, radius)
        nk, h, w = tf.shape(logs)[0], tf.shape(logs)[2], tf.shape(logs)[3]
        hw = h * w
        lsr = tf.reshape(logs, (nk, -1))
        pos = tf.cast(tf.argmax(lsr, axis=1), tf.int32)
        rs = pos // hw
        ys = (pos % hw) // w
        xs = (pos % hw) % w
        indices = tf.stack([tf.range(nk), rs], axis=1)
        logs = tf.gather_nd(logs, indices)

        nx = tf.size(rs)
        indices = tf.stack([tf.range(nx), ys, xs], axis=1) 
        imgmin = tf.math.reduce_min(imgs, axis=(1, 2))
        imgmax = tf.math.reduce_max(imgs, axis=(1, 2))
        logpeak = tf.gather_nd(logs, indices)
        firmness = logpeak / (imgmax - imgmin)
        return firmness, rs, ys, xs

    nk = footprint.shape[0]
    dataset = tf.data.Dataset.from_tensor_slices(footprint)
    dataset = unmasked(dataset, mask)
    dataset = dataset.batch(batch)

    mask_t = tf.convert_to_tensor(mask, tf.bool)
    radius_t = tf.convert_to_tensor(radius, tf.float32)

    out = _evaluate(dataset, mask_t, radius_t, prog=prog)
    f, r, y, x = (o.numpy() for o in out)
    return dict(x=x, y=y, radius=radius[r], firmness=f)


def clean_footprint(xs, mask, threshold=0.01):
    normalize = lambda x: (x - threshold) / (xmax - threshold)
    tmp = np.zeros_like(mask, dtype=xs.dtype)
    for xi in xs:
        xmax = xi.max()
        tmp[mask] = xi
        y, x = np.where(tmp == xmax)
        obj, n = ndi.label(tmp > threshold * xmax)
        region = obj == obj[y[0], x[0]]
        tmp[...] = np.where(region, normalize(tmp), 0.0)
        xi[:] = tmp[mask]


def check_accept(segment, peaks, radius, sim, area, overwrap):
    peaks.insert(0, "segid", np.arange(segment.shape[0]))

    binary = (segment > area).astype(np.float32)
    peaks["area"] = area = binary.sum(axis=1)

    peaks.insert(1, "kind", "-")
    peaks["kind"] = "cell"
    peaks.loc[peaks.radius == radius[-1], "kind"] = "local"
    peaks.loc[peaks.radius == radius[0], "kind"] = "remove"

    check_overwrap(binary, peaks, overwrap)
    check_sim(segment, peaks, sim)

    peaks.sort_values("firmness", ascending=False, inplace=True)
    peaks.insert(2, "id", -1)
    cell = peaks.query("kind == 'cell'")
    peaks.loc[cell.index, "id"] = np.arange(cell.shape[0])
    local = peaks.query("kind == 'local'")
    peaks.loc[local.index, "id"] = np.arange(local.shape[0])
    return segment[cell.segid.to_numpy()], segment[local.segid.to_numpy()], peaks


def check_sim(segment, peaks, threshold):
    peaks.sort_values("firmness", ascending=False, inplace=True)
    select = peaks.query("kind == 'cell'")
    segment = segment[select.segid.to_numpy()]
    score = []
    index = []
    remove = []
    segment = segment / np.sqrt((segment**2).sum(axis=1, keepdims=True))
    cov = segment @ segment.T
    for i in range(1, segment.shape[0]):
        val = cov[i, :i]
        j = np.argmax(val)
        score.append(val[j])
        index.append(j)
        if score[-1] > threshold:
            remove.append(True)
            cov[:, i] = 0.0
        else:
            remove.append(False)
    selectx = select.iloc[1:]
    peaks.loc[selectx.index, "similarity"] = score
    peaks["sim_with"] = -1
    peaks.loc[selectx.index, "sim_with"] = select.segid.to_numpy()[index]
    peaks.loc[selectx.loc[remove].index, "kind"] = "remove"


def check_overwrap(binary, peaks, threshold):
    peaks.sort_values("area", ascending=False, inplace=True)
    select = peaks.query("kind == 'cell'")
    binary = binary[select.segid.to_numpy()]
    score = []
    index = []
    local = []
    area = binary.sum(axis=1)
    cov = (binary @ binary.T) / area
    for i in range(binary.shape[0] - 1):
        val = cov[i+1:, i]
        j = np.argmax(val)
        score.append(val[j])
        index.append(i + 1 + j)
        local.append(score[-1] > threshold)
    selectx = select.iloc[:-1]
    peaks.loc[selectx.index, "overwrap"] = score
    peaks["wrap_with"] = -1
    peaks.loc[selectx.index, "wrap_with"] = select.segid.to_numpy()[index]
    peaks.loc[selectx.loc[local].index, "kind"] = "remove"


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
