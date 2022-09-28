import numpy as np
import pandas as pd
import tensorflow as tf

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
    footprint /= np.where(no_seg, 1.0, second)[:, None]
    return no_seg


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
            #val = remove_noise(val)
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

    mask_t = tf.convert_to_tensor(mask, tf.bool)
    radius_t = tf.convert_to_tensor(radius, tf.float32)

    out = _clean(dataset, mask_t, radius_t, prog=prog)
    footprint, f, r, y, x = (o.numpy() for o in out)
    peaks = dict(x=x, y=y, radius=radius[r], firmness=f)
    peaks = pd.DataFrame(peaks, index=index)
    return footprint, peaks


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
