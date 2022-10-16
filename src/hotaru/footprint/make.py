import numpy as np
import tensorflow as tf

from ..filter.laplace import gaussian_laplace_multi
from ..filter.util import erosion
from ..util.distribute import ReduceOp
from ..util.distribute import distributed
from ..util.progress import Progress
from .segment import get_segment_mask


@distributed(ReduceOp.LIST, ReduceOp.CONCAT)
@tf.function(
    input_signature=[
        (
            tf.TensorSpec([None], tf.int64),
            tf.TensorSpec([None, None, None], tf.float32),
        ),
        tf.TensorSpec([None], tf.int64),
        tf.TensorSpec([None], tf.int64),
        tf.TensorSpec([None], tf.int64),
        tf.TensorSpec([None], tf.float32),
    ]
)
def _filter(args, ts, ys, xs, rs):
    def _zero():
        return tf.zeros_like(imgs[:0])

    def _nonzero():
        tl = tf.gather(ts, ids) - idx[0]
        yl = tf.gather(ys, ids)
        xl = tf.gather(xs, ids)
        rl = tf.gather(rs, ids)
        tlist, tl = tf.unique(tl)
        rlist, rl = tf.unique(rl)
        select = tf.gather(imgs, tlist)
        logs = gaussian_laplace_multi(select, rlist)
        logs = tf.gather_nd(logs, tf.stack([tl, rl], axis=1))
        return logs

    idx, imgs = args
    cond = (idx[0] <= ts) & (ts <= idx[-1])
    ids = tf.cast(tf.where(cond)[:, 0], tf.int32)
    logs = tf.cond(tf.size(ids) > 0, _nonzero, _zero)
    return logs, ids


@distributed(ReduceOp.LIST)
@tf.function(
    input_signature=[
        (
            tf.TensorSpec([None, None], tf.float32),
            tf.TensorSpec([], tf.int64),
            tf.TensorSpec([], tf.int64),
        ),
        tf.TensorSpec([None, None], tf.bool),
    ]
)
def _segment(args, mask):
    log, y, x = args
    seg = get_segment_mask(log, y, x, mask)
    seg = tf.ensure_shape(seg, mask.shape)
    dmax = tf.math.reduce_max(tf.boolean_mask(log, seg))
    dmin = tf.math.reduce_max(tf.boolean_mask(log, ~erosion(seg) & seg))
    log = tf.where(seg, log, 0)
    dat = tf.boolean_mask(log, mask)
    dat = tf.where(dat > dmin, (dat - dmin) / (dmax - dmin), 0)
    return dat[None, :]


def make_segment(imgs, mask, info, batch):
    """"""

    nt, h, w = imgs.shape
    nk = info.shape[0]
    nx = np.count_nonzero(mask)

    mask = tf.convert_to_tensor(mask, tf.bool)
    ts = tf.convert_to_tensor(info.t.to_numpy(), tf.int64)
    ys = tf.convert_to_tensor(info.y.to_numpy(), tf.int64)
    xs = tf.convert_to_tensor(info.x.to_numpy(), tf.int64)
    rs = tf.convert_to_tensor(info.radius.to_numpy(), tf.float32)

    imgs = Progress(imgs.enumerate(), "filter", nt, unit="frame", batch=batch)
    logs_unsort, ids = _filter(imgs, ts, ys, xs, rs)

    ids = ids.numpy()
    logs = [None] * ids.size
    for i, j in enumerate(ids):
        logs[j] = logs_unsort[i]

    logs_ds = tf.data.Dataset.from_generator(
        lambda: zip(logs, ys, xs),
        output_signature=(
            tf.TensorSpec([h, w], tf.float32),
            tf.TensorSpec([], tf.int64),
            tf.TensorSpec([], tf.int64),
        ),
    )
    logs_ds = Progress(logs_ds, "segment", nk, unit="cell")
    data = _segment(logs_ds, mask)
    return data
