import tensorflow as tf

from ..image.gaussian import gaussian
from ..image.laplace import gaussian_laplace_multi
from .segment import get_segment_index
from .util import get_normalized_val, get_magnitude, ToDense


def make_footprint(dataset, mask, gauss, ts, rs, ys, xs, batch):

    @tf.function
    def _prepare(times, data, gauss, ts, rs, ys, xs):
        times = tf.cast(times, tf.int32)
        ts = tf.cast(ts, tf.int32)
        cond = (times[0] <= ts) & (ts <= times[-1])
        ts, tsi = tf.unique(tf.boolean_mask(ts, cond) - times[0])
        rs, rsi = tf.unique(tf.boolean_mask(rs, cond))
        ys = tf.boolean_mask(ys, cond)
        xs = tf.boolean_mask(xs, cond)

        imgs = tf.gather(data, ts)
        if gauss > 0.0:
            imgs = gaussian(imgs, gauss)
        gls = gaussian_laplace_multi(imgs, rs) 
        return gls, tf.stack((tsi, rsi, ys, xs), axis=1)


    def _finish_prepare(gls, peaks):
        gls = tf.concat(strategy.experimental_local_results(gls), axis=0)
        peaks = tf.concat(strategy.experimental_local_results(peaks), axis=0)
        return gls, peaks

    @tf.function
    def _get_footprint(gl, y, x):
        pos = get_segment_index(gl, y, x, mask)
        val = get_normalized_val(gl, pos)
        mag = get_magnitude(gl, pos)
        return to_dense(pos, val), mag

    strategy = tf.distribute.get_strategy()
    dataset = dataset.enumerate().batch(batch)
    to_dense = ToDense(mask)
    out = []
    prog = tf.keras.utils.Progbar(tf.size(ts).numpy())
    for times, data in strategy.experimental_distribute_dataset(dataset):
        local_gls, local_peaks = _finish_prepare(*strategy.run(
            _prepare, (times, data, gauss, ts, rs, ys, xs),
        ))
        for t, r, y, x in local_peaks:
            gl = local_gls[t, :, :, r]
            out.append(_get_footprint(gl, y, x))
            prog.add(1)
    footprints, mags = zip(*out)
    return tf.stack(footprints), tf.stack(mags)
