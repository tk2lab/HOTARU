import tensorflow as tf

from ..image.laplace import gaussian_laplace_multi
from ..data.dataset import unmasked
from .segment import get_segment_index
from .util import get_normalized_val, get_magnitude, ToDense


def clean_footprint(data, mask, gauss, radius, batch):

    def _gen():
        for d in data:
            yield d

    @tf.function
    def _get_footprint(img, gl, y, x):
        pos = get_segment_index(gl, y, x, mask)
        val = get_normalized_val(img, pos)
        mag = get_magnitude(gl, pos)
        footprint = to_dense(pos, val)
        return footprint, mag

    strategy = tf.distribute.get_strategy()
    nk = data.shape[0]
    dataset = tf.data.Dataset.from_generator(_gen, tf.float32)
    if mask is None:
        mask = tf.ones_like(data[0], tf.bool)
    else:
        dataset = unmasked(dataset, mask)
    dataset = dataset.batch(batch)
    radius = tf.convert_to_tensor(radius)
    to_dense = ToDense(mask)
    out = []
    prog = tf.keras.utils.Progbar(nk)
    for imgs in strategy.experimental_distribute_dataset(dataset):
        local_imgs, local_gls, local_peaks = _finish_prepare(strategy.run(
            _prepare, (imgs, mask, gauss, radius),
        ))
        for t, y, x, r in local_peaks:
            img = local_imgs[t]
            gl = local_gls[t, :, :, r]
            out.append(_get_footprint(img, gl, y, x))
            prog.add(1)
    footprints, mags = zip(*out)
    return tf.stack(footprints), tf.stack(mags)


@tf.function
def _prepare(imgs, mask, gauss, radius):
    if gauss > 0.0:
        imgs = gaussian(imgs, gauss)
    gls = gaussian_laplace_multi(imgs, radius) 
    max_gls = tf.reduce_max(gls, axis=(1, 2, 3), keepdims=True)
    pos_bin = tf.equal(gls, max_gls) & mask[..., None]
    pos = tf.cast(tf.where(pos_bin), tf.int32)
    return imgs, gls, pos


def _finish_prepare(args):
    imgs, gls, pos = args
    strategy = tf.distribute.get_strategy()
    imgs = tf.concat(strategy.experimental_local_results(imgs), axis=0)
    gls = tf.concat(strategy.experimental_local_results(gls), axis=0)
    pos = tf.concat(strategy.experimental_local_results(pos), axis=0)
    return imgs, gls, pos
