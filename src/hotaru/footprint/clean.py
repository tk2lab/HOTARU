import tensorflow as tf

from ..data.dataset import unmasked
from ..image.gaussian import gaussian
from ..image.laplace import gaussian_laplace_multi
from ..util.distribute import distributed, ReduceOp
from .segment import get_segment_index
from .util import get_normalized_val, get_magnitude, ToDense


def clean_footprint(data, mask, gauss, radius, batch):
    strategy = tf.distribute.get_strategy()

    dataset = tf.data.Dataset.from_generator(_gen(data), tf.float32)
    mask = tf.convert_to_tensor(mask, tf.bool)
    dataset = unmasked(dataset, mask)
    dataset = dataset.batch(batch)
    to_dense = ToDense(mask)

    gauss = tf.convert_to_tensor(gauss)
    radius = tf.convert_to_tensor(radius)

    prog = tf.keras.utils.Progbar(data.shape[0])
    fps = tf.TensorArray(tf.float32, 0, True)
    mgs = tf.TensorArray(tf.float32, 0, True)
    for data in strategy.experimental_distribute_dataset(dataset):
        imgs, gls, peaks = _prepare(data, mask, gauss, radius)
        for t, y, x, r in peaks:
            gl = gls[t, :, :, r]
            pos = get_segment_index(gl, y, x, mask)
            mag = get_magnitude(gl, pos)
            img = imgs[t]
            val = get_normalized_val(img, pos)
            footprint = to_dense(pos, val)
            mgs = mgs.write(mgs.size(), mag)
            fps = fps.write(fps.size(), footprint)
            prog.add(1)
    return fps.stack(), mgs.stack()


def _gen(data):
    def func():
        for d in data:
            yield tf.convert_to_tensor(d, tf.float32)
    return func


@distributed(*[ReduceOp.CONCAT for _ in range(4)], loop=False)
def _prepare(imgs, mask, gauss, radius):
    if gauss > 0.0:
        imgs = gaussian(imgs, gauss)
    gls = gaussian_laplace_multi(imgs, radius) 
    max_gls = tf.reduce_max(gls, axis=(1, 2, 3), keepdims=True)
    pos_bin = tf.equal(gls, max_gls) & mask[..., None]
    pos = tf.cast(tf.where(pos_bin), tf.int32)
    return imgs, gls, pos
