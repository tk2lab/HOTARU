import tensorflow as tf

from ..image.gaussian import gaussian
from ..image.laplace import gaussian_laplace_multi
from ..data.dataset import unmasked
from ..util.distribute import distributed, ReduceOp
from .segment import get_segment_index
from .util import get_normalized_val, get_magnitude, ToDense


def clean_footprint(data, mask, gauss, radius, batch):

    def _gen():
        for d in data:
            yield tf.convert_to_tensor(d, tf.float32)

    @tf.function
    def _get_footprint(img, gl, y, x):
        pos = get_segment_index(gl, y, x, mask)
        val = get_normalized_val(img, pos)
        mag = get_magnitude(gl, pos)
        footprint = to_dense(pos, val)
        return footprint, mag

    strategy = tf.distribute.get_strategy()

    dataset = tf.data.Dataset.from_generator(_gen, tf.float32)
    if mask is None:
        mask = tf.ones_like(data[0], tf.bool)
    else:
        dataset = unmasked(dataset, mask)
    dataset = dataset.batch(batch)
    to_dense = ToDense(mask)

    gauss = tf.convert_to_tensor(gauss)
    radius = tf.convert_to_tensor(radius)

    prog = tf.keras.utils.Progbar(data.shape[0])
    fps = tf.TensorArray(tf.float32, 0, True)
    mgs = tf.TensorArray(tf.float32, 0, True)
    for imgs in strategy.experimental_distribute_dataset(dataset):
        local_imgs, local_gls, local_peaks = _prepare(imgs, mask, gauss, radius)
        for t, y, x, r in local_peaks:
            img = local_imgs[t]
            gl = local_gls[t, :, :, r]
            fp, mg = _get_footprint(img, gl, y, x)
            fps = fps.write(fps.size(), fp)
            mgs = fps.write(mgs.size(), mg)
            prog.add(1)
    return fps.stack(), mgs.stack()


@distributed(ReduceOp.CONCAT, ReduceOp.CONCAT, ReduceOp.CONCAT, loop=False)
def _prepare(imgs, mask, gauss, radius):
    if gauss > 0.0:
        imgs = gaussian(imgs, gauss)
    gls = gaussian_laplace_multi(imgs, radius) 
    max_gls = tf.reduce_max(gls, axis=(1, 2, 3), keepdims=True)
    pos_bin = tf.equal(gls, max_gls) & mask[..., None]
    pos = tf.cast(tf.where(pos_bin), tf.int32)
    return imgs, gls, pos
