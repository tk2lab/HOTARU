from logging import getLogger

import jax
import jax.numpy as jnp
import tensorflow as tf

logger = getLogger(__name__)


def mapped_imgs(dataset, nt, calc, types, init, batch, pad_value=jnp.nan, in_axes=0, args=()):
    @jax.jit
    def aggrigate(val, out):
        for i, k in enumerate(types):
            match k:
                case "min":
                    vali = val[i].min(axis=0)
                    out[i] = jnp.minimum(out[i], vali)
                case "max":
                    vali = val[i].max(axis=0)
                    out[i] = jnp.maximum(out[i], vali)
                case "add":
                    vali = val[i].sum(axis=0)
                    out[i] += vali
                case ("stack", index):
                    s0, s1, *shape = val[i].shape
                    vali = val[i].reshape([s0 * s1] + shape)
                    out[i] = out[i].at[val[index].ravel()].set(vali)
                case ("argmax", index):
                    idx = jnp.argmax(val[-1], axis=0, keepdims=True)
                    val0 = jnp.max(val[k[1]], axis=0)
                    vali = jnp.take_along_axis(val[i], idx, axis=0)[0]
                    out[i] = jnp.where(out[index] > val0, out[i], vali)
                case "nouse":
                    pass
        return out

    value_map = {
        tf.float32: pad_value,
        tf.uint16: tf.constant(0, tf.uint16),
        tf.int16: tf.constant(0, tf.int16),
        tf.int32: -1,
        tf.int64: -1,
    }

    calc = jax.pmap(calc, in_axes=in_axes)

    spec = dataset.element_spec
    batch_size = batch[0] * batch[1]
    shapes = tuple((batch[1], *s.shape.as_list()) for s in spec)
    values = tuple(value_map[s.dtype] for s in spec)

    dataset = dataset.batch(batch[1])
    dataset = dataset.padded_batch(batch[0], shapes, values)
    dataset = dataset.prefetch(10)
    #logger.debug("batch %s", batch)
    #logger.debug("shapes %s", shapes)
    #logger.debug("values %s", values)

    end = 0
    out = init
    for clip in dataset.as_numpy_iterator():
        start, end = end, end + batch_size
        clip = [jnp.array(c) for c in clip]
        val = calc(*clip, *args)
        #logger.debug("val/out %s/%s", [v.shape for v in val], [o.shape for o in out])
        out = aggrigate(val, out)
        n = batch_size if end < nt else nt - start
        logger.info("%s: %s %d", "pbar", "update", n)
    return out
