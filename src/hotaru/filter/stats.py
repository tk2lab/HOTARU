from collections import namedtuple
from logging import getLogger

import jax
import jax.numpy as jnp
import numpy as np
import tensorflow as tf

from ..utils import (
    from_tf,
    get_gpu_env,
)
from .neighbor import neighbor

logger = getLogger(__name__)

Stats = namedtuple("Stats", "avgx avgt std0 min0 max0 min1 max1")


def movie_stats(raw_imgs, mask=None, env=None, factor=1, prefetch=1):
    @jax.jit
    def update(avgt, sumi, sqi, sumn, sqn, cor, min0, max0, imin, imax, index, imgs):
        if mask is None:
            masked = imgs.reshape(batch, h * w)
        else:
            masked = imgs[:, mask]

        avgti = jnp.nanmean(masked, axis=1)
        avgt = avgt.at[index].set(avgti)

        diff = imgs - avgti[..., jnp.newaxis, jnp.newaxis]
        sumi += jnp.nansum(diff, axis=0)
        sqi += jnp.nansum(jnp.square(diff), axis=0)

        neig = neighbor(diff)
        sumn += jnp.nansum(neig, axis=0)
        sqn += jnp.nansum(jnp.square(neig), axis=0)

        cor += jnp.nansum(diff * neig, axis=0)

        min0 = jnp.minimum(min0, jnp.nanmin(imgs, axis=0))
        max0 = jnp.maximum(max0, jnp.nanmax(imgs, axis=0))
        imin = jnp.minimum(imin, jnp.nanmin(diff, axis=0))
        imax = jnp.maximum(imax, jnp.nanmax(diff, axis=0))

        return avgt, sumi, sqi, sumn, sqn, cor, min0, max0, imin, imax

    nt, h, w = raw_imgs.shape

    env = get_gpu_env(env)
    batch = env.batch(float(factor) * h * w, nt)
    nd = env.num_devices
    sharding = env.sharding((nd, 1))

    logger.info("stats batch: %d", batch)
    dataset = tf.data.Dataset.from_generator(
        lambda: zip(range(nt), raw_imgs),
        output_signature=(
            tf.TensorSpec((), tf.int32),
            tf.TensorSpec((h, w), tf.float32),
        ),
    )
    dataset = dataset.batch(batch)
    dataset = dataset.prefetch(prefetch)

    avgt = jnp.empty((nt + 1,))

    sumi = jnp.zeros((h, w))
    sqi = jnp.zeros((h, w))

    sumn = jnp.zeros((h, w))
    sqn = jnp.zeros((h, w))

    cor = jnp.zeros((h, w))

    min0 = jnp.full((h, w), jnp.inf)
    max0 = jnp.full((h, w), -jnp.inf)
    imin = jnp.full((h, w), jnp.inf)
    imax = jnp.full((h, w), -jnp.inf)

    avgt, sumi, sqi, sumn, sqn, cor, min0, max0, imin, imax = (
        jax.device_put(v, sharding) for v in [
            avgt, sumi, sqi, sumn, sqn, cor, min0, max0, imin, imax,
        ]
    )

    logger.info("%s: %s %s %d", "pbar", "start", "stats", nt)
    for data in dataset:
        data = (from_tf(v) for v in data)
        index, imgs = (jax.device_put(v) for v in data)

        count = index.size
        diff = batch - count
        if diff > 0:
            pad = (0, diff), (0, 0), (0, 0)
            index = jnp.pad(index, ((0, diff)), constant_values=-1)
            imgs = jnp.pad(imgs, pad, constant_values=jnp.nan)

        index = jax.device_put(index, sharding)
        imgs = jax.device_put(imgs, sharding)

        avgt, sumi, sqi, sumn, sqn, cor, min0, max0, imin, imax = update(
            avgt, sumi, sqi, sumn, sqn, cor, min0, max0, imin, imax, index, imgs,
        )
        logger.info("%s: %s %d", "pbar", "update", count)
    logger.info("%s: %s", "pbar", "close")

    avgt = avgt[:-1]
    avgx = sumi / nt
    varx = sqi / nt - jnp.square(avgx)
    if mask is None:
        varx_masked = varx.ravel()
    else:
        varx_masked = varx[mask]

    std0 = jnp.sqrt(varx_masked.mean())
    stdx = jnp.sqrt(varx)
    istd = stdx / std0

    avgn = sumn / nt
    stdn = jnp.sqrt(sqn / nt - jnp.square(avgn))
    icor = (cor / nt - avgx * avgn) / (stdx * stdn)

    imin = (imin - avgx) / std0
    imax = (imax - avgx) / std0

    if mask is not None:
        avgx.at[~mask].set(jnp.nan)
        imin.at[~mask].set(jnp.nan)
        imax.at[~mask].set(jnp.nan)
        istd.at[~mask].set(jnp.nan)
        icor.at[~mask].set(jnp.nan)

    stats = avgx, avgt, std0, min0.min(), max0.max(), imin.min(), imax.max()
    simgs = imax, istd, icor
    return Stats(*map(np.array, stats)), *map(np.array, simgs)
