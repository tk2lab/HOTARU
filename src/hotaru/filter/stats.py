from collections import namedtuple
from logging import getLogger

import jax.numpy as jnp
import numpy as np
import tensorflow as tf

from ..utils import get_gpu_env
from .map import mapped_imgs
from .neighbor import neighbor

logger = getLogger(__name__)

Stats = namedtuple("Stats", "avgx avgt std0 min0 max0 min1 max1")
StatsImages = namedtuple("StatsImages", "imax istd icor")


def calc_stats(imgs, mask=None, env=None, factor=100):
    def calc(index, imgs):
        imgs = imgs.astype(jnp.float32)
        if mask is None:
            masked = imgs.reshape(-1, h * w)
        else:
            masked = imgs[:, mask]
        min0 = jnp.nanmin(imgs, axis=0)
        max0 = jnp.nanmax(imgs, axis=0)
        avgt = jnp.nanmean(masked, axis=1)
        diff = imgs - avgt[..., None, None]
        imin = jnp.nanmin(diff, axis=0)
        imax = jnp.nanmax(diff, axis=0)
        neig = neighbor(diff)
        sumi = jnp.nansum(diff, axis=0)
        sumn = jnp.nansum(neig, axis=0)
        sqi = jnp.nansum(jnp.square(diff), axis=0)
        sqn = jnp.nansum(jnp.square(neig), axis=0)
        cor = jnp.nansum(diff * neig, axis=0)
        return avgt, min0, max0, imin, imax, sumi, sumn, sqi, sqn, cor, index

    def finish(avgt, min0, max0, imin, imax, sumi, sumn, sqi, sqn, cor):
        avgt = avgt[:-1]
        avgx = sumi / nt
        avgn = sumn / nt
        varx = sqi / nt - jnp.square(avgx)
        stdx = jnp.sqrt(varx)
        stdn = jnp.sqrt(sqn / nt - jnp.square(avgn))
        if mask is None:
            varx_masked = varx.ravel()
        else:
            varx_masked = varx[mask]
        std0 = jnp.sqrt(varx_masked.mean())

        imin = (imin - avgx) / std0
        imax = (imax - avgx) / std0
        istd = stdx / std0
        icor = (cor / nt - avgx * avgn) / (stdx * stdn)

        if mask is not None:
            avgx.at[~mask].set(jnp.nan)
            imin.at[~mask].set(jnp.nan)
            imax.at[~mask].set(jnp.nan)
            istd.at[~mask].set(jnp.nan)
            icor.at[~mask].set(jnp.nan)

        stats = avgx, avgt, std0, min0.min(), max0.max(), imin.min(), imax.max()
        simgs = imax, istd, icor
        return Stats(*map(np.array, stats)), StatsImages(*map(np.array, simgs))

    nt, h, w = imgs.shape
    batch = get_gpu_env(env).batch(float(factor) * h * w, nt)

    logger.info("stats: %s %d", imgs.shape, -1 if mask is None else mask.sum())
    dataset = tf.data.Dataset.from_generator(
        lambda: zip(range(nt), imgs),
        output_signature=(
            tf.TensorSpec((), tf.int32),
            tf.TensorSpec((h, w), imgs.dtype),
        ),
    )
    types = [("stack", -1)] + ["min", "max"] * 2 + ["add"] * 5
    init = [jnp.empty((nt + 1,))] + [jnp.zeros((h, w))] * 9

    logger.info("%s: %s %s %d", "pbar", "start", "stats", nt)
    stats, simgs = finish(*mapped_imgs(dataset, nt, calc, types, init, batch))
    logger.info("%s: %s", "pbar", "close")
    return stats, simgs
