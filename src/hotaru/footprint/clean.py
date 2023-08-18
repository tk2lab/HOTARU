from collections import namedtuple
from logging import getLogger

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import tensorflow as tf
from scipy.ndimage import grey_closing

from ..filter import gaussian_laplace
from ..utils import (
    from_tf,
    get_gpu_env,
)
from .radius import get_radius
from .reduce import reduce_peaks_simple
from .segment import get_segment_mask

logger = getLogger(__name__)

Footprint = namedtuple("Footprint", "foootprit y x radius intensity")


def clean(oldstats, segs, radius, select, env, factor, prefetch):
    uid = oldstats.uid.to_numpy()
    kind = oldstats.kind.to_numpy()
    bg = kind == "background"
    cell_to_bg = (kind == "cell") & (oldstats.udense > select.max_udense)
    bg_to_cell = (
        bg
        & (oldstats.bsparse > select.min_bsparse)
        & (oldstats.radius < select.max_radius)
    )
    logger.debug("old_bg: uid=%s", oldstats[bg].uid.to_numpy())
    logger.debug("cell_to_bg: uid=%s", oldstats[cell_to_bg].uid.to_numpy())
    logger.debug("bg_to_cell: uid=%s", oldstats[bg_to_cell].uid.to_numpy())

    bg = list(np.where((bg & ~bg_to_cell) | cell_to_bg)[0])
    select = dict(
        min_radius=select.min_radius,
        max_radius=select.max_radius,
        min_distance_ratio=select.min_distance_ratio.clean,
    )
    args = radius, env, factor, prefetch
    segments, y, x, radius, firmness = clean_footprints(segs, *args)
    cell, bg = reduce_peaks_simple(y, x, radius, firmness, bg, **select)

    peaks = pd.DataFrame(dict(uid=uid, y=y, x=x, radius=radius, firmness=firmness))
    peaks["kind"] = "remove"
    peaks.loc[cell, "kind"] = "cell"
    peaks.loc[bg, "kind"] = "background"
    peaks["sum"] = segments.sum(axis=(1, 2))
    peaks["area"] = np.count_nonzero(segments > 0, axis=(1, 2))

    cell, bg, removed = [
        peaks[peaks.kind == key].sort_values("firmness", ascending=False)
        for key in ["cell", "background", "remove"]
    ]
    logger.info("clean: %d %d %d", cell.shape[0], bg.shape[0], removed.shape[0])

    peaks = pd.concat([cell, bg, removed], axis=0)
    segments = segments[peaks[peaks.kind != "remove"].index]
    return segments, peaks.reset_index(drop=True)


def clean_footprints(segs, radius, env=None, factor=1, prefetch=1):
    @jax.jit
    def calc(imgs):
        gl = gaussian_laplace(imgs, radius, -3)
        nk, nr, h, w = gl.shape
        idx = jnp.argmax(gl.reshape(nk, nr * h * w), axis=1)
        k = jnp.arange(nk)
        r, y, x = idx // (h * w), (idx // w) % h, idx % w
        g = gl[k, r, y, x]
        seg = jax.vmap(get_segment_mask)(gl[k, r], y, x)
        imgs = jnp.where(seg, imgs, jnp.nan)
        dmin = jnp.nanmin(jnp.where(seg, imgs, jnp.nan), axis=(1, 2), keepdims=True)
        dmax = jnp.nanmax(jnp.where(seg, imgs, jnp.nan), axis=(1, 2), keepdims=True)
        imgs = jnp.where(seg, (imgs - dmin) / (dmax - dmin), 0)
        return imgs, y, x, r, g

    radius = get_radius(radius)
    nk, h, w = segs.shape

    env = get_gpu_env(env)
    nd = env.num_devices
    sharding = env.sharding((nd, 1))
    batch = env.batch(float(factor) * h * w * len(radius), nk)

    dataset = tf.data.Dataset.from_generator(
        lambda: zip(range(nk), segs),
        output_signature=(
            tf.TensorSpec((), tf.int32),
            tf.TensorSpec((h, w), tf.float32),
        ),
    )
    dataset = dataset.batch(batch)
    dataset = dataset.prefetch(prefetch)

    out = jnp.empty((nk + 1, h, w))
    y = jnp.empty((nk + 1,), jnp.int32)
    x = jnp.empty((nk + 1,), jnp.int32)
    r = jnp.empty((nk + 1,), jnp.int32)
    g = jnp.empty((nk + 1,), jnp.float32)

    logger.info("clean: %s %s", (factor, h, w), batch)
    logger.info("%s: %s %s %d", "pbar", "start", "clean", nk)
    for data in dataset:
        data = (from_tf(v) for v in data)
        idx, img = (jax.device_put(v, sharding) for v in data)

        count = idx.size
        diff = batch - count
        if diff > 0:
            pad = (0, diff), (0, 0), (0, 0)
            idx = jnp.pad(idx, pad[:1], constant_values=-1)
            img = jnp.pad(img, pad, constant_values=jnp.nan)

        outi, yi, xi, ri, gi = calc(img)
        out = out.at[idx].set(outi)
        y = y.at[idx].set(yi)
        x = x.at[idx].set(xi)
        r = r.at[idx].set(ri)
        g = g.at[idx].set(gi)
        logger.info("%s: %s %d", "pbar", "update", count)
    logger.info("%s: %s", "pbar", "close")

    out, y, x, r, g = (np.array(v[:-1]) for v in (out, y, x, r, g))
    out = grey_closing(out, (1, 10, 10))
    r = np.array(radius)[r]
    return Footprint(out, y, x, r, g)
