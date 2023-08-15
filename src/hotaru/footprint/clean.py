from collections import namedtuple
from logging import getLogger

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import tensorflow as tf

from ..filter import (
    gaussian_laplace,
)
from ..utils import get_gpu_env, from_tf
from .radius import get_radius
from .reduce import reduce_peaks_simple
from .segment import get_segment_mask

logger = getLogger(__name__)

Footprint = namedtuple("Footprint", "foootprit y x radius intensity")


def clean(uid, vals, shape, mask, radius, density, env, factor, prefetch):
    args = shape, mask, radius, env, factor, prefetch
    segments, y, x, radius, firmness = clean_footprints(vals, *args)
    cell, bg = reduce_peaks_simple(y, x, radius, firmness, density)

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
    peaks = pd.concat([cell, bg, removed], axis=0)

    segments = segments[peaks[peaks.kind != "remove"].index]
    return segments, peaks


def clean_footprints(vals, shape, mask, radius, env=None, factor=1, prefetch=1):
    @jax.jit
    def calc(imgs):
        gl = gaussian_laplace(imgs, radius, -3)
        if mask is not None:
            gl.at[:, :, ~mask].set(jnp.nan)
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
    nk, nx = vals.shape
    h, w = shape

    env = get_gpu_env(env)
    nd = env.num_devices
    sharding = env.sharding((nd, 1))
    batch = env.batch(float(factor) * h * w * len(radius), nk)

    dataset = tf.data.Dataset.from_generator(
        lambda: zip(range(nk), vals),
        output_signature=(
            tf.TensorSpec((), tf.int32),
            tf.TensorSpec((nx,), tf.float32),
        ),
    )
    dataset = dataset.batch(batch)
    dataset = dataset.prefetch(prefetch)

    seg = jnp.empty((nk + 1, h, w))
    y = jnp.empty((nk + 1,), jnp.int32)
    x = jnp.empty((nk + 1,), jnp.int32)
    r = jnp.empty((nk + 1,), jnp.int32)
    g = jnp.empty((nk + 1,), jnp.float32)

    mask = None if mask is None else jax.device_put(jnp.array(mask, bool), sharding)

    logger.info("clean: %s %s", (factor, h, w), batch)
    logger.info("%s: %s %s %d", "pbar", "start", "clean", nk)
    for data in dataset:
        data = (from_tf(v) for v in data)
        idx, val = (jax.device_put(v, sharding) for v in data)

        count = idx.size
        if mask is None:
            img = val.reshape(count, h, w)
        else:
            img = jnp.zeros((count, h, w)).at[mask].set(val)

        diff = batch - count
        if diff > 0:
            pad = (0, diff), (0, 0), (0, 0)
            idx = jnp.pad(idx, pad[:1], constant_values=-1)
            img = jnp.pad(img, pad, constant_values=jnp.nan)

        segi, yi, xi, ri, gi = calc(img)
        seg = seg.at[idx].set(segi)
        y = y.at[idx].set(yi)
        x = x.at[idx].set(xi)
        r = r.at[idx].set(ri)
        g = g.at[idx].set(gi)
        logger.info("%s: %s %d", "pbar", "update", count)
    logger.info("%s: %s", "pbar", "close")

    seg, y, x, r, g = (np.array(v[:-1]) for v in (seg, y, x, r, g))
    r = np.array(radius)[r]
    return Footprint(seg, y, x, r, g)
