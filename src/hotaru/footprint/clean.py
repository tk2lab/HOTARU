from collections import namedtuple
from logging import getLogger

import tensorflow as tf
import jax
import jax.numpy as jnp
import pandas as pd
import numpy as np

from ..filter import (
    gaussian_laplace,
    mapped_imgs,
)
from ..utils import get_gpu_env
from .radius import get_radius
from .segment import get_segment_mask

logger = getLogger(__name__)

Footprint = namedtuple("Footprint", "foootprit y x r intensity")


def clean(old_peaks, vals, shape, mask, radius, env, factor):
    args = vals, shape, mask, radius, env, factor
    segments, y, x, r, firmness = clean_footprints(*args)
    old_peaks = old_peaks[old_peaks.kind != "remove"]
    nr = len(radius)
    kind = old_peaks.kind.to_numpy()
    kind[r == 0] = "remove"
    kind[r == nr - 1] = "background"

    peaks = pd.DataFrame(
        dict(
            uid=old_peaks.uid.to_numpy(),
            y=y,
            x=x,
            radius=np.array(radius)[r],
            firmness=firmness,
            kind=kind,
        )
    )
    peaks["sum"] = segments.sum(axis=(1, 2))
    peaks["area"] = np.count_nonzero(segments > 0, axis=(1, 2))

    cell, bg, removed = [
        peaks[peaks.kind == key].sort_values("firmness", ascending=False)
        for key in ["cell", "background", "remove"]
    ]
    peaks = pd.concat([cell, bg, removed], axis=0)
    logger.info("%s", peaks)
    segments = segments[peaks[peaks.kind != "remove"].index]
    return segments, peaks


def clean_footprints(vals, shape, mask, radius, env, factor):
    def calc(index, vals):
        if mask is None:
            imgs = vals.reshape(index.size, *shape)
        else:
            imgs = jnp.zeros((index.size, *mask.shape), vals.dtype)
            imgs.at[:, mask].set(vals)
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
        return imgs, r, y, x, g, index

    radius = get_radius(radius)
    nk, nx = vals.shape
    h, w = shape

    dataset = tf.data.Dataset.from_generator(
        lambda: enumerate(vals),
        output_signature=(
            tf.TensorSpec((), tf.int32),
            tf.TensorSpec((nx,), tf.float32),
        ),
    )

    batch = get_gpu_env(env).batch(float(factor) * h * w * len(radius), nk)
    types = [("stack", -1)] * 5
    init = [
        jnp.empty((nk + 1, *shape)),
        jnp.empty((nk + 1,), jnp.int32),
        jnp.empty((nk + 1,), jnp.int32),
        jnp.empty((nk + 1,), jnp.int32),
        jnp.empty((nk + 1,), jnp.float32),
    ]
    logger.info("clean: %s %s", (factor, h, w), batch)
    logger.info("%s: %s %s %d", "pbar", "start", "clean", nk)
    imgs, r, y, x, firmness = mapped_imgs(dataset, nk, calc, types, init, batch)
    logger.info("%s: %s", "pbar", "close")
    return Footprint(*(np.array(v[:-1]) for v in (imgs, y, x, r, firmness)))
