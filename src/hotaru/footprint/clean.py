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
from .segment import get_segment_mask

logger = getLogger(__name__)

Footprint = namedtuple("Footprint", "foootprit y x radius intensity")


def clean(
    oldstats,
    segs,
    radius,
    cell_range,
    thr_active_area,
    thr_bg_udense,
    thr_bg_signal,
    thr_bg_firmness,
    thr_cell_bsparse,
    env,
    factor,
    prefetch,
):
    logger.info(
        "clean: %f %f %f %s",
        thr_active_area,
        thr_bg_udense,
        thr_bg_firmness,
        thr_cell_bsparse,
    )
    oldstats = oldstats.query("kind != 'remove'")

    segments, y, x, radius, firmness = clean_footprints(
        segs,
        radius,
        env,
        factor,
        prefetch,
    )
    nk, h, w = segments.shape
    segmask = segments > thr_active_area

    def is_dup(i, js):
        yi, xi = y[i], x[i]
        for j in js:
            yj, xj = y[j], x[j]
            if segmask[i, yj, xj] and segmask[j, yi, xi]:
                return True
        return False

    kind = oldstats.kind.to_numpy()
    signal = oldstats.signal.to_numpy()
    udense = oldstats.udense.to_numpy()
    bsparse = oldstats.udense.to_numpy()

    flg = np.argsort(firmness)[::-1]
    cell = []
    bg = []
    remove = []
    while flg.size > 0:
        i, flg = flg[0], flg[1:]
        if radius[i] < cell_range[0]:
            logger.debug("remove small %s %s %s", i, radius[i], cell_range[0])
            remove.append(i)
        elif (
            (
                (kind[i] == "background")
                and ((thr_cell_bsparse is None) or (bsparse[i] < thr_cell_bsparse))
            )
            or (udense[i] > thr_bg_udense)
            or (signal[i] < thr_bg_signal)
            or (firmness[i] < thr_bg_firmness)
            or (radius[i] > cell_range[1])
        ):
            if bg and is_dup(i, bg):
                logger.debug(
                    "remove dup bg %s %s %s %s", i, kind[i], udense[i], radius[i]
                )
                remove.append(i)
            else:
                logger.debug("bg %s %s %s %s", i, kind[i], udense[i], radius[i])
                bg.append(i)
        else:
            if cell and is_dup(i, cell):
                logger.debug(
                    "remove dup cell %s %s %s %s", i, kind[i], udense[i], radius[i]
                )
                remove.append(i)
            else:
                logger.debug("cell %s %s %s %s", i, kind[i], udense[i], radius[i])
                cell.append(i)

    kind = pd.Series(["remove"] * nk)
    kind[cell] = "cell"
    kind[bg] = "background"
    stats = pd.DataFrame(
        dict(
            uid=oldstats.uid.to_numpy(),
            kind=kind,
            y=y,
            x=x,
            radius=radius,
            firmness=firmness,
            signal=None,
            asum=segments.sum(axis=(1, 2)),
            udense=None,
            bmax=None,
            bsparse=None,
            **{
                f"old_{k}": oldstats[k].to_numpy()
                for k in (
                    "kind",
                    "radius",
                    "intensity",
                    "firmness",
                    "signal",
                    "asum",
                    "udense",
                    "bmax",
                    "bsparse",
                )
                if k in oldstats.columns
            },
        )
    )

    cell, bg, removed = [
        stats[stats.kind == key].sort_values("firmness", ascending=False)
        for key in ["cell", "background", "remove"]
    ]
    logger.info("clean: %d %d %d", cell.shape[0], bg.shape[0], removed.shape[0])

    """
    x, y, r = cell.x.to_numpy(), cell.y.to_numpy(), cell.radius.to_numpy()
    dist = np.hypot(x - x[:, np.newaxis], y - y[:, np.newaxis]) / r
    np.fill_diagonal(dist, np.inf)
    dist = np.sort(dist.ravel())
    logger.info("small distance: %s", dist[dist < 1])
    """

    stats = pd.concat([cell, bg, removed], axis=0)
    segments = segments[stats.query("kind != 'remove'").index]
    logger.info("segments.shape=%s", segments.shape)
    return segments, stats.reset_index(drop=True)


def clean_footprints(segs, radius, env=None, factor=1, prefetch=1):
    @jax.jit
    def calc(imgs):
        imgs /= imgs.max(axis=(1, 2), keepdims=True)
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
