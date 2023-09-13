from collections import namedtuple
from logging import getLogger

import jax
import jax.numpy as jnp
import numpy as np
import pandas  as pd
import tensorflow as tf
from scipy.ndimage import grey_closing

from ..filter import gaussian_laplace
from ..utils import (
    from_tf,
    get_gpu_env,
)
from .radius import get_radius
from .segment import get_segment_mask

# from sklearn.mixture import GaussianMixture as CluModel


logger = getLogger(__name__)

Footprint = namedtuple("Footprint", "foootprit y x radius intensity")


def clean(
    stats,
    segs,
    radius,
    cell_range,
    thr_move,
    env=None,
    factor=1,
    prefetch=1,
    no_bg=False,
):
    stats = stats.sort_values("segid")
    oldy = stats.y.to_numpy()
    oldx = stats.x.to_numpy()
    oldr = stats.radius.to_numpy()

    segments, y, x, radius, firmness = clean_footprints(
        segs,
        radius,
        env,
        factor,
        prefetch,
    )
    stats["y"] = y
    stats["x"] = x
    stats["radius"] = radius
    stats["firmness"] = firmness
    stats["pos_move"] = np.hypot(x - oldx, y - oldy) / oldr

    stats["old_kind"] = stats.kind
    stats["kind"] = "cell"

    bg_cond = (stats.old_kind == "background")
    remove_cond = (stats.old_kind == "cell") & (stats.pos_move > thr_move)
    if no_bg:
        remove_cond |= (stats.radius > cell_range[-1])
    else:
        bg_cond |= (stats.radius > cell_range[-1])
    remove_cond |= (stats.radius < cell_range[0])

    stats.loc[bg_cond, "kind"] = "background"
    stats.loc[remove_cond, "kind"] = "remove"
    stats["min_dist_id"] = -1
    stats["max_dup_id"] = -1

    cell_cond = (stats.kind == "cell").to_numpy()
    cell_df = stats.loc[cell_cond].copy()

    ys = y[cell_cond]
    xs = x[cell_cond]
    rs = radius[cell_cond]

    dist_mat = np.hypot(xs[:, np.newaxis] - xs, ys[:, np.newaxis] - ys) / rs
    np.fill_diagonal(dist_mat, np.inf)
    cell_df["min_dist_id"] = cell_df.index[np.argmin(dist_mat, axis=0)]
    cell_df["min_dist"] = dist_mat.min(axis=0)

    val_mat = segments[cell_cond][:, ys, xs]
    np.fill_diagonal(val_mat, 0)
    cell_df["max_dup_id"] = cell_df.index[np.argmax(val_mat, axis=0)]
    cell_df["max_dup"] = val_mat.max(axis=0)

    bg_df = stats.query("kind == 'background'").copy()
    remove_df = stats.query("kind == 'remove'").copy()
    stats = pd.concat([cell_df, bg_df, remove_df], axis=0)
    stats["spkid"] = -1
    stats["bgid"] = -1

    nc = cell_df.shape[0]
    nb = bg_df.shape[0]
    nr = remove_df.shape[0]
    logger.info("cell/bg/remove = %d/%d/%d", nc, nb, nr)
    return stats, segments


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

    logger.debug("clean: %s %s", (factor, h, w), batch)
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


def get_dupfilter(**args):
    match args:
        case {"thr_active_area": thr}:
            return DupFilter(thr)
        case _:
            raise ValueError()


def get_bgfilter(**args):
    match args:
        case {"kind": "new", **kwargs}:
            return BackgroundFilter(**kwargs)
        case {"kind": "simple", **kwargs}:
            return SimpleBackgroundFilter(**kwargs)
        case {"kind": "factor", **kwargs}:
            return FactorBackgroundFilter(**kwargs)
        case _:
            return OldBackgroundFilter(**args)


class DupFilter:
    def __init__(self, thr_active_area):
        self._thr = thr_active_area

    def set(self, stats, segments):
        self._y = stats.y.to_numpy()
        self._x = stats.x.to_numpy()
        self._segmask = segments > self._thr

    def dup_id(self, i, js):
        yi, xi = self._y[i], self._x[i]
        for j in js:
            yj, xj = self._y[j], self._x[j]
            if self._segmask[i, yj, xj] and self._segmask[j, yi, xi]:
                return j
        return -1


class BackgroundFilter:
    def __init__(
        self,
        bias=0,
        coef_log_snr=0,
        coef_firmness=0,
    ):
        self._bias = bias
        self._coef_log_snr = coef_log_snr
        self._coef_firmness = coef_firmness

    def set(self, stats):
        self._stats = stats

    def is_background(self, i):
        s = self._stats
        si = s.iloc[i]
        score = (
            self._bias
            + self._coef_log_snr * np.log10(si.snratio)
            + self._coef_firmness * si.firmness
        )
        return score < 0


class FactorBackgroundFilter:
    def __init__(
        self,
        thr_rsn_factor=10,
        thr_firmness=0,
    ):
        self._thr_s = thr_rsn_factor
        self._thr_f = thr_firmness

    def set(self, stats):
        stats["rsn"] = 1 / stats.snratio
        med = np.median(stats.rsn)
        std = 1.4826 * np.median(np.abs(stats.rsn - med))
        stats["zrsn"] = (stats.rsn - med) / std
        self._stats = stats

    def is_background(self, i):
        s = self._stats
        si = s.iloc[i]
        return (
            (si.zrsn > self._thr_s)
            or (si.firmness < self._thr_f)
        )


class SimpleBackgroundFilter:
    def __init__(
        self,
        thr_snr_inv=1,
        thr_firmness=0,
    ):
        self._thr_s = thr_snr_inv
        self._thr_f = thr_firmness

    def set(self, stats):
        self._stats = stats

    def is_background(self, i):
        s = self._stats
        si = s.iloc[i]
        return (
            (1 / si.snratio > self._thr_s)
            or (si.firmness < self._thr_f)
        )


class OldBackgroundFilter:
    def __init__(
        self,
        thr_bg_udense=1,
        thr_bg_signal=0,
        thr_bg_firmness=0,
        thr_bg_cluster=1,
    ):
        self._thr_d = thr_bg_udense
        self._thr_s = thr_bg_signal
        self._thr_f = thr_bg_firmness
        self._thr_c = thr_bg_cluster

    def set(self, stats):
        """
        cindex = oldstats.query("kind == 'cell'").cid
        v = np.stack([udense, firmness, np.log(signal)], axis=1)
        v = v[cindex]
        clu = CluModel(
            n_components=2,
            weights_init=[0.95, 0.05],
            means_init=[[0.03, 0.45, 0.5], [0.13, 0.35, -0.3]],
        )
        clu.fit(v)
        z[cindex] = clu.predict_proba(v)[:, 1]
        self._stats.
        """
        self._stats = stats

    def is_background(self, i):
        s = self._stats
        si = s.at[s.index[i]]
        return (
            # (si.z > thr_bg_cluster)
            (si.udense > self._thr_d)
            or (si.signal < self._thr_s)
            or (si.firmness < self._thr_f)
        )
