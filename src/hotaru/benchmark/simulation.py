from logging import getLogger
from math import nan

import numpy as np
import polars as pl
from keras import ops
from keras.saving import load_model
from keras.saving import save_model
from tqdm import trange

from ..data.io import make_movie
from ..random import Generator
from ..saving import auto_save_config
from ..spatial import Footprints
from ..spatial.simulation import sim_footprints
from ..spatial.simulation import sim_neuropil_basis
from ..temporal import DoubleExpKernel
from ..temporal import ExpKernel
from ..temporal import Traces
from ..temporal.simulation import sim_dendrites
from ..temporal.simulation import sim_neuropil_traces
from ..temporal.simulation import sim_spikes

logger = getLogger(__name__)


@auto_save_config(0.7)
def make_cell(
    path,
    num_frames,
    height,
    width,
    hz,
    num,
    radius_m,
    radius_s,
    overwrap,
    noise,
    intensity_mm,
    intensity_mb,
    intensity_s,
    isi_mm,
    isi_ms,
    isi_sm,
    isi_ss,
    tau1,
    tau2,
    upsample_factor,
    seed,
):
    rng = Generator(seed)

    kernel = DoubleExpKernel(tau1, tau2, hz=hz, scale='max')
    spks = sim_spikes(
        num_frames * upsample_factor + kernel.pad_size(upsample_factor),
        num,
        num_frames,
        intensity_mm,
        intensity_mb,
        intensity_s,
        isi_mm * hz * upsample_factor,
        isi_ms,
        isi_sm,
        isi_ss,
        rng=rng,
    )
    trs = kernel(spks, upsample_factor=upsample_factor)[:, ::upsample_factor]
    save_model(trs_model := Traces(trs), path / 'trs.keras')

    r_cell = [radius_m] * num
    s_cell = [radius_s] * num
    fps = sim_footprints(height, width, r_cell, r_cell, s_cell, noise, overwrap, rng)
    save_model(fps_model := Footprints(fps), path / 'fps.keras')

    stats = make_stats(spks, trs, fps)
    stats.write_csv(path / 'stats.csv')
    logger.info('ground truth stats: %s', stats.describe())

    imgs = make_imgs(fps_model, trs_model, path)
    make_movie(imgs, hz, path)

    return path


@auto_save_config(0.2)
def make_dendrite(
    path,
    cell_path,
    num_frames,
    height,
    width,
    hz,
    num,
    radius_m1,
    radius_m2,
    radius_s,
    tau,
    beta,
    num_mix,
    noise,
    seed,
):
    _ = num_frames
    _ = hz
    rng = Generator(seed)

    cell_fps = load_model(cell_path / 'fps.keras')
    cell_trs = load_model(cell_path / 'trs.keras')

    kernel = ExpKernel(tau, hz=20.0)
    r1_dend_n = [radius_m1] * num
    r2_dend_n = [radius_m2] * num
    s_dend_n = [radius_s] * num
    fps = sim_footprints(height, width, r1_dend_n, r2_dend_n, s_dend_n, noise, 2.0, rng)
    save_model(fps := Footprints(fps), path / 'fps.keras')

    yc, xc = cell_fps.peaks.T
    yd, xd = fps.peaks.T[:, :, None]
    dist_mat = ops.hypot(yc - yd, xc - xd)
    trs = sim_dendrites(
        dist_mat,
        cell_trs.obs.value,
        kernel.kernel(),
        beta,
        num_mix,
        rng=rng,
    )
    save_model(trs := Traces(trs), path / 'trs.keras')

    _imgs = make_imgs(fps, trs, path)
    # make_movie(imgs, hz, path)


@auto_save_config(0.3)
def make_neuropil(path, num_frames, height, width, hz, scale, n_basis, tau, seed):
    rng = Generator(seed)

    fps = sim_neuropil_basis(height, width, scale, n_basis)
    save_model(fps := Footprints(fps), path / 'neuropil_x.keras')

    trs = sim_neuropil_traces(fps.shape[0], num_frames, tau, rng, hz=hz)
    save_model(trs := Traces(trs), path / 'neuropil_t.keras')

    _imgs = make_imgs(fps, trs, path)
    # vmax = np.max(np.abs(imgs))
    # make_movie(imgs, hz, path, -vmax, vmax, 'bwr')


@auto_save_config(0.3)
def make_sim(
    path,
    cell,
    dendrite,
    neuropil,
    hz,
    w_dendrite,
    w_neuropil,
    poisson_base,
    poisson_intensity,
    gauss_intensity,
    seed,
):
    cell = np.load(cell / 'imgs.npy')
    dendrite = np.load(dendrite / 'imgs.npy')
    neuropil = np.load(neuropil / 'imgs.npy')
    nt, h, w = cell.shape

    print(cell.min(), cell.max(), cell.std())
    print(dendrite.min(), dendrite.max(), dendrite.std())
    print(neuropil.min(), neuropil.max(), neuropil.std())

    imgs = np.lib.format.open_memmap(path / 'imgs.npy', 'w+', 'float32', cell.shape)
    for t in trange(nt, ncols=150, desc='mix'):
        imgs[t] = cell[t] + w_dendrite * dendrite[t] + w_neuropil * neuropil[t]

    rng = Generator(seed)
    for t in trange(nt, ncols=150, desc='add noise'):
        v = ops.maximum(0, imgs[t] + poisson_base)
        s = ops.square(poisson_intensity) * v + ops.square(gauss_intensity)
        imgs[t] += ops.convert_to_numpy(rng.normal(0, ops.sqrt(s), shape=(h, w)))

    make_movie(imgs, hz, path)


def make_imgs(fps, trs, path, scale=1.0, batch_size=200):
    fps_val = fps.segs.numpy()
    trs_val = trs.obs.numpy()
    nk, h, w = fps_val.shape
    _nk, nt = trs_val.shape

    fps_flat = fps_val.reshape(nk, h * w)
    out = np.lib.format.open_memmap(path / 'imgs.npy', 'w+', 'float32', (nt, h, w))
    for s in trange(0, nt, batch_size, ncols=150, desc='mix fps and trs'):
        e = min(s + batch_size, nt)
        out[s:e] = scale * np.reshape(trs_val[:, s:e].T @ fps_flat, (e - s, h, w))
    return out


def make_stats(spks, trs, fps):
    spks = ops.convert_to_numpy(spks)
    trs = ops.convert_to_numpy(trs)
    nk, nt = spks.shape
    fps = np.reshape(fps, (nk, -1))

    events = spks > 0
    idx_matrix = np.arange(nt, dtype='int32')
    masked_indices = np.where(events, idx_matrix, nt)
    sorted_indices = np.sort(masked_indices, axis=1)
    valid_isi_mask = (sorted_indices[:, 1:] < nt) & (sorted_indices[:, :-1] < nt)
    isi_darty = np.diff(sorted_indices, axis=1)
    isi = np.where(valid_isi_mask, isi_darty, nan)

    num_spk = np.sum(events, axis=1)
    intensity = np.sum(spks, axis=1) / num_spk
    cv_isi = np.nanstd(isi, axis=1) / np.nanmean(isi)

    px = np.sum(np.square(fps), axis=1)
    py = np.sum(np.square(trs), axis=1)
    power = px * py
    sx = fps / np.sqrt(px[:, None])
    overwrap = np.sum((sx @ sx.T) * power, axis=1) / power - 1

    stats = {
        'power': power,
        'area': fps.sum(axis=1),
        'intensity': intensity,
        'num_spk': num_spk,
        'cv_isi': cv_isi,
        'overwrap': overwrap,
    }
    return pl.DataFrame(stats)
