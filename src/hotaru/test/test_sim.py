import keras
import numpy as np
from keras import ops
from matplotlib.pyplot import get_cmap
from tqdm import trange

from hotaru.data.io import to_movie
from hotaru.random import Generator
from hotaru.spatial.simulation import sim_footprints
from hotaru.spatial.simulation import sim_neuropil_basis
from hotaru.temporal import DoubleExpKernel
from hotaru.temporal import ExpKernel
from hotaru.temporal import Traces
from hotaru.temporal.simulation import sim_dendrites
from hotaru.temporal.simulation import sim_neuropil_traces
from hotaru.temporal.simulation import sim_traces

nt, h, w = 1200, 300, 400
hz = 20.0
upsample_factor = 10

neuropil_scale = 400
neuropil_tau = 1.0
n_neuropil_basis = 7

n_pycell = 1000
r_pycell = 6.0
s_pycell = 0.06
g_s_pycell = 1.0
isi_mm_pycell = 1.0
isi_ms_pycell = 0.1
isi_sm_pycell = 2.0
isi_ss_pycell = 0.1

n_incell = 0
r_incell = 4.0
s_incell = 0.06
g_s_incell = 1.0
isi_mm_incell = 0.1
isi_ms_incell = 0.1
isi_sm_incell = 10.0
isi_ss_incell = 0.1

cell_noise = 0.05
cell_overwrap = 0.2

n_dend = 2000
r1_dend = 2.0
r2_dend = 8.0
s_dend = 0.2
dend_noise = 0.25
dend_overwrap = 2.0
dend_beta = 1.0
dend_num_mix = 5

shot_base = 1.0
shot_noise = 0.5
gauss_noise = 0.5

rng = Generator()


def test_neuropil():
    fps = sim_neuropil_basis(h, w, neuropil_scale, n_neuropil_basis)
    fps_val = fps.segs.numpy()
    keras.saving.save_model(fps, 'neuropil_x.keras')

    trs = sim_neuropil_traces(fps.shape[0], nt, neuropil_tau, rng, hz=hz)
    trs_val = trs.obs.numpy()
    keras.saving.save_model(trs, 'neuropil_t.keras')

    out = np.lib.format.open_memmap('neuropil.npy', 'w+', 'float32', (nt, h, w))
    for t in trange(nt, ncols=150, desc='mix fps and trs'):
        out[t] = np.einsum('kyx,k->yx', fps_val, trs_val[:, t])

    vmax = max(-out.min(), out.max())
    cmap = get_cmap('bwr')

    def img_iter():
        for o in out:
            v = (o + vmax) / (2 * vmax)
            yield (255 * cmap(v)).astype('uint8')

    to_movie('neuropil.mp4', img_iter(), out.shape, hz)


def test_cell():
    r_cell = [r_pycell] * n_pycell + [r_incell] * n_incell
    s_cell = [s_pycell] * n_pycell + [s_incell] * n_incell
    fps = sim_footprints(h, w, r_cell, r_cell, s_cell, cell_noise, cell_overwrap, rng)
    fps_val = fps.segs.numpy()
    keras.saving.save_model(fps, 'cell_x.keras')

    kernel = DoubleExpKernel(0.08, 0.16, hz=20.0)
    isi_mm = [isi_mm_pycell] * n_pycell + [isi_mm_incell] * n_incell
    isi_ms = [isi_ms_pycell] * n_pycell + [isi_ms_incell] * n_incell
    isi_sm = [isi_sm_pycell] * n_pycell + [isi_sm_incell] * n_incell
    isi_ss = [isi_ss_pycell] * n_pycell + [isi_ss_incell] * n_incell
    g_s = [g_s_pycell] * n_pycell + [g_s_incell] * n_incell
    trs = sim_traces(nt, upsample_factor, kernel, isi_mm, isi_ms, isi_sm, isi_ss, g_s, rng)
    trs_val = trs.obs.numpy()
    keras.saving.save_model(trs, 'cell_t.keras')

    out = np.lib.format.open_memmap('cell.npy', 'w+', 'float32', (nt, h, w))
    for t in trange(nt, ncols=150, desc='mix fps and trs'):
        out[t] = np.einsum('kyx,k->yx', fps_val, trs_val[:, t])

    vmin, vmax = out.min(), out.max()
    cmap = get_cmap('Greens')

    def img_iter():
        for o in out:
            v = (o - vmin) / (vmax - vmin)
            yield (255 * cmap(v)).astype('uint8')

    to_movie('cell.mp4', img_iter(), out.shape, hz)


def test_dendrite():
    kernel = ExpKernel(1.0, hz=20.0)
    r1_dend_n = [r1_dend] * n_dend
    r2_dend_n = [r2_dend] * n_dend
    s_dend_n = [s_dend] * n_dend
    fps = sim_footprints(h, w, r1_dend_n, r2_dend_n, s_dend_n, dend_noise, cell_overwrap, rng)
    fps_val = fps.segs.value
    keras.saving.save_model(fps, 'dend_x.keras')
    cell_fps = keras.saving.load_model('cell_x.keras')
    cell_trs = keras.saving.load_model('cell_t.keras')
    yc, xc = cell_fps.peaks.T
    yd, xd = fps.peaks.T[:, :, None]
    dist_mat = ops.hypot(yc - yd, xc - xd)
    trs = sim_dendrites(
        dist_mat,
        cell_trs.obs.value,
        kernel.kernel(),
        dend_beta,
        dend_num_mix,
        rng=rng,
    )
    trs = Traces(trs)
    trs_val = trs.obs.value
    keras.saving.save_model(trs, 'dend_t.keras')

    out = np.lib.format.open_memmap('dend.npy', 'w+', 'float32', (nt, h, w))
    for t in trange(nt, ncols=150, desc='mix fps and trs'):
        out[t] = np.einsum('kyx,k->yx', fps_val, trs_val[:, t])

    vmin, vmax = out.min(), out.max()
    cmap = get_cmap('Greens')

    def img_iter():
        for o in out:
            v = (o - vmin) / (vmax - vmin)
            yield (255 * cmap(v)).astype('uint8')

    to_movie('dend.mp4', img_iter(), out.shape, hz)


def test_sim():
    cell = np.load('cell.npy', mmap_mode='r')
    dend = np.load('dend.npy', mmap_mode='r')
    neuropil = np.load('neuropil.npy', mmap_mode='r')
    print(cell.min(), dend.min(), neuropil.min())
    print(cell.max(), dend.max(), neuropil.max())

    out = np.lib.format.open_memmap('imgs.npy', 'w+', 'float32', cell.shape)
    nt, h, w = out.shape
    for t in trange(nt, ncols=150, desc='mix'):
        out[t] = cell[t] + 0.01 * dend[t] + 100.0 * neuropil[t]

    vmin = out.min() - shot_base
    for t in trange(nt, ncols=150, desc='add noise'):
        v = out[t] - vmin
        v += ops.convert_to_numpy(rng.normal(shape=(h, w))) * np.sqrt(shot_noise * v)
        v += ops.convert_to_numpy(rng.normal(shape=(h, w))) * gauss_noise
        out[t] = v

    vmin, vmax = out.min(), out.max()
    cmap = get_cmap('Greens')

    def img_iter():
        for o in out:
            v = (o - vmin) / (vmax - vmin)
            yield (255 * cmap(v)).astype('uint8')

    to_movie('imgs.mp4', img_iter(), out.shape, hz)


# test_neuropil()
#test_cell()
#test_dendrite()
test_sim()
