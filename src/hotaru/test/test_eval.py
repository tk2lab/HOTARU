from pathlib import Path

import numpy as np
from keras.saving import load_model
from keras.saving import save_model
from suite2p.detection import detection_wrapper
from suite2p.extraction import extraction_wrapper

from hotaru.data import CalciumImagingDataWithStats
from hotaru.simulation import greedy_matching_r2
from hotaru.spatial import Footprints
from hotaru.spatial import PeakList
from hotaru.spatial import PeakMap
from hotaru.spatial import Radius
from hotaru.temporal import Traces

base_path = Path('benchmark')
run_path = base_path / 'run'
thr = 2.0

gt_fps = load_model(base_path / 'gt' / 'cell_x.keras')
gt_fps_val = np.nan_to_num(gt_fps.segs.numpy(), nan=0)
scale = np.sqrt(np.sum(np.square(gt_fps_val), axis=(1, 2), keepdims=True))
gt_fps_val /= np.where(scale > 0, scale, 1)

gt_trs = load_model(base_path / 'gt' / 'cell_t.keras')
gt_trs_val = np.nan_to_num(gt_trs.obs.numpy(), nan=0)
gt_trs_val -= gt_trs_val.mean(axis=1, keepdims=True)
scale = np.sqrt(np.sum(np.square(gt_trs_val), axis=1, keepdims=True))
gt_trs_val /= np.where(scale > 0, scale, 1)

path = run_path / 'stats.keras'
if path.exists():
    print('load stats')
    data = load_model(path)
else:
    data = CalciumImagingDataWithStats(path='benchmark/imgs.npy', hz=20.0)
    data.calc(batch_size=100)
    save_model(data, path)
print(data.shape, data.dtype)
nt, h, w = data.shape

# print(data.imgs.min(), data.imgs.max())
# imgs = (32767 * (data.imgs - data.imgs.min()) / (data.imgs.max() - data.imgs.min())).astype('int16')

detect_outputs, stat, _ = detection_wrapper(
    data.imgs,
    diameter=[12.0, 12.0],
    tau=0.16,
    fs=20.0,
    yrange=[0, h],
    xrange=[0, w],
)
np.save(run_path / 'suite2p.npy', stat)

nk = stat.size
re_fps_val = np.zeros((nk, h, w))
for i, si in enumerate(stat):
    re_fps_val[i, si['ypix'], si['xpix']] = si['lam']
scale = np.sqrt(np.sum(np.square(re_fps_val), axis=(1, 2), keepdims=True))
re_fps_val /= np.where(scale > 0, scale, 1)

r2_mat = np.square(re_fps_val.reshape(-1, h * w) @ gt_fps_val.reshape(-1, h * w).T)
pair, r2_ind = greedy_matching_r2(r2_mat)
print(nk, (r2_ind > 0.6).sum())
with np.printoptions(3, suppress=True):
    print(r2_ind[:20])

f, f_neu, _, _ = extraction_wrapper(
    stat,
    data.imgs,
)
np.save(run_path / 'suite2p_f.npy', f)
np.save(run_path / 'suite2p_fneu.npy', f_neu)

re_trs_val = f - 0.7 * f_neu
re_trs_val -= re_trs_val.mean(axis=1, keepdims=True)
scale = np.sqrt(np.sum(np.square(re_trs_val), axis=1, keepdims=True))
re_trs_val /= np.where(scale > 0, scale, 1)
print(re_trs_val.shape, gt_trs_val.shape)

trs_cor = np.array([np.inner(re_trs_val[i], gt_trs_val[j]) for i, j in pair])
with np.printoptions(3, suppress=True):
    print(trs_cor[:20])
print(trs_cor.mean(), (trs_cor > 0.6).sum())
for (i, j), a, b in zip(pair, r2_ind, trs_cor, strict=True):
    print(i, j, a, b)


"""
path = run_path / 'cell_peakmap.keras'
if path.exists():
    print('load peakmap')
    peakmap = load_model(path)
else:
    radius = Radius(kind='logscale', min=2.0, max=16.0, num=10)
    print(radius)
    peakmap = PeakMap(radius)
    peakmap.calc(data, batch_size=100)
    save_model(peakmap, path)

path = run_path / f'cell_peaklist_{thr}.keras'
if path.exists():
    print('load peaklist')
    peaklist = load_model(path)
else:
    peaklist = PeakList(thr)
    peaklist.calc(peakmap, block_size=100)
    save_model(peaklist, path)

path = run_path / f'cell_fps0_{thr}.keras'
if path.exists():
    print('load fps')
    cell_fps = load_model(path)
else:
    cell_fps = Footprints.from_peaklist(data, peaklist, batch_size=100)
    save_model(cell_fps, path)
re_val = np.nan_to_num(cell_fps.segs.numpy(), nan=0)
scale = np.sqrt(np.sum(np.square(re_val), axis=(1, 2), keepdims=True))
re_val /= np.where(scale > 0, scale, 1)

nk, h, w = re_val.shape
r2_mat = np.square(re_val.reshape(-1, h * w) @ gt_val.reshape(-1, h * w).T)
pair, r2_ind = greedy_matching_r2(r2_mat)
print((r2_ind > 0.6).sum())
"""
