from pathlib import Path

import numpy as np
from keras.saving import load_model
from keras.saving import save_model

from hotaru.data import CalciumImagingDataWithStats
from hotaru.simulation import greedy_matching_r2
from hotaru.spatial import Footprints
from hotaru.spatial import PeakList
from hotaru.spatial import PeakMap
from hotaru.spatial import Radius
from hotaru.temporal import Traces

base_path = Path('benchmark')
run_path = base_path / 'hotaru'
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
