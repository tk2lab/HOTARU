from pathlib import Path

import numpy as np
from keras import ops
from keras.saving import load_model
from keras.saving import save_model
from suite2p import default_settings
from suite2p.classification import builtin_classfile
from suite2p.classification import classify
from suite2p.detection import assign_overlaps
from suite2p.detection import detection_wrapper
from suite2p.extraction import extraction_wrapper
from suite2p.extraction import oasis
from suite2p.extraction import preprocess

from hotaru.data import CalciumImagingDataWithStats
from hotaru.simulation import greedy_matching_r2
from hotaru.spatial import Footprints  # noqa
from hotaru.temporal import ExpKernel
from hotaru.temporal import Traces  # noqa

settings = default_settings()

base_path = Path('benchmark')
run_path = base_path / 'run'
thr = 2.0

path = run_path / 'stats.keras'
if path.exists():
    print('load stats')
    data = load_model(path)
else:
    data = CalciumImagingDataWithStats(path='benchmark/imgs.npy', hz=20.0)
    data.calc(batch_size=100)
    save_model(data, path)
nt, h, w = data.shape

gt_fps = load_model(base_path / 'gt' / 'cell_x.keras')
gt_fps_val = np.nan_to_num(gt_fps.segs.numpy(), nan=0)
scale = np.sqrt(np.sum(np.square(gt_fps_val), axis=(1, 2), keepdims=True))
gt_fps_val /= np.where(scale > 0, scale, 1)

gt_trs = load_model(base_path / 'gt' / 'cell_t.keras')
gt_trs_val = np.nan_to_num(gt_trs.obs.numpy(), nan=0)
gt_trs_val -= gt_trs_val.mean(axis=1, keepdims=True)
scale = np.sqrt(np.sum(np.square(gt_trs_val), axis=1, keepdims=True))
gt_trs_val /= np.where(scale > 0, scale, 1)

detect_outputs, stat, _ = detection_wrapper(
    data.imgs,
    diameter=[12.0, 12.0],
    tau=0.16,
    fs=20.0,
    yrange=[0, h],
    xrange=[0, w],
)
print(stat.size)

iscell = classify(stat, builtin_classfile)
stat = stat[iscell[:, 0].astype(bool)]
print(stat.size)

f, f_neu, _, _ = extraction_wrapper(
    stat,
    data.imgs,
)

df = f - 0.7 * f_neu
snr = 1 - 0.5 * np.diff(df, axis=-1).var(axis=1) / df.var(axis=1)
keep_rois = snr > 0.3
print(keep_rois.sum())

stat = stat[keep_rois]
stat = assign_overlaps(stat, h, w)

f, f_neu, _, _ = extraction_wrapper(
    stat,
    data.imgs,
)

df = f - 0.7 * f_neu
snr = 1 - 0.5 * np.diff(df, axis=-1).var(axis=1) / df.var(axis=1)
keep_rois = snr > 0.0
print(keep_rois.sum())
print(df.shape)

nk = stat.size
re_fps_val = np.zeros((nk, h, w))
for i, si in enumerate(stat):
    re_fps_val[i, si['ypix'], si['xpix']] = si['lam']
re_fps_val -= re_fps_val.mean(axis=(1, 2), keepdims=True)
scale = np.sqrt(np.sum(np.square(re_fps_val), axis=(1, 2), keepdims=True))
re_fps_val /= np.where(scale > 0, scale, 1)

df = preprocess(df, fs=data.hz, **settings['dcnv_preprocess'])
print(df.shape)
spk = oasis(df, tau=0.16, fs=data.hz, batch_size=settings['extraction']['batch_size'])
print(spk.shape)
re_trs_val = ops.convert_to_numpy(ExpKernel(0.16, hz=data.hz)(ops.pad(spk, ((0, 0), (15, 0),))))
print(re_trs_val.shape)

re_trs_val -= re_trs_val.mean(axis=1, keepdims=True)
scale = np.sqrt(np.sum(np.square(re_trs_val), axis=1, keepdims=True))
re_trs_val /= np.where(scale > 0, scale, 1)

fps_cor = re_fps_val.reshape(-1, h * w) @ gt_fps_val.reshape(-1, h * w).T
trs_cor = re_trs_val @ gt_trs_val.T
dm = (fps_cor < 0) & (trs_cor < 0)
fps_cor[dm] *= -1
trs_cor[dm] *= -1
cor = fps_cor * trs_cor
pair, cor_ind = greedy_matching_r2(cor)
print(nk, (cor_ind > 0.6).sum())

fps_cor_sum = 0
trs_cor_sum = 0
for i, j in pair:
    print(i, j, fps_cor[i, j], trs_cor[i, j])
    fps_cor_sum += fps_cor[i, j]
    trs_cor_sum += trs_cor[i, j]
print(fps_cor_sum / nk, trs_cor_sum / nk)

np.save(run_path / 'suite2p_x.npy', re_fps_val)
np.save(run_path / 'suite2p_t.npy', re_trs_val)
