import json
from pathlib import Path

import hydra
import numpy as np
import polars as pl
from hydra.utils import get_original_cwd
from keras.saving import load_model
from keras.saving import save_model
from omegaconf import OmegaConf
from tqdm import trange

from ..data import CalciumImagingDataWithStats
from ..saving import auto_save_config
from ..saving import make_link
from ..spatial import Footprints
from ..spatial import PeakList
from ..spatial import PeakMap
from ..temporal import Traces
from .evaluation import evaluate


@hydra.main(version_base=None, config_path='.', config_name='hotaru')
def main(cfg):
    OmegaConf.resolve(cfg)
    cwd = Path().absolute()
    path = (Path(get_original_cwd()) / cfg.path).relative_to(cwd, walk_up=True)
    data_path = path / cfg.simdata
    run_path = path / 'hotaru'

    gt = (
        load_model(data_path / 'cell' / 'fps.keras').segs.numpy(),
        load_model(data_path / 'cell' / 'trs.keras').obs.numpy(),
        pl.read_csv(data_path / 'cell' / 'stats.csv'),
    )

    imgs_path = data_path / 'imgs'
    stats_path = hotaru_stats(run_path / 'stats', imgs_path, **cfg.stats)
    make_link(cwd, 'stats', stats_path)

    peakmap_path = hotaru_peakmap(run_path / 'peakmap', stats_path, **cfg.peakmap)
    make_link(cwd, 'peakmap', peakmap_path)
    pre_path = hotaru_peaklist(run_path / 'peaklist', peakmap_path, **cfg.peaklist)
    make_link(cwd, 'peaklist', pre_path)
    for i in range(cfg.num_trial):
        if i == 0:
            fps_path = hotaru_clip(run_path / 'fps', stats_path, pre_path, **cfg.clip)
        else:
            fps_path = hotaru_spatial(run_path / 'trs', stats_path, pre_path, **cfg.spatial)
        make_link(cwd, f'fps{i:03d}', fps_path)
        '''
        pre_path = trs_path = hotaru_temporal(run_path / 'trs', fps_path, **cfg.temporal)
        make_link(cwd, f'trs{i:03d}', trs_path)
        evaluate_hotaru(spatial_path, temporal_path, *gt).write_csv(spatial_path / 'eval.csv')
        '''


@auto_save_config(0.1)
def hotaru_stats(path, imgs_path, batch_size):
    hz = float(json.loads((imgs_path / 'config.json').read_text()).get('hz'))
    data = CalciumImagingDataWithStats(path=imgs_path / 'imgs.npy', hz=hz)
    data.calc(batch_size=batch_size)
    save_model(data, path / 'stats.keras')


@auto_save_config(0.5)
def hotaru_peakmap(path, stats_path, **kwargs):
    for key, cfg in kwargs.items():
        data = load_model(stats_path / 'stats.keras')
        peakmap = PeakMap.generate(data, **cfg, desc=f'peakmap ({key})')
        save_model(peakmap, path / f'{key}.keras')


@auto_save_config(0.3)
def hotaru_peaklist(path, peakmap_path, **kwargs):
    for key, cfg in kwargs.items():
        peakmap = load_model(peakmap_path / f'{key}.keras')
        peaklist = PeakList(cfg.min_distance_ratio)
        peaklist.calc(peakmap, **cfg.fit, desc=f'peaklist ({key})')
        save_model(peaklist, path / f'{key}.keras')


@auto_save_config(0.1)
def hotaru_clip(path, stats_path, peaklist_path, **kwargs):
    for key, cfg in kwargs.items():
        data = load_model(stats_path / 'stats.keras')
        peaklist = load_model(peaklist_path / f'{key}.keras')
        fps = Footprints.from_peaklist(data, peaklist, kind=key, **cfg)
        save_model(fps, path / f'{key}.keras')


@auto_save_config(0.1)
def hotaru_temporal(path, stats_path, fps_path, **kwargs):
    data = load_model(stats_path / 'stats.keras')
    fps = load_model(fps_path / 'fps.keras')

    save_model(fps, path / 'fps.keras')


@auto_save_config(0.1)
def hotaru_spatial(path, stats_path, peaklist_path, **kwargs):
    data = load_model(stats_path / 'stats.keras')
    peaklist = load_model(peaklist_path / 'peaklist.keras')
    fps = Footprints.from_peaklist(data, peaklist, **kwargs)
    save_model(fps, path / 'fps.keras')


def evaluate_caiman(est, fps_gt, trs_gt, stats):
    _, h, w = fps_gt.shape
    fps = est.A.toarray().reshape(w, h, -1).transpose(2, 1, 0)
    trs = est.C.copy()
    df = evaluate(fps, trs, fps_gt, trs_gt)
    df = df.with_columns(rval=est.r_values, snr=est.SNR_comp)
    df = pl.concat((df, stats[df['gt'].to_numpy()]), how='horizontal')
    return df


@auto_save_config(0.3, exclude=('images', 'cluster'))
def caiman_fit(path, images, cluster, **kwargs):
    match kwargs:
        case {'imgs_path': _, 'params': params}:
            cnmf_model = CNMF(params=CNMFParams(params_dict=params), **cluster)
            cnmf_model.fit(images)
        case {'prev_path': prev_path}:
            cnmf_model = load_CNMF(prev_path / 'cnmfe.hdf5', **cluster)
            cnmf_model.refit(images)
        case _:
            raise ValueError(f'{kwargs}')
    cnmf_model.save(str(path / 'cnmfe.hdf5'))


if __name__ == '__main__':
    main()
