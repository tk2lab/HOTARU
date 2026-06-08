import json
from os import environ
from pathlib import Path

environ['MKL_NUM_THREADS'] = '1'
environ['OPNEBLAS_NUM_THREADS'] = '1'
environ['VECLIB_MAXIMUM_THREADS'] = '1'

import caiman
import hydra
import numpy as np
import polars as pl
from caiman.source_extraction.cnmf import CNMF
from caiman.source_extraction.cnmf.cnmf import load_CNMF
from caiman.source_extraction.cnmf.params import CNMFParams
from hydra.utils import get_original_cwd
from keras.saving import load_model
from omegaconf import OmegaConf
from tqdm import trange

from ..saving import auto_save_config
from ..saving import make_link
from ..spatial import Footprints  # noqa
from ..temporal import Traces  # noqa
from .evaluation import evaluate


@hydra.main(version_base=None, config_path='.', config_name='caiman')
def main(cfg):
    OmegaConf.resolve(cfg)
    cwd = Path().absolute()
    path = (Path(get_original_cwd()) / cfg.path).relative_to(cwd, walk_up=True)
    data_path = path / cfg.simdata

    imgs_path = data_path / 'imgs'
    fname_mmap = caiman.save_memmap([str(imgs_path / 'imgs.npy')], order='C')
    yr, (h, w), nt = caiman.load_memmap(fname_mmap)
    images = np.reshape(yr.T, (nt, h, w), order='F')
    hz = float(json.loads((imgs_path / 'config.json').read_text()).get('hz'))
    OmegaConf.update(cfg.params.data, 'fr', hz, force_add=True)

    gt = (
        load_model(data_path / 'cell' / 'fps.keras').segs.numpy(),
        load_model(data_path / 'cell' / 'trs.keras').obs.numpy(),
        pl.read_csv(data_path / 'cell' / 'stats.csv'),
    )

    _c, dview, n_processes = caiman.cluster.setup_cluster(**cfg.cluster)
    cluster = {'dview': dview, 'n_processes': n_processes}

    caiman_path = path / 'caiman'
    kwargs = {'imgs_path': imgs_path, 'params': OmegaConf.to_container(cfg.params)}
    for i in trange(cfg.num_refit, ncols=150, desc='caiman fit'):
        step_path = caiman_fit(caiman_path, images, cluster, **kwargs)
        model = load_CNMF(step_path / 'cnmfe.hdf5', **cluster)
        est = model.estimates
        est.evaluate_components(images, model.params, dview=cluster['dview'])
        evaluate_caiman(est, *gt).write_csv(step_path / 'eval.csv')
        make_link(cwd, f'step{i:03d}', step_path)
        kwargs = {'prev_path': step_path}


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
