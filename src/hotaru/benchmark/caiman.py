import json
from os import environ
from pathlib import Path

environ['MKL_NUM_THREADS'] = '1'
environ['OPNEBLAS_NUM_THREADS'] = '1'
environ['VECLIB_MAXIMUM_THREADS'] = '1'

import caiman
import hydra
import numpy as np
from caiman.source_extraction.cnmf import CNMF
from caiman.source_extraction.cnmf.cnmf import load_CNMF
from caiman.source_extraction.cnmf.params import CNMFParams
from hydra.utils import get_original_cwd
from omegaconf import OmegaConf
from tqdm import trange

from ..saving import auto_save_config
from ..saving import make_link


@hydra.main(version_base=None, config_path='.', config_name='caiman')
def main(cfg):
    OmegaConf.resolve(cfg)
    cwd = Path().absolute()
    path = (Path(get_original_cwd()) / cfg.path).relative_to(cwd, walk_up=True)

    imgs_path = path / cfg.simdata / 'imgs'
    fname_mmap = caiman.save_memmap([str(imgs_path / 'imgs.npy')], order='C')
    yr, (h, w), nt = caiman.load_memmap(fname_mmap)
    images = np.reshape(yr.T, (nt, h, w), order='F')
    hz = float(json.loads((imgs_path / 'config.json').read_text()).get('hz'))
    OmegaConf.update(cfg.params.data, 'fr', hz, force_add=True)

    _c, dview, n_processes = caiman.cluster.setup_cluster(**cfg.cluster)
    cluster = {'dview': dview, 'n_processes': n_processes}

    caiman_path = path / 'caiman'
    kwargs = {'imgs_path': imgs_path, 'params': OmegaConf.to_container(cfg.params)}
    for i in trange(cfg.num_refit, ncols=150, desc='caiman fit'):
        step_path = caiman_fit(caiman_path, images, cluster, **kwargs)
        make_link(cwd, f'step{i:03d}', step_path)
        kwargs = {'prev_path': step_path}


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


"""
        re_fps_val = cnmf_refit.estimates.A.toarray().reshape(w, h, -1).transpose(2, 1, 0)
        np.save(path / f'fps{i}.npy', re_fps_val)

        re_trs0_val = cnmf_refit.estimates.C.copy()
        np.save(path / f'trs{i}_0.npy', re_trs0_val)

        cnmf_refit.estimates.detrend_df_f(**detrend)
        re_trs1_val = cnmf_refit.estimates.F_dff.copy()
        np.save(path / f'trs{i}_1.npy', re_trs1_val)
"""


if __name__ == '__main__':
    main()
