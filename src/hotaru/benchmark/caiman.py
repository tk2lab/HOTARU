import json
from pathlib import Path

import caiman
import hydra
import numpy as np
from caiman.source_extraction.cnmf import CNMF
from caiman.source_extraction.cnmf.params import CNMFParams
from hydra.utils import get_original_cwd

from ..simulation import auto_save_config
from ..simulation import make_link


@hydra.main(version_base=None, config_path='.', config_name='caiman')
def main(cfg):
    cwd = Path().absolute()
    path = (Path(get_original_cwd()) / cfg.path).relative_to(cwd, walk_up=True)
    imgs_path = path / cfg.simdata / 'imgs'
    caiman_path = run_caiman(
        path / 'caiman',
        imgs_path,
        cfg.use_cluster,
        cfg.num_refit,
        cfg.params,
        cfg.detrend,
    )
    make_link(cwd, 'results', caiman_path)


@auto_save_config
def run_caiman(path, imgs_path, use_cluster, num_refit, params, detrend):
    n_processes = 1
    cluster = None
    if use_cluster:
        _c, cluster, n_processes = caiman.cluster.setup_cluster(
            backend='multiprocessing',
            n_processes=None,
            single_thread=False,
        )

    imgs_config = json.loads((imgs_path / 'config.json').read_text(encoding='utf-8'))
    fname_mmap = caiman.save_memmap([str(imgs_path / 'imgs.npy')], order='C')
    yr, (h, w), nt = caiman.load_memmap(fname_mmap)
    images = np.reshape(yr.T, (nt, h, w), order='F')

    params = CNMFParams(fr=float(imgs_config.get('hz')), **params)
    cnmf_model = CNMF(n_processes=n_processes, dview=cluster, params=params)
    cnmf_model.fit(images)

    for i in range(num_refit):
        cnmf_refit = cnmf_model.refit(images, dview=cluster)

        re_fps_val = cnmf_refit.estimates.A.toarray().reshape(w, h, -1).transpose(2, 1, 0)
        np.save(path / f'fps{i}.npy', re_fps_val)

        re_trs0_val = cnmf_refit.estimates.C.copy()
        np.save(path / f'trs{i}_0.npy', re_trs0_val)

        cnmf_refit.estimates.detrend_df_f(**detrend)
        re_trs1_val = cnmf_refit.estimates.F_dff.copy()
        np.save(path / f'trs{i}_1.npy', re_trs1_val)


if __name__ == '__main__':
    main()
