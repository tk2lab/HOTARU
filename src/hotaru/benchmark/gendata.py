from pathlib import Path

import hydra
from hydra.utils import get_original_cwd

from ..random import Generator
from ..saving import make_link
from .simulation import make_cell
from .simulation import make_dendrite
from .simulation import make_neuropil
from .simulation import make_sim


@hydra.main(version_base=None, config_path='.', config_name='gendata')
def main(cfg):
    cwd = Path().absolute()
    path = (Path(get_original_cwd()) / cfg.path).relative_to(cwd, walk_up=True)
    rng = Generator(cfg.seed)

    cell_path = make_cell(
        path / 'cell',
        **cfg.shape,
        **cfg.cell,
        seed=rng.gen_seed(),
    )
    make_link(cwd, 'cell', cell_path)

    dend_path = make_dendrite(
        path / 'dend',
        cell_path,
        **cfg.shape,
        **cfg.dendrite,
        seed=rng.gen_seed(),
    )
    make_link(cwd, 'dend', dend_path)

    npil_path = make_neuropil(
        path / 'npil',
        **cfg.shape,
        **cfg.neuropil,
        seed=rng.gen_seed(),
    )
    make_link(cwd, 'npil', npil_path)

    imgs_path = make_sim(
        path / 'imgs',
        cell_path,
        dend_path,
        npil_path,
        cfg.shape.hz,
        **cfg.output,
        seed=rng.gen_seed(),
    )
    make_link(cwd, 'imgs', imgs_path)


if __name__ == '__main__':
    main()
