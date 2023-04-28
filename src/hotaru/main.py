import pathlib
import logging

import jax
import hydra
import numpy as np
import pandas as pd
from omegaconf import OmegaConf
from tqdm import tqdm

from .io.image import load_imgs
from .io.mask import (
    load_mask,
    mask_range,
)
from .io.saver import (
    load,
    save,
)
from .jax.filter.stats import (
    Stats,
    calc_stats,
)
from .jax.footprint.find import (
    PeakVal,
    find_peaks_batch,
)
from .jax.footprint.reduce import reduce_peaks_block
from .jax.footprint.make import make_segment_batch
from .jax.train.dynamics import SpikeToCalcium
from .jax.train.temporal import temporal_optimizer
from .jax.proxmodel.regularizer import MaxNormNonNegativeL1


logger = logging.getLogger("HOTARU")


def autosave(path, cfg):
    def decorator(func):
        def wrapped_func():
            filepath = path / cfg.outfile
            if not cfg.force and filepath.exists():
                return load(filepath)
            obj = func(cfg)
            save(filepath, obj)
            return obj
        return wrapped_func
    return decorator


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg):
    if cfg.env.num_devices is None:
        cfg.env.num_devices = jax.local_device_count()
    logger.info(cfg.stage)

    path = pathlib.Path(cfg.data.dir)
    imgs = load_imgs(path / cfg.data.imgs)
    mask = load_mask(path / cfg.data.mask, imgs)
    data = [imgs, mask, None, None, None]

    @autosave(path, cfg.stats)
    def wrap_calc_stats(p):
        stats = calc_stats(imgs, mask, p.batch, tqdm)
        data[2:] = stats.avgx, stats.avgt, stats.std0
        return stats

    @autosave(path, cfg.init.find)
    def wrap_find_peaks(p):
        radius = np.geomspace(p.rmin, p.rmax, p.rnum)
        return find_peaks_batch(*data, radius, p.batch, tqdm)

    @autosave(path, cfg.init.reduce)
    def wrap_reduce_peaks(p):
        return reduce_peaks_block(find, p.rmin, p.rmax, p.thr, p.block_size)

    @autosave(path, cfg.init.segment)
    def wrap_make_segments(p):
        return make_segment_batch(*data, reduce, p.batch, tqdm)

    @autosave(path, cfg.init.temporal)
    def wrap_update_spike(p):
        la = MaxNormNonNegativeL1(p.la)
        lu = MaxNormNonNegativeL1(p.lu)
        optimizer = temporal_optimizer(footprint, data, dynamics, lu, la, p.bt, p.bx, p.batch, tqdm)
        optimizer.set_params(p.lr, p.scale, p.reset)
        print(p.n_iter)
        for i in range(p.n_iter):
            optimizer.step()
        return optimizer.val

    p = cfg.dynamics
    dynamics = SpikeToCalcium.double_exp(p.tau1, p.tau2, p.duration, cfg.data.hz)

    stats = wrap_calc_stats()
    find = wrap_find_peaks()
    reduce = wrap_reduce_peaks()
    footprint = wrap_make_segments()
    spike = wrap_update_spike()
    print(spike)

    """
    footprint = check(path, p := cfg.body.spatial)
    if footprint is None:
        dat, cov, out = prepare(spike, *data, p.bt, p.bx, False, p.batch, tqdm)
        #spike = get_spike(dat, cov, out, **p.dynamics,
        #save(spike_path, spike)

    spike = check(path, p := cfg.body.temporal)
    if spike is None:
        dat, cov, out = prepare(footprint, *data, p.bx, p.bt, True, p.batch, tqdm)
        #spike = get_spike(dat, cov, out, **p.dynamics,
        #save(spike_path, spike)
    """


main()
