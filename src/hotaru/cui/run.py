import sys
from datetime import datetime
from importlib import import_module
from multiprocessing import Process
from pathlib import Path

import numpy as np

from ..plot import seg_max_fig
from ..plot import spike_image
from ..plot import to_csv
from ..plot import to_multipage_tif
from .common import finish
from .common import print_stats
from .common import set_env


def call(name, *args, **kwargs):
    def wrap(cfg, *args, **kwargs):
        set_env(cfg)
        return target(cfg, *args, **kwargs)

    target = getattr(import_module(f'hotaru.cui.{name}'), name)
    p = Process(target=wrap, args=args, kwargs=kwargs)
    p.start()
    p.join()
    if p.exitcode != 0:
        sys.exit()


def run(cfg):
    now = datetime.now().isoformat()

    for stage in range(cfg.max_train_step + 1):
        if stage == 0:
            call('normalize', cfg)
            call('init', cfg)
        else:
            call('spatial', cfg, stage)
        call('temporal', cfg, stage)

        print_stats(cfg, stage)
        if finish(cfg, stage):
            break

    path = Path(cfg.outputs.outputs.dir) / now
    path.mkdir(parents=True, exist_ok=True)
    to_csv(cfg, stage, path)
    to_multipage_tif(cfg, stage, path)
    seg_max_fig(cfg, stage).write_image(path / 'run_footprints.pdf')
    spike_image(cfg, stage, tsel=np.arange(4096))[0].save(path / 'run_spike.pdf')
