import sys
from datetime import datetime
from importlib import import_module
from multiprocessing import Process
from pathlib import Path

from ..plot import peak_stats_fig
from ..plot import seg_max_fig
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


def test(cfg):
    now = datetime.now().isoformat()

    call('normalize', cfg)
    call('init', cfg)

    path = Path(cfg.outputs.figs.dir) / now
    path.mkdir(parents=True, exist_ok=True)
    peak_stats_fig(cfg, 0).write_image(path / 'test_stats.pdf')
    seg_max_fig(cfg, 0).write_image(path / 'test_footprints.pdf')
