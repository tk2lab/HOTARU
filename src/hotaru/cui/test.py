import sys
from pathlib import Path
from importlib import import_module
from multiprocessing import Process

from ..plot import (
    peak_stats_fig,
    seg_max_fig,
)
from .common import set_env


def call(name, *args, **kwargs):
    def wrap(cfg, *args, **kwargs):
        set_env(cfg)
        return target(cfg, *args, **kwargs)

    target = getattr(import_module(f"hotaru.cui.{name}"), name)
    p = Process(target=wrap, args=args, kwargs=kwargs)
    p.start()
    p.join()
    if p.exitcode != 0:
        sys.exit()


def test(cfg):
    call("normalize", cfg)
    call("init", cfg)

    path = Path(cfg.outputs.figs.dir)
    path.mkdir(parents=True, exist_ok=True)
    peak_stats_fig(cfg, 0).write_image(path / "test_stats.pdf")
    seg_max_fig(cfg, 0).write_image(path / "test_footprints.pdf")
