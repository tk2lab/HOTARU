import sys
from importlib import import_module
from multiprocessing import Process

from .common import set_env, finish


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


def run(cfg):
    for stage in range(cfg.max_train_step + 1):
        if stage == 0:
            call("normalize", cfg)
            call("init", cfg)
        else:
            call("spatial", cfg, stage)
        call("temporal", cfg, stage)

        if finish(cfg, stage):
            break
