import os

import hydra

from .main import HotaruApp


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg):
    app = HotaruApp(cfg)
    app.run_server(**cfg.server)


main()
