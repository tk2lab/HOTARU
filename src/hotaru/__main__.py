import webbrowser

import hydra

from .cui import (
    cui_main,
    plotter,
)


@hydra.main(version_base=None, config_path="pkg://hotaru.conf", config_name="config")
def main(cfg):
    match cfg.mode:
        case "run":
            cui_main(cfg)
        case "plot":
            plotter(cfg)
        case "gui":
            pass
            if cfg.gui.open_brower:
                webbrowser.open_new(f"http://{cfg.gui.userver.host}:{cfg.gpu.server.port}")
        case _:
            raise ValueError()
