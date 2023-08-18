import webbrowser
from pathlib import Path
from importlib.resources import read_text

import hydra

from .cui import (
    cui_main,
    plotter,
)


@hydra.main(version_base=None, config_path="pkg://hotaru.conf", config_name="config")
def _main(cfg):
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


def main():
    cfg_path = Path("hotaru.yaml")
    if not cfg_path.exists():
        cfg = read_text("hotaru.conf", "user_sample.yaml")
        cfg_path.write_text(cfg)

    _main()


if __name__ == "__main__":
    main()
