import logging
import webbrowser

import hydra

from .cui import cui_main


@hydra.main(version_base=None, config_path="pkg://hotaru.conf", config_name="config")
def main(cfg):
    if cfg.gui.use:
        pass
        # app = gui(cfg)
        # app.run_server(**cfg.gui.server)
        if cfg.gui.open_brower:
            webbrowser.open_new(f"http://{cfg.gui.userver.host}:{cfg.gpu.server.port}")
    else:
        cui_main(cfg)
