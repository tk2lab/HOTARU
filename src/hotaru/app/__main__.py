import webbrowser

import hydra

from .main import HotaruApp


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg):
    app = HotaruApp(cfg)
    app.run_server(**cfg.server)
    if cfg.server.open_brower:
        webbrowser.open_new(f"http://{cfg.server.host}:{cfg.server.port}")


main()
