import hydra

from .main import MainApp


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg):
    app = MainApp(cfg)
    app.run_server(debug=cfg.app.debug, port=cfg.app.port)


main()
