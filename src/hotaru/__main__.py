import os
import webbrowser

import hydra

from .app.cui import HotaruCUI
from .app.gui import HotaruGUI


@hydra.main(version_base=None, config_path="pkg://hotaru.conf", config_name="config")
def main(cfg):
    print(cfg)
    print(os.getcwd())
    print(hydra.utils.get_original_cwd())
    print(hydra.utils.to_absolute_path("conf"))
    print(hydra.utils.to_absolute_path("/etc"))
    match cfg.interface:
        case "GUI":
            app = HotaruGUI(cfg)
            app.run_server(**cfg.server)
            if cfg.server.open_brower:
                webbrowser.open_new(f"http://{cfg.server.host}:{cfg.server.port}")
        case "CUI":
            HotaruCUI(cfg)


if __name__ == "__main__":
    main()
