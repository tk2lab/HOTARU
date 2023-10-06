import webbrowser
from pathlib import Path
from importlib.resources import read_text

import hydra

from .cui import (
    run,
    test,
)


@hydra.main(version_base=None, config_path="pkg://hotaru.conf", config_name="config")
def _main(cfg):
    match cfg.mode:
        case "test":
            test(cfg)
        case "run":
            run(cfg)
        case _:
            raise ValueError()


def main():
    cfg_path = Path("hotaru/default.yaml")
    if not cfg_path.exists():
        cfg_path.parent.mkdir(parents=True, exist_ok=True)
        cfg = read_text("hotaru.conf", "user_sample.yaml")
        cfg_path.write_text(cfg)

    _main()


if __name__ == "__main__":
    main()
