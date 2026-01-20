import hydra

from .cui import run
from .cui import test


@hydra.main(version_base=None, config_path=None, config_name='config')
def main(cfg):
    match cfg.mode:
        case 'test':
            test(cfg)
        case 'run':
            run(cfg)
        case _:
            raise ValueError()


if __name__ == '__main__':
    main()
