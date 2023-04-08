import hydra
from tqdm import tqdm

from hotaru.jax.model import Model


@hydra.main(version_base=None, config_path=".", config_name="config")
def main(cfg):
    model = Model(cfg)
    model.load_imgs("sample/imgs.tif", "0.pad", 20.0)
    model.calc_stats(tqdm())
    print(model.stats)


main()
