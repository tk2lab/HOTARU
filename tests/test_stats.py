import hydra
from tqdm import tqdm

from hotaru.jax.model import Model


@hydra.main(version_base=None, config_path=".", config_name="config")
def main(cfg):
    model = Model(cfg)
    model.calc_stats("sample/imgs.tif", "0.pad", 20.0, tqdm(), True)
    print(model.stats)
    '''
    if not model.load_peaks(3, 9, 2.0):
        model.calc_peaks()
    if not model.load_footprints():
        model.make_footprints()
    pbar = tqdm.tqdm()
    model.buffer = 2 ** 20
    print(model.prepare(pbar))
    '''


main()
