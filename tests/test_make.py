import hydra
from tqdm import tqdm

from hotaru.jax.model import Model


@hydra.main(version_base=None, config_path=".", config_name="config")
def main(cfg):
    model = Model(cfg)
    model.load_imgs(cfg.data.imgs, cfg.data.mask, cfg.data.hz)
    if not model.load_stats():
        model.calc_stats(tqdm(desc="calc_stats"))
    if not model.load_peakval(2.0, 16.0, 11) or True:
        model.calc_peakval(tqdm(desc="calc_peakval"))
    if not model.load_peaks(2, 9, 2.0) or True:
        model.calc_peaks()
    model.make_footprints(tqdm(desc="make_footprints"))
    '''
    if not model.load_footprints():
        model.make_footprints()
    pbar = tqdm.tqdm()
    model.buffer = 2 ** 20
    print(model.prepare(pbar))
    '''


main()
