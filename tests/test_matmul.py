import hydra
import tqdm

from hotaru.jax.model import Model


@hydra.main(version_base=None, config_path=".", config_name="config")
def main(cfg):
    model = Model(cfg)
    model.load_imgs("sample/imgs.tif", "0.pad", 20.0)
    if not model.load_stats():
        model.calc_stats()
    if not model.load_peakval(2.0, 16.0, 11):
        model.calc_peakval()
    if not model.load_peaks(3, 9, 2.0):
        model.calc_peaks()
    if not model.load_footprints():
        model.make_footprints()
    pbar = tqdm.tqdm()
    model.buffer = 2 ** 20
    print(model.prepare(pbar))


main()
