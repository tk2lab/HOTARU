import tqdm
import numpy as np

from hotaru.jax.model import Model


model = Model()
model.load_imgs("sample/imgs.tif", "0.pad", 20.0)
if not model.load_stats():
    model.calc_stats()
if not model.load_peaks(2, 16, 11):
    model.calc_peaks()
pbar = tqdm.tqdm(total=model.shape[0])
if not model.load_footprints(1.5):
    model.make_footprints(pbar=pbar)
print(np.count_nonzero(model.footprints, axis=(1, 2)))
