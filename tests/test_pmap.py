from hotaru.jax.io.image import load_imgs
from hotaru.jax.filter.stats import calc_stats


data = load_imgs("Data1/imgs.tif")
print(calc_stats(data))
