import jax.numpy as jnp

from hotaru.jax.io.image import load_imgs
from hotaru.jax.filter.stats import calc_stats, Stats
from hotaru.jax.filter.laplace import gen_gaussian_laplace

radius = 3.0, 4.0, 5.0

data = load_imgs("Data3/imgs.tif")
stats, maxi, stdi, cori = calc_stats(data)
stats.save("Data3/stats.npz")
stats = Stats.load("Data3/stats.npz")

for out in gen_gaussian_laplace(data, radius, stats):
    pass
