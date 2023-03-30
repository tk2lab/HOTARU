from hotaru.jax.io.image import load_imgs
from hotaru.jax.filter.stats import calc_stats
from hotaru.jax.filter.laplace import gaussian_laplace_multi


data = load_imgs("Data1/imgs.tif")
avgt, avgx, std0, maxi, stdi, cori = (o.block_until_ready() for o in calc_stats(data))

out = gaussian_laplace_multi(data, [3.0, 4.0, 5.0], avgt, avgx)
print(out)
