import jax.numpy as jnp

from hotaru.jax.io.image import load_imgs
from hotaru.jax.filter.stats import calc_stats, Stats
from hotaru.jax.footprint.find import find_peak, PeakVal
from hotaru.jax.footprint.reduce import reduce_peak_block

radius = 2.0, 4.0, 8.0
thr_distance = 1.5
block_size = 100

data = load_imgs("Data1/imgs.tif")

#stats, maxi, stdi, cori = calc_stats(data)
#stats.save("Data1/stats.npz")
stats = Stats.load("Data1/stats.npz")

peakval = find_peak(data, radius, stats, buffer=2 ** 30)
peakval.save("Data1/peakv.npz")
peakval = PeakVal.load("Data1/peakv.npz")
print(peakval.val)
print(peakval.t)
print(peakval.r)

peaks = reduce_peak_block(peakval, thr_distance, block_size)
print(peaks)
