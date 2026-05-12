from dataclasses import dataclass
from logging import getLogger

import matplotlib.pyplot as plt
import numpy as np

from .data import MovieData
from .imgs import to_movie
from .stats import Stats
from .stats import StatsCalculator

logger = getLogger(__name__)


@dataclass
class MovieWithStats(MovieData):
    stats: Stats

    def __init__(self, imgs, mask, hz, **kwargs):
        super().__init__(imgs, mask, hz)
        self.stats = StatsCalculator().get_stats(self, **kwargs)

    def original_movie(self, outfile, cmap='Greens', **kwargs):
        imgs = self.imgs
        vmin = self.stats.min0
        scale = self.stats.max0 - self.stats.min0
        cmap = plt.get_cmap(cmap)
        def gen():
            for t in range(self.num_frames):
                val = (imgs[t] - vmin) / scale
                yield (255 * cmap(val)).astype(np.uint8)
        return to_movie(outfile, gen(), imgs.shape, self.hz, **kwargs)

    def normalized_movie(self, outfile, cmap='Greens', **kwargs):
        imgs = self.imgs
        stats = self.stats
        min0 = stats.imin.min()
        max0 = stats.imax.max()
        cmap = plt.get_cmap(cmap)
        def gen():
            for t in range(self.num_frames):
                val = (imgs[t] - stats.avgt[t] - stats.avgx) / stats.std0
                val = (val - min0) / (max0 - min0)
                yield (255 * cmap(val)).astype(np.uint8)
        return to_movie(outfile, gen(), imgs.shape, self.hz, **kwargs)
