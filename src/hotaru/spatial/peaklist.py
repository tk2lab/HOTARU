from logging import getLogger

import numpy as np

from ..models import Layer
from ..saving import Config
from .calc_peaklist import calc_peaklist
from .peakmap import PeakMap

logger = getLogger(__name__)


class PeakList(Layer):
    def __init__(self, min_distance_ratio: float, **kwargs):
        super().__init__(**kwargs)
        self.min_distance_ratio = min_distance_ratio

    def get_config(self) -> Config:
        return {'min_distance_ratio': self.min_distance_ratio, **super().get_config()}

    @property
    def size(self) -> int:
        return self.tlist.shape[0]

    def calc(self, peakmap: PeakMap, block_size: int, desc: str = 'reduce') -> None:
        radius = peakmap.radius
        tmap = peakmap.tmap.numpy()
        rmap = peakmap.rmap.numpy()
        gmap = peakmap.gmap.numpy()
        vals = calc_peaklist(radius, tmap, rmap, gmap, self.min_distance_ratio, block_size, desc)

        self.build((vals[0].size,))
        keys = 'tlist', 'rlist', 'ylist', 'xlist', 'glist'
        for key, val in zip(keys, vals, strict=True):
            getattr(self, key).assign(val)

        r, n = np.unique(self.rlist, return_counts=True)
        logger.info('find: %s', list(zip(r.tolist(), n.tolist(), strict=True)))

    def build(self, input_shape):
        (size,) = input_shape
        for key in ('tlist', 'ylist', 'xlist'):
            setattr(self, key, self.add_weight((size,), dtype='int32', name=key))
        for key in ('rlist', 'glist'):
            setattr(self, key, self.add_weight((size,), name=key))
        super().build(input_shape)
