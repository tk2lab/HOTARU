import math
from collections import namedtuple

import tensorflow as tf

from ..filter.stats import calc_mean_std
from ..io.image import ImageStack
from ..io.mask import get_mask
from ..util.dataset import (
    masked,
    normalized,
    normalized_masked_image,
)
from .dynamics import (
    CalciumToSpike,
    SpikeToCalcium,
)

Penalty = namedtuple("Penalty", ["la", "lu", "lx", "lt", "bx", "bt"])


class HotaruModelBase(tf.keras.layers.Layer):
    """Variable"""

    def __init__(self, imgs, mask, hz, tauduration=1.0, taursize=3, name="Hotaru"):
        super().__init__(name=name)

        tausize = int(math.ceil(tauduration * hz))

        imgs = ImageStack(imgs).dataset()
        mask = get_mask(mask, imgs)
        data = masked(imgs, mask)

        self._raw_imgs = imgs
        self._raw_data = data
        self._mask = mask
        self._hz = hz

        self._spike_to_calcium = SpikeToCalcium(tausize)
        self._calcium_to_spike = CalciumToSpike(tausize, taursize)
        self._penalty = Penalty(
            *[
                self.add_weight(name, (), tf.float32, trainable=False)
                for name in Penalty._fields
            ]
        )

    def update_penalty(self, **kwargs):
        nx, nt = self.raw_data.shape
        nm = nx * nt + nx + nt
        for name, val in kwargs.items():
            if name not in ["bx", "bt"]:
                val /= nm
            getattr(self._penalty, name).assign(val)

    @property
    def penalty(self):
        return Penalty(*[v.numpy() for v in self._penalty])

    @property
    def stats(self):
        if not hasattr(self, "_stats"):
            stats = calc_mean_std(self.raw_data, self.batch)
            self._stats = stats
        return self._stats

    @property
    def raw_imgs(self):
        return self._raw_imgs

    @property
    def mask(self):
        return self._mask

    @property
    def hz(self):
        return self._hz

    @property
    def raw_data(self):
        return self._raw_data

    @property
    def data(self):
        if not hasattr(self, "_data"):
            self._data = normalized(self.raw_data, *self.stats)
        return self._data

    @property
    def imgs(self):
        if not hasattr(self, "_imgs"):
            self._imgs = normalized_masked_image(
                self.raw_imgs,
                self.mask,
                *self.stats,
            )
        return self._imgs
