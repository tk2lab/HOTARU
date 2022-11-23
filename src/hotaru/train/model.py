from collections import namedtuple
from os import fspath
from os import PathLike

import numpy as np
import pandas as pd
import tensorflow as tf

from ..filter.stats import calc_mean_std
from ..io.image import ImageStack
from ..io.mask import get_mask
from ..proxmodel.optimizer import ProxOptimizer
from ..util.dataset import masked
from ..util.dataset import normalized
from ..util.dataset import normalized_masked_image
from .driver import SpatialModel
from .driver import TemporalModel
from .dynamics import CalciumToSpike
from .dynamics import SpikeToCalcium
from .input import DynamicL2InputLayer as L2
from .input import DynamicMaxNormNonNegativeL1InputLayer as ML1

Penalty = namedtuple("Penalty", ["la", "lu", "lx", "lt", "bx", "bt"])


class HotaruModel(tf.keras.layers.Layer):
    """Variable"""

    def __init__(self, imgs, mask, hz, tausize, name="Hotaru"):
        super().__init__(name=name)

        imgs = ImageStack(imgs).dataset()
        mask = get_mask(mask, imgs)
        data = masked(imgs, mask)

        self.raw_imgs = imgs
        self.raw_data = data
        self.mask = mask
        self.hz = hz

        self.spike_to_calcium = SpikeToCalcium(tausize, name="from_spike")
        self.local_to_calcium = SpikeToCalcium(tausize, name="from_local")
        self.calcium_to_spike = CalciumToSpike(tausize, 3, name="to_spike")

        self._penalty = Penalty(
            *[
                self.add_weight(name, (), tf.float32, trainable=False)
                for name in Penalty._fields
            ]
        )

        self._built = False
        self._saved = None

    def set_stats(self, path: PathLike, batch, force=False):
        if force or not path.exists():
            stats = calc_mean_std(self.raw_data, batch)
            np.savez(path, **stats._asdict())
        else:
            stats = np.load(path)
            stats = stats["avgx"], stats["avgt"], stats["std"]
        self.stats = stats
        self.data = normalized(self.raw_data, *stats)
        self.imgs = normalized_masked_image(self.raw_imgs, self.mask, *stats)

    def set_double_exp(self, tau1, tau2):
        if tau1 > tau2:
            tau1, tau2 = tau2, tau1

        tau1 *= self.hz
        tau2 *= self.hz
        r = tau1 / tau2
        d = tau1 - tau2
        scale = np.power(r, -tau2 / d) - np.power(r, -tau1 / d)
        t = np.arange(1, self.spike_to_calcium.kernel.size + 1)
        e1 = np.exp(-t / tau1)
        e2 = np.exp(-t / tau2)

        kernel = (e1 - e2) / scale
        self.spike_to_calcium.kernel = kernel

        kernel = np.array([1.0, -e1[0] - e2[0], e1[0] * e2[0]]) / kernel[0]
        self.calcium_to_spike.kernel = kernel

    def set_penalty(self, **kwargs):
        nx, nt = self.raw_data.shape
        nm = nx * nt + nx + nt
        for name, val in kwargs.items():
            if name not in ["bx", "bt"]:
                val /= nm
            getattr(self._penalty, name).assign(val)

    def build(self, nk=None):
        if self._built:
            return
        if nk is None:
            nk = self.info.shape[0]
        nt, nx = self.data.shape
        nu = nt + self.spike_to_calcium.kernel.size - 1
        self.max_nk = nk
        self.footprint = ML1(nk, nx, self._penalty.la, name="footprint")
        self.spike = ML1(nk, nu, self._penalty.lu, name="spike")
        self.localx = ML1(nk, nx, self._penalty.lx, name="localx")
        self.localt = L2(nk, nt, self._penalty.lt, name="localt")
        self.spatial_model = SpatialModel(self)
        self.temporal_model = TemporalModel(self)
        self._built = True

    def compile(self, temporal_args=None, spatial_args=None):
        if temporal_args is None:
            temporal_args = {}
        if spatial_args is None:
            spatial_args = {}
        self.temporal_model.compile(**temporal_args)
        self.spatial_model.compile(**spatial_args)

    @property
    def penalty(self):
        return Penalty(*[v.numpy() for v in self._penalty])

    def exists(self, path: PathLike):
        return (path / "saved_model.pb").exists()

    def save(self, path: PathLike):
        module = tf.Module()
        module.footprint = tf.Variable(self.footprint.val_tensor())
        module.spike = tf.Variable(self.spike.val_tensor())
        module.localx = tf.Variable(self.localx.val_tensor())
        module.localt = tf.Variable(self.localt.val_tensor())
        tf.saved_model.save(module, fspath(path))
        self.info.to_csv(path / "info.csv")
        self._saved = path

    def load(self, path):
        if self._saved == path:
            return
        self.info = pd.read_csv(path / "info.csv", index_col=0)
        self.build_and_compile(nk=self.info.shape[0])
        module = tf.saved_model.load(fspath(path))
        self.footprint.val = module.footprint
        self.spike.val = module.spike
        self.localx.val = module.localx
        self.localt.val = module.localt
        self._saved = path
