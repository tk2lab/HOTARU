import numpy as np
import tensorflow as tf

from .base import HotaruModelBase
from .driver import SpatialModel
from .driver import TemporalModel
from .input import DynamicL2InputLayer as L2
from .input import DynamicMaxNormNonNegativeL1InputLayer as ML1


class HotaruModel(HotaruModelBase):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._spatial_compile_args = {}
        self._temporal_compile_args = {}
        self._spatial_fit_args = {}
        self._temporal_fit_args = {}
        self._built = False
        self._compiled = False

    def set_double_exp(self, tau1, tau2):
        if tau1 > tau2:
            tau1, tau2 = tau2, tau1
        tau1 *= self.hz
        tau2 *= self.hz
        r = tau1 / tau2
        d = tau1 - tau2
        scale = np.power(r, -tau2 / d) - np.power(r, -tau1 / d)
        t = np.arange(1, self._spike_to_calcium.kernel.size + 1)
        e1 = np.exp(-t / tau1)
        e2 = np.exp(-t / tau2)
        kernel = (e1 - e2) / scale
        self._spike_to_calcium.kernel = kernel
        kernel = np.array([1.0, -e1[0] - e2[0], e1[0] * e2[0]]) / kernel[0]
        self._calcium_to_spike.kernel = kernel

    def _split_args(self, **kwargs):
        temporal = {}
        spatial = {}
        for k, v in kwargs.items():
            if k.startswith("temporal_"):
                temporal[k[9:]] = v
            elif k.startswith("spatial_"):
                spatial[k[8:]] = v
            else:
                temporal[k] = v
                spatial[k] = v
        return temporal, spatial

    def update_compile_args(self, **kwargs):
        temporal, spatial = self._split_args(**kwargs)
        self._temporal_compile_args.update(temporal)
        self._spatial_compile_args.update(spatial)

    def update_fit_args(self, **kwargs):
        temporal, spatial = self._split_args(**kwargs)
        self._temporal_fit_args.update(temporal)
        self._spatial_fit_args.update(spatial)

    def build(self, nk=None):
        if self._built:
            return
        if nk is None:
            nk = self.info.shape[0]
        nt, nx = self.data.shape
        nu = nt + self._spike_to_calcium.kernel.size - 1
        self.max_nk = nk
        self.footprint = ML1(nk, nx, self._penalty.la, name="footprint")
        self.spike = ML1(nk, nu, self._penalty.lu, name="spike")
        self.localx = ML1(nk, nx, self._penalty.lx, name="localx")
        self.localt = L2(nk, nt, self._penalty.lt, name="localt")
        self.spatial_model = SpatialModel(self)
        self.temporal_model = TemporalModel(self)
        self._built = True

    def compile(self):
        if self._compiled:
            return 
        self.temporal_model.compile(**self._temporal_compile_args)
        self.spatial_model.compile(**self._spatial_compile_args)
        self._compiled = True

    def fit_spatial(self):
        self.compile()
        spike = self.spike.val_tensor()
        self.spike.val = spike / tf.math.reduce_max(
            spike, axis=1, keepdims=True
        )
        localt = self.localt.val_tensor()
        self.localt.val = localt / tf.math.reduce_max(
            localt, axis=1, keepdims=True
        )
        self.spatial_model.fit(**self._spatial_fit_args)

    def fit_temporal(self):
        self.compile()
        self.temporal_model.fit(**self._temporal_fit_args)
