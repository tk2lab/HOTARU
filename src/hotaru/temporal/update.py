import numpy as np
from keras import ops
from keras.callbacks import EarlyStopping
from keras.callbacks import TerminateOnNaN

from ..config import TotalProperty
from ..data import MovieWithStats
from ..losses import Minimize
from ..models import ProxModel
from ..saving import Data
from ..saving import DataDict
from ..saving import cached_getter
from ..spatial import Footprints
from ..typing import Array
from ..typing import Tensor
from .prepare import CorrCalculator


class TemporalComponents(Data):
    core: Array
    obs: Array


class TemporalUpdater(ProxModel):
    def __init__(self, props: TotalProperty, **kwargs):
        super().__init__(**kwargs)
        self.props = props

    def compile(self, **kwargs) -> None:
        kwargs.setdefault('loss', Minimize())
        super().compile(**kwargs)

    def prepare(self, data: MovieWithStats, *footprints: Footprints, **kwargs) -> None:
        nt, h, w = data.shape
        nx = h * w
        nks = tuple(fp.shape[0] for fp in footprints)

        if not self.built:
            self.build((nt, nx, nks))

        self.scale = float(nt) * float(nx)

        for k, nk in enumerate(nks):
            ak = footprints[k].data.reshape(nk, nx)
            ak /= np.sqrt(np.square(ak).mean(axis=1, keepdims=True))
            fac = self.props.component_properties[k].temporal_factor(ak)
            self.spatial_comp[k].assign(ak)
            self.activity[k].regularizer.fac.assign(fac)

        for i, ai in enumerate(self.spatial_comp):
            self.spatial_mean[i].assign(ops.mean(ai, axis=1))
            for j, aj in enumerate(self.spatial_comp[: i + 1]):
                sqrd = self.spatial_sqrd[i][j]
                sqrd.assign(ai.value @ aj.value.T / nx)

        prepare = CorrCalculator(self.props, self.spatial_comp, self.spatial_corr)
        prepare.calc_corr(data, **kwargs)

    @property
    def num(self) -> int:
        return len(self.activity)

    @cached_getter(DataDict)
    def update(self, *, reset: bool = True, **kwargs) -> list[TemporalComponents]:
        if reset:
            self.reset()

        kwargs.setdefault('steps_per_epoch', 1)

        callbacks = kwargs.setdefault('callbacks', [])
        callbacks.append(TerminateOnNaN())
        if (early_stopping := kwargs.pop('early_stopping')) is not None:
            callbacks.append(EarlyStopping(monitor='total_loss', mode='min', **early_stopping))

        scale = float(self.scale)
        kwargs.setdefault('scale', {'loss': scale, 'penalty': scale, 'total_loss': scale})

        _history = self.fit(**kwargs)
        self.update_observation()

        return [
            TemporalComponents(u, v) for u, v in zip(self.activity, self.observation, strict=True)
        ]

    def build(self, input_shape):
        nt, nx, nks = input_shape

        self.spatial_comp = [self.add_weight((nk, nx), trainable=False) for nk in nks]
        self.spatial_mean = [self.add_weight((nk,), trainable=False) for nk in nks]
        self.spatial_corr = [self.add_weight((nk, nt), trainable=False) for nk in nks]
        self.spatial_sqrd = []
        for k, nk in enumerate(nks):
            self.spatial_sqrd.append([])
            for nl in nks[: k + 1]:
                self.spatial_sqrd[-1].append(self.add_weight((nk, nl), trainable=False))

        self.activity = []
        self.observation = []
        for k, nk in enumerate(nks):
            p = self.props.component_properties[k]
            ntau = nt + p.temporal_kernel.size - 1
            regularizer = p.temporal_regularizer
            self.activity.append(self.add_weight((nk, ntau), regularizer=regularizer))
            self.observation.append(self.add_weight((nk, nt), trainable=False))
        self.temporal_baseline = self.add_weight((nt,))
        self.spatial_baseline = self.add_weight((nx,))

        super().build(input_shape)

    def reset(self):
        def var_reset(x):
            x.assign(ops.zeros(x.shape, x.dtype))

        for ai in self.activity:
            var_reset(ai)
        var_reset(self.temporal_baseline)
        var_reset(self.spatial_baseline)

    def update_observation(self):
        def conv(d, k):
            return ops.conv(d[:, :, None], k[:, None, None], 1, 'valid', 'channels_last')[..., 0]

        for i, p in enumerate(self.props.component_properties):
            vi = conv(self.activity[i], p.temporal_kernel)
            self.observation[i].assign(vi)

    def call(self, _dummy: Tensor) -> Tensor:
        a1 = self.spatial_comp
        am = self.spatial_mean
        fa = self.spatial_corr
        a2 = self.spatial_sqrd

        self.update_observation()
        v = [v.value for v in self.observation]
        vm = [ops.mean(vi, axis=1) for vi in v]

        bt = self.temporal_baseline
        bt -= ops.mean(bt)
        bx = self.spatial_baseline
        bx -= ops.mean(bx)
        b0 = ops.sum([ops.sum(ami * vmi) for ami, vmi in zip(am, vm, strict=True)])

        nt, nx = bt.size, bx.size
        loss = ops.mean(ops.square(bt)) + ops.mean(ops.square(bx)) - ops.square(b0)
        for i, a2i in enumerate(a2):
            for j, a2ij in enumerate(a2i):
                loss += ops.sum(a2ij * (v[i] @ v[j].T)) / nt
        for fai, a1i, ami, vi in zip(fa, a1, am, v, strict=True):
            loss -= 2 * ops.sum(fai * vi) / nt
            loss -= 2 * ops.sum(bt * (ami @ vi)) / nt
            loss -= 2 * ops.sum((a1i @ bx)[:, None] * vi) / nt / nx
        return self.scale * loss
