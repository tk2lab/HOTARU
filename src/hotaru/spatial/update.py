import numpy as np
from keras import ops
from keras.callbacks import CSVLogger
from keras.callbacks import EarlyStopping
from keras.callbacks import History
from keras.callbacks import TerminateOnNaN

from ..config import TotalProperty
from ..data import MovieWithStats
from ..losses import Minimize
from ..models import ProxModel
from ..saving import Data
from ..saving import cached_getter
from ..temporal import Traces
from ..typing import Array
from ..typing import Tensor
from .corr import CorrCalculator
from .footprint import Footprints


class CorrData(Data):
    data: Array


class SpatialUpdater(ProxModel):
    def __init__(self, props: TotalProperty, **kwargs):
        super().__init__(**kwargs)
        self.props = props

    def prepare(self, data: MovieWithStats, *traces: Traces, **kwargs) -> None:
        nt, h, w = data.shape
        nx = h * w

        if not self.built:
            nks = tuple(fp.shape[0] for fp in traces)
            self.build((nt, nx, nks))

        for k in range(self.num):
            vk = traces[k].obs
            vk /= np.sqrt(np.square(vk).mean(axis=1, keepdims=True))
            fac = self.props.component_properties[k].spatial_factor(vk)
            self.comp[k].assign(vk)
            self.mean[k].assign(np.mean(vk, axis=1))
            self.activity[k].regularizer.fac.assign(fac)

        for i, vi in enumerate(self.comp):
            for j, vj in enumerate(self.comp[: i + 1]):
                self.sqrd[i][j].assign(vi.value @ vj.value.T / nx)

        corr = self.prepare_corr(data, **kwargs)
        for k in range(self.num):
            self.corr[k].assign(corr[k].data)

    @cached_getter(CorrData)
    def prepare_corr(self, data: MovieWithStats, **kwargs) -> list[CorrData]:
        def calc_corr(ak):
            corr = self.corr_calculator.calc(data, ak.value, **kwargs)
            return CorrData(ops.convert_to_numpy(corr))

        return [calc_corr(ak) for ak in self.spatial_comp]

    def compile(self, **kwargs) -> None:
        kwargs.setdefault('loss', Minimize())
        super().compile(**kwargs)

    @cached_getter(Footprints)
    def update(self, *, reset: bool = True, **kwargs) -> list[Footprints]:
        _history = self.fit(reset=reset, **kwargs)
        return [Footprints(a) for a in self.observations()]

    def fit(self, *_args, **kwargs) -> History:
        if kwargs.pop('reset', True):
            self.reset()

        scale_map = {'loss': self.scale, 'penalty': self.scale, 'total_loss': self.scale}
        kwargs.setdefault('scale', scale_map)
        kwargs.setdefault('steps_per_epoch', 1)

        callbacks = kwargs.setdefault('callbacks', [])
        callbacks.append(TerminateOnNaN())
        if (early_stopping := kwargs.pop('early_stopping', None)) is not None:
            callbacks.append(EarlyStopping(monitor='total_loss', mode='min', **early_stopping))
        if (csv_path := kwargs.pop('csv_path', None)) is not None:
            callbacks.append(CSVLogger(csv_path))

        for k, pk in enumerate(self.props.component_properties):
            fac = pk.spatial_factor(self.comp[k])
            pk.temporal_regularizer.fac.assign(fac)
        return super().fit(**kwargs)

    def build(self, input_shape):
        nt, nx, nks = input_shape
        self.num = len(nks)
        self.scale = float(nt) * float(nx)

        self.corr_calculator = CorrCalculator(max(nks))
        self.comp = []
        self.mean = []
        self.sqrd = []
        self.corr = []
        param_args = {'initializer': 'zeros', 'trainable': False}
        for i, ni in enumerate(nks):
            self.comp.append(self.add_weight((ni, nt), **param_args, name=f'comp{i}'))
            self.mean.append(self.add_weight((ni,), **param_args, name=f'mean{i}'))
            self.corr.append(self.add_weight((ni, nx), **param_args, name=f'corr{i}'))
            self.sqrd.append([])
            for j, nj in enumerate(nks[: i + 1]):
                w = self.add_weight((ni, nj), **param_args, name=f'sq{i}{j}')
                self.spatial_sqrd[-1].append(w)

        self.regularizer = []
        self.activity = []
        for k, (pk, nk) in enumerate(zip(self.props.component_properties, nks, strict=True)):
            self.regularizer.append(r := pk.temporal_regularizer)
            w = self.add_weight((nk, nx), regularizer=r, name=f'act{k}')
            self.activity.append(w)

        self.regularizer.append(r := self.props.temporal_baseline_regularizer)
        self.temporal_baseline = self.add_weight((nt,), regularizer=r)
        self.regularizer.append(r := self.props.spatial_baseline_regularizer)
        self.spatial_baseline = self.add_weight((nx,), regularizer=r)

        super().build(input_shape)

    def reset(self):
        for x in (*self.activity, self.temporal_baseline, self.spatial_baseline):
            x.assign(ops.zeros(x.shape, x.dtype))

    def observations(self):
        def conv(d, k):
            d, k = d[:, :, :, None], k[:, :, None, None]
            return ops.conv(d, k, 1, 'valid', 'channels_last')[..., 0]

        return [
            conv(self.activity[i].value, p.temporal_kernel)
            for i, p in enumerate(self.props.component_properties)
        ]

    def call(self, _dummy: Tensor) -> Tensor:
        # a1 = self.spatial_comp
        vm = self.mean
        fv = self.corr
        v2 = self.sqrd
        _nk, nx = fv[0].shape

        a = self.observations()
        am = [ops.mean(ai, axis=1) for ai in a]

        # bt = self.temporal_baseline.value
        # bt -= ops.mean(bt)
        # bx = self.spatial_baseline
        # bx -= ops.mean(bx)
        b0 = ops.sum([ops.sum(ami.value * vmi) for ami, vmi in zip(am, vm, strict=True)])

        # nt, nx = bt.size, bx.size
        # loss = ops.mean(ops.square(bt)) + ops.mean(ops.square(bx)) - ops.square(b0)
        loss = ops.square(b0)
        for i, v2i in enumerate(v2):
            for j, v2ij in enumerate(v2i):
                fac = 1 if i == j else 2
                loss += fac * ops.sum(v2ij.value * (a[i] @ a[j].T)) / nx
        # for fai, a1i, ami, vi in zip(fa, a1, am, v, strict=True):
        for fvi, ai in zip(fv, a, strict=True):
            loss -= 2 * ops.sum(fvi.value * ai) / nx
            # loss -= 2 * ops.sum(bt * (ami @ vi)) / nt
            # loss -= 2 * ops.sum((a1i @ bx)[:, None] * vi) / nt / nx
        return self.scale * loss
