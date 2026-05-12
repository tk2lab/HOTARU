from keras import ops
from keras.callbacks import EarlyStopping
from keras.callbacks import TerminateOnNaN

from ..config import TotalProperty
from ..data import MovieWithStats
from ..losses import Minimize
from ..models import ProxModel
from ..spatial import Footprints
from ..typing import Tensor
from .prepare import TemporalPrepare


class TemporalUpdater(ProxModel):
    def __init__(self, props: TotalProperty, **kwargs):
        super().__init__(**kwargs)
        self.props = props

    def compile(self, **kwargs) -> None:
        kwargs.setdefault('loss', Minimize())
        super().compile(**kwargs)

    def prepare(self, data: MovieWithStats, *footprints: Footprints, **kwargs) -> None:
        prepare = TemporalPrepare(self)
        prepare.update(data, *footprints, **kwargs)

    def update(self, *, reset: bool = True, **kwargs):
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
        temporal_activities = [a.numpy() for a in self.activity]
        return temporal_activities

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
        for k, nk in enumerate(nks):
            p = self.props.component_properties[k]
            ntau = nt + p.temporal_kernel.size - 1
            regularizer = p.temporal_regularizer
            self.activity.append(self.add_weight((nk, ntau), regularizer=regularizer))
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

    def call(self, _dummy: Tensor) -> Tensor:
        def conv(d, k):
            return ops.conv(d[:, :, None], k[:, None, None], 1, 'valid', 'channels_last')[..., 0]

        a1 = self.spatial_comp
        am = self.spatial_mean
        fa = self.spatial_corr
        a2 = self.spatial_sqrd

        num = len(a1)
        _, nx = a1[0].shape
        _, nt = fa[0].shape

        kernel = [p.temporal_kernel for p in self.props.component_properties]
        v = [conv(a, k) for a, k in zip(self.activity, kernel, strict=True)]
        vm = [ops.mean(v[i], axis=1) for i in range(num)]

        bt = self.temporal_baseline
        bt -= ops.mean(bt)
        bx = self.spatial_baseline
        bx -= ops.mean(bx)
        b0 = ops.sum([ops.sum(am[i] * vm[i]) for i in range(num)])

        loss = ops.mean(ops.square(bt)) + ops.mean(ops.square(bx)) - ops.square(b0)
        for i in range(num):
            for j in range(i + 1):
                loss += ops.sum(a2[i][j] * (v[i] @ v[j].T)) / nt
        for i in range(num):
            loss -= 2 * ops.sum(fa[i] * v[i]) / nt
            loss -= 2 * ops.sum(bt * (am[i] @ v[i])) / nt
            loss -= 2 * ops.sum((a1[i] @ bx)[:, None] * v[i]) / nt / nx
        return self.scale * loss
