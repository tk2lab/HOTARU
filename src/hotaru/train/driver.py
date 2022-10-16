import tensorflow as tf

from ..proxmodel import ProxModel
from ..util.distribute import distributed_matmul
from .input import dummy_inputs
from .input import dummy_tensor
from .loss import CacheLayer
from .loss import OutputLayer
from .loss import LossLayer
from .loss import IdentityLoss


class CommonModel(ProxModel):
    """"""

    def __init__(self, cache_layer, xval, base, **kwargs):
        xout = OutputLayer()(xval)
        cache = cache_layer(dummy_inputs)
        loss = LossLayer(*base.data.shape)([xout, cache])
        super().__init__(dummy_inputs, loss, **kwargs)
        self.data = base.data
        self.spike_to_calcium = base.spike_to_calcium
        self.cache = cache_layer

    def compile(self, **kwargs):
        super().compile(loss=IdentityLoss(), **kwargs)

    def fit(self, epochs, **kwargs):
        data = tf.data.Dataset.from_tensors((dummy_tensor, dummy_tensor))
        return super().fit(data.repeat(), epochs=epochs, **kwargs)


class SpatialModel(CommonModel):
    """"""

    def __init__(self, base, **kwargs):
        nt, nx = base.data.shape
        nk = base.max_nk
        bx = base._penalty.bx
        bt = base._penalty.bt
        cache_layer = CacheLayer(nk, nx, bx, bt, name="cache")
        footprint = base.footprint(dummy_inputs)
        localx = base.localx(dummy_inputs)
        xval = tf.keras.layers.concatenate([footprint, localx], axis=0)
        super().__init__(cache_layer, xval, base, **kwargs)
        self.add_loss(base.spike._callable_losses)
        self.add_loss(base.localt._callable_losses)

        self.get_spike = base.spike.val_tensor
        self.get_localt = base.localt.val_tensor
        self.footprint = base.footprint
        self.localx = base.localx

    def prepare(self, batch):
        spike = self.get_spike()
        localt = self.get_localt()
        calcium = self.spike_to_calcium(spike)
        val = tf.concat([calcium, localt], axis=0)
        cor = distributed_matmul(val, self.data, batch, False)
        self.cache.prepare(val, cor)
        self.footprint.clear(tf.shape(spike)[0])
        self.localx.clear(tf.shape(localt)[0])


class TemporalModel(CommonModel):
    """"""

    def __init__(self, base, **kwargs):
        nt, nx = base.data.shape
        nk = base.max_nk
        bx = base._penalty.bx
        bt = base._penalty.bt
        cache_layer = CacheLayer(nk, nt, bt, bx, name="cache")
        spike = base.spike(dummy_inputs)
        localt = base.localt(dummy_inputs)
        calcium = base.spike_to_calcium(spike)
        tval = tf.keras.layers.concatenate([calcium, localt], axis=0)
        super().__init__(cache_layer, tval, base, **kwargs)
        self.add_loss(base.footprint._callable_losses)
        self.add_loss(base.localx._callable_losses)

        self.get_footprint = base.footprint.val_tensor
        self.get_localx = base.localx.val_tensor
        self.spike = base.spike
        self.localt = base.localt

    def prepare(self, batch):
        footprint = self.get_footprint()
        localx = self.get_localx()
        val = tf.concat([footprint, localx], axis=0)
        cor = distributed_matmul(val, self.data, batch, True)
        self.cache.prepare(val, cor)
        self.spike.clear(tf.shape(footprint)[0])
        self.localt.clear(tf.shape(localx)[0])
