import tensorflow as tf

from ..util.distribute import ReduceOp
from ..util.distribute import distributed
from .dynamics import DoubleExpMixin
from .variable import HotaruVariableMixin as VariableMixin
from .optimizer import ProxOptimizer as Optimizer
from .config import HotaruConfigMixin as ConfigMixin
from .loss import HotaruLoss as Loss


class HotaruModel(tf.Module, DoubleExpMixin, VariableMixin, ConfigMixin):
    """Model"""

    @tf.Module.with_name_scope
    def build(self, data, nk, nx, nt, hz, tausize, local_strategy=None):
        dummy_input = tf.keras.Input(type_spec=tf.TensorSpec((), tf.float32))
        concat_layer = tf.keras.layers.Concatenate(axis=0)

        self.init_double_exp(hz, tausize)
        self.init_variable(nk, nx, nt, tausize)

        footprint = self.footprint(dummy_input)
        spike = self.spike(dummy_input)
        localx = self.localx(dummy_input)
        localt = self.localt(dummy_input)
        penalty = self.penalty(footprint, spike, localx, localt)

        xval = concat_layer([footprint, localx])
        spatial_loss = self.spatial_loss(xval)

        calcium = self.spike_to_calcium(spike)
        tval = concat_layer([calcium, localt])
        temporal_loss = self.temporal_loss(tval)

        spatial_model = tf.keras.Model(dummy_input, spatial_loss, name="spatial")
        spatial_model.add_metric(penalty, "penalty")
        spatial_model.add_metric(spatial_loss + penalty, "score")

        temporal_model = tf.keras.Model(dummy_input, temporal_loss, name="temporal")
        temporal_model.add_metric(penalty, "penalty")
        temporal_model.add_metric(temporal_loss + penalty, "score")

        self.local_strategy = local_strategy
        self.data = data
        self.spatial = spatial_model
        self.temporal = temporal_model

    def compile_spatial(self, **kwargs):
        self.spatial.compile(optimizer=Optimizer(), loss=Loss(), **kwargs)

    def compile_temporal(self, **kwargs):
        self.temporal.compile(optimizer=Optimizer(), loss=Loss(), **kwargs)

    def prepare_spatial(self, batch, prog=None):
        @distributed(ReduceOp.SUM, strategy=self.local_strategy)
        def _matmul(tdat, val):
            t, dat = tdat
            val = tf.gather(val, t, axis=1)
            return tf.linalg.matmul(val, dat)

        calcium = self.spike_to_calcium(self.spike.val)
        localt = self.localt.val
        val = tf.concat([calcium, localt], axis=0)

        data = self.data.enumerate().batch(batch)
        cor = _matmul(data, val, prog=prog)
        lipschitz = self.spatial_loss._cache(val, cor)
        self.spatial.optimizer.set_lipschitz(lipschitz.numpy())

    def prepare_temporal(self, batch, prog=None):
        @distributed(ReduceOp.CONCAT, strategy=self.local_strategy)
        def _matmul(dat, val):
            return tf.matmul(dat, val, False, True)

        footprint = self.footprint.val
        localx = self.localx.val
        val = tf.concat([footprint, localx], axis=0)

        data = self.data.batch(batch)
        cor = tf.transpose(_matmul(data, val, prog=prog))
        lipschitz = self.temporal_loss._cache(val, cor)
        lipschitz *= tf.math.reduce_sum(self.spike_to_calcium.kernel)
        self.temporal.optimizer.set_lipschitz(lipschitz.numpy())

    def fit_spatial(self, **kwargs):
        self.footprint.clear(self.spike.get_num())
        self.localx.clear(self.localt.get_num())
        return self._fit(self.spatial, **kwargs)

    def fit_temporal(self, **kwargs):
        self.spike.clear(self.footprint.get_num())
        self.localt.clear(self.localx.get_num())
        return self._fit(self.temporal, **kwargs)

    def _fit(self, model, epochs=100, **kwargs):
        callbacks = kwargs.pop("callbacks", [])
        callbacks += self.callbacks
        dummy = tf.zeros((1, 1))
        return model.fit(
            tf.data.Dataset.from_tensors((dummy, dummy)).repeat(),
            steps_per_epoch=model.optimizer.reset_interval,
            epochs=epochs,
            callbacks=callbacks,
            **kwargs,
        )
