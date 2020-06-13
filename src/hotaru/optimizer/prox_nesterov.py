import tensorflow as tf

from .regularizer import get_prox


class ProxNesterov(tf.keras.optimizers.Optimizer):

    _HAS_AGGREGATE_GRAD = True

    def __init__(self, learning_rate=1.0, scale=20.0,
                 name='ProxNesterov', **kwargs):
        super().__init__(name, **kwargs)
        self._set_hyper('learning_rate', kwargs.get('lr', learning_rate))
        self._set_hyper('scale', scale)
        self._set_hyper('start', 0)

    def _create_slots(self, var_list):
        for var in var_list:
            self.add_slot(var, 'old')

    def _prepare_local(self, var_device, var_dtype, apply_state):
        super()._prepare_local(var_device, var_dtype, apply_state)
        scale = self._get_hyper('scale', var_dtype)
        step = tf.cast(self.iterations - self.start, var_dtype)
        apply_state[(var_device, var_dtype)].update(dict(
            t0=(scale + step) / scale,
            t1=(scale + 1) / (scale + step + 1),
        ))

    def _resource_apply_dense(self, grad, var, apply_state=None):
        key = var.device, var.dtype.base_dtype
        coef = apply_state.get(key)
        lr = coef['lr_t']
        t0 = coef['t0']
        t1 = coef['t1']
        old = self.get_slot(var, 'old')
        new_t = get_prox(var)(var - lr * grad, lr)
        tmp_t = (1.0 - t0) * old + t0 * new_t
        upd_t = (1.0 - t1) * new_t + t1 * tmp_t
        return tf.group(
            old.assign(new_t, use_locking=self._use_locking),
            var.assign(upd_t, use_locking=self._use_locking),
        )

    def get_config(self):
        config = super().get_config()
        config.update(dict(
            learning_rate=self._serialize_hyperparameter('learning_rate'),
            scale=self._serialize_hyperparameter('scale'),
        ))
        return config
