import tensorflow as tf

from .regularizer import get_prox


class ProxOptimizer(tf.keras.optimizers.Optimizer):

    _HAS_AGGREGATE_GRAD = True

    def __init__(self, learning_rate=1.0, name='Prox', **kwargs):
        super().__init__(name, **kwargs)
        self._set_hyper('learning_rate', kwargs.get('lr', learning_rate))

    def _resource_apply_dense(self, grad, var, apply_state=None):
        key = var.device, var.dtype.base_dtype
        coef = apply_state.get(key)
        lr = coef['lr_t']
        new_var = get_prox(var)(var - lr * grad, lr)
        return var.assign(new_var, use_locking=self._use_locking)

    def get_config(self):
        config = super().get_config()
        config.update(dict(
            learning_rate=self._serialize_hyperparameter('learning_rate'),
        ))
        return config
