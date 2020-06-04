import tensorflow as tf


def get_prox(var):
    if hasattr(var, 'regularizer') and hasattr(var.regularizer, 'prox'):
        return var.regularizer.prox
    else:
        return lambda x: x


class ProxOptimizer(tf.keras.optimizers.Optimizer):

    _HAS_AGGREGATE_GRAD = True

    def __init__(self, learning_rate=0.01, name='Prox', **kwargs):
        super().__init__(name, **kwargs)
        self._set_hyper('learning_rate', kwargs.get('lr', learning_rate))
    
    def _create_slots(self, var_list):
        pass

    def _prepare_local(self, var_device, var_dtype, apply_state):
        super().__prepare_local(var_device, var_dtype, apply_state)
        apply_state[(var_device, var_dtype)].update(dict(
        ))

    def _resource_apply_dense(self, grad, var, apply_state=None):
        key = var.device, var.dtype.base_dtype
        coef = apply_state.get(key)
        lr = coef['lr_t']
        new_var = get_prox(var)(var + lr * grad, lr)
        return tf.assign(var, new_var, use_locking=self._use_locking).op

    def get_config(self):
        config = super().get_config()
        config.update(dict(
            learning_rate=self._serialize_hyperparameter('learning_rate'),
        ))
