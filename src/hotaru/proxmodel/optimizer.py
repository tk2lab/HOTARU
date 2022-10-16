import tensorflow as tf


class ProxOptimizer(tf.keras.optimizers.Optimizer):

    _HAS_AGGREGATE_GRAD = True

    def __init__(
        self,
        learning_rate=1000.0,
        nesterov_scale=20.0,
        reset_interval=100,
        **kwargs
    ):
        name = kwargs.setdefault("name", "Prox")
        super().__init__(**kwargs)
        self._set_hyper("learning_rate", learning_rate)
        self._set_hyper("nesterov_scale", nesterov_scale)
        self._set_hyper("reset_interval", reset_interval)
        self._nesterov = (
            isinstance(nesterov_scale, tf.Tensor)
            or callable(nesterov_scale)
            or nesterov_scale > 0.0
        )

    def set(
        self, learning_rate=None, nesterov_scale=None, reset_interval=None
    ):
        if learning_rate:
            self.learning_rate = learning_rate
        if nesterov_scale:
            self.nesterov_scale = nesterov_scale
        if reset_interval:
            self.reset_interval = reset_interval

    def _create_slots(self, var_list):
        if self._nesterov:
            for var in var_list:
                self.add_slot(var, "old")

    def _prepare_local(self, var_device, var_dtype, apply_state):
        super()._prepare_local(var_device, var_dtype, apply_state)
        if self._nesterov:
            scale = self._get_hyper("nesterov_scale", var_dtype)
            reset = self._get_hyper("reset_interval", tf.int64)
            stepi = self.iterations % reset
            last = tf.equal(stepi, reset - 1)
            stepi = tf.where(last, tf.cast(0, tf.int64), stepi)
            step = tf.cast(stepi, var_dtype)
            apply_state[(var_device, var_dtype)].update(
                dict(
                    t0=(scale + step) / scale,
                    t1=(scale + 1) / (scale + step + 1),
                )
            )

    def _resource_apply_dense(self, grad, var, apply_state=None):
        key = var.device, var.dtype.base_dtype
        coef = apply_state.get(key)
        lr = coef["lr_t"]
        new_t = var - lr * grad
        if hasattr(var, "prox"):
            new_t = var.prox(new_t, lr)
        if self._nesterov:
            t0 = coef["t0"]
            t1 = coef["t1"]
            old = self.get_slot(var, "old")
            old_t = tf.identity(old)
            tmp_t = (1.0 - t0) * old_t + t0 * new_t
            upd_t = (1.0 - t1) * new_t + t1 * tmp_t
            return tf.group(
                old.assign(new_t, use_locking=self._use_locking),
                var.assign(upd_t, use_locking=self._use_locking),
            )
        else:
            return var.assign(new_t, use_locking=self._use_locking)

    def get_config(self):
        config = super().get_config()
        config.update(
            dict(
                learning_rate=self._serialize_hyperparameter("learning_rate"),
                scale=self._serialize_hyperparameter("scale"),
                reset=self._serialize_hyperparameter("reset"),
            )
        )
        return config
