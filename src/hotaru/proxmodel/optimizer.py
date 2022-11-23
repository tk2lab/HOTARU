import tensorflow as tf


class ProxOptimizer(tf.keras.optimizers.Optimizer):
    """"""

    def __init__(
        self,
        learning_rate=1000.0,
        nesterov_scale=20.0,
        reset_interval=100,
        **kwargs
    ):
        name = kwargs.setdefault("name", "Prox")
        super().__init__(**kwargs)
        self._learning_rate = self._build_learning_rate(learning_rate)
        self.nesterov_scale = nesterov_scale
        self.reset_interval = reset_interval

    def build(self, var_list):
        super().build(var_list)
        if hasattr(self, "_built") and self._built:
            return
        if self.nesterov_scale > 0:
            self.old = [
                self.add_variable_from_reference(
                    model_variable=var, variable_name="old"
                )
                for var in var_list
            ]
        self._built = True

    def update_step(self, gradient, variable):
        lr = tf.cast(self.learning_rate, variable.dtype)
        scale = tf.cast(self.nesterov_scale, variable.dtype)
        stepi = self.iterations % self.reset_interval
        step = tf.cast(stepi, variable.dtype)
        t0 = (scale + step) / scale
        t1 = (scale + 1) / (scale + step + 1)
        new_t = variable - lr * gradient
        if hasattr(variable, "prox"):
            new_t = variable.prox(new_t, lr)
        if self.nesterov_scale > 0:
            var_key = self._var_key(variable)
            old_v = self.old[self._index_dict[var_key]]
            tmp_t = (1.0 - t0) * old_v + t0 * new_t
            upd_t = (1.0 - t1) * new_t + t1 * tmp_t
            old_v.assign(new_t)
            variable.assign(upd_t)
        else:
            variable.assign(new_t)

    def get_config(self):
        config = super().get_config()
        config.update(
            dict(
                learning_rate=self._serialize_hyperparameter(
                    self._learning_rate,
                ),
                nesterov_scale=self.nesterov_scale,
                reset_interval=self.reset_interval,
            )
        )
