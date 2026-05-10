from keras import Optimizer
from keras import ops

from ..saving import Config
from .prox_regularizer import ProxRegularizer


class ProxOptimizer(Optimizer):
    def __init__(self, nesterov: float = 1.0, name='prox', **kwargs):
        super().__init__(name=name, **kwargs)
        self.nesterov = nesterov

    def get_config(self) -> Config:
        return {**super().get_config(), 'nesterov': self.nesterov}

    def build(self, var_list):
        if self.built:
            return
        super().build(var_list)

        self._old_x = self.add_optimizer_variables(var_list, 'old_x')

    def update_step(self, gradient, variable, learning_rate):
        nesterov = self.nesterov
        i = ops.cast(self.iterations + 1, variable.dtype)
        j = i + 1
        t0 = (nesterov + i) / (nesterov + 1)
        t1 = (nesterov + 1) / (nesterov + j)

        lr = ops.cast(learning_rate, variable.dtype)
        gradient = ops.cast(gradient, variable.dtype)
        old_x = x = self._old_x[self._get_variable_index(variable)]
        old_y = y = variable
        new_x = old_y - lr * gradient
        if isinstance(variable.regularizer, ProxRegularizer):
            new_x = variable.regularizer.prox(new_x, lr)
        new_v = (1 - t0) * old_x + t0 * new_x
        new_y = (1 - t1) * new_x + t1 * new_v
        self.assign(y, new_y)
        self.assign(x, new_x)

    def finalize_valiable_values(self, var_list):
        for variable in var_list:
            variable.assign(self._old_x[self._get_variable_index(variable)])
