from keras import Model as KerasModel
from keras import StatelessScope as KerasStatelessScope
from keras.backend import backend
from keras.callbacks import ProgbarLogger

from .callbacks import TqdmProgbar
from .saving import Serializable


class Model(Serializable, KerasModel):
    def fit(self, *args, **kwargs):
        desc = kwargs.pop('desc', self.name)
        leave = kwargs.pop('leave', True)
        callbacks = kwargs.pop('callbacks', [])
        if not any(isinstance(c, ProgbarLogger) for c in callbacks):
            callbacks = [*callbacks, TqdmProgbar(desc=desc, leave=leave)]
        kwargs['callbacks'] = callbacks
        return super().fit(*args, **kwargs)

    def train_step(self, *args):
        if not hasattr(self, 'custom_train_step'):
            return super().train_step(*args)
        match backend():
            case 'jax':
                state, args = args
                with StatelessScope(self, state) as scope:
                    logs = self.custom_train_step(*args)
                return logs, scope.state
            case 'tensorflow' | 'torch':
                return self.custom_train_step(*args)


class StatelessScope(KerasStatelessScope):
    def __init__(self, model: KerasModel, state, *args, **kwargs):
        variables = (
            model.trainable_variables,
            model.non_trainable_variables,
            model.optimizer.variables,
            model.metrics_variables,
        )
        mapping = []
        for vs, ss in zip(variables, state, strict=True):
            mapping.extend(zip(vs, ss, strict=True))
        super().__init__(mapping, *args, **kwargs)
        self.variables = variables
        self.state = state

    def __exit__(self, *args, **kwargs):
        super().__exit__(*args, **kwargs)
        get_current_value = super().get_current_value
        self.state = tuple([get_current_value(v) for v in vs] for vs in self.variables)
