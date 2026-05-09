from keras import ops
from keras.callbacks import Callback
from keras.callbacks import History
from keras.src.backend import get_stateless_scope
from keras.src.backend import in_stateless_scope
from keras.src.backend import in_symbolic_scope

from ..model import Model
from .prox_optimizer import ProxOptimizer
from .prox_regularizer import ProxRegularizer


class ProxCallback(Callback):
    def on_epoch_end(self, epoch, logs=None):
        _ = epoch
        _ = logs
        self.model.optimizer.finalize_variable_values(self.model.trainable_variables)
        self.model.optimizer.iterations.assign(0)


class ProxModel(Model):
    def fit(self, x, y, **kwargs) -> History:
        def data_iterator():
            while True:
                yield x, y

        callbacks = kwargs.setdefault('callbacks', [])
        callbacks.append(ProxCallback())
        return super().fit(data_iterator(), **kwargs)

    def _get_regularization_losses(self, *, prox: bool = False):
        in_scope = in_stateless_scope() and not in_symbolic_scope()
        scope = get_stateless_scope() if in_scope else None
        regularizer_losses = []
        for variable in self.trainable_weights:
            if variable.regularizer is None:
                continue
            v = variable if scope is None else scope.get_current_value(variable)
            if isinstance(self.optimizer, ProxOptimizer):
                is_prox = isinstance(variable.regularizer, ProxRegularizer)
                if (prox and is_prox) or (not prox and not is_prox):
                    regularizer_losses.append(variable.regularizer(v))
            else:
                regularizer_losses.append(variable.regularizer(v))
        return regularizer_losses

    def post_train_step(self, logs: dict) -> dict:
        if isinstance(self.optimizer, ProxOptimizer):
            prox_losses = self._get_regularization_losses(prox=True)
            prox_loss = ops.sum([self._aggregate_additional_loss(loss) for loss in prox_losses])
            logs['loss'] += prox_loss
        return logs
