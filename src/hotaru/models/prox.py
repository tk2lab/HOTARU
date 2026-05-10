from keras import ops
from keras.callbacks import Callback
from keras.callbacks import History
from keras.metrics import Mean as Metrix
from keras.src.backend import get_stateless_scope
from keras.src.backend import in_stateless_scope
from keras.src.backend import in_symbolic_scope

from ..optimizers import ProxOptimizer
from ..regularizers import ProxRegularizer
from .model import Model


class ProxCallback(Callback):
    def on_epoch_end(self, epoch, logs=None):
        _ = epoch
        _ = logs
        self.model.optimizer.finalize_variable_values(self.model.trainable_variables)
        self.model.optimizer.iterations.assign(0)


class ProxModel(Model):
    def compile(self, **kwargs) -> None:
        if 'optimizer' not in kwargs:
            optimizer_kwargs = kwargs.pop('optimizer_kwargs', {})
            optimizer_kwargs.setdefault('learning_rate', kwargs.pop('learning_rate'))
            optimizer_kwargs.setdefault('nesterov', kwargs.pop('nesterov', 1.0))
            kwargs['optimizer'] = ProxOptimizer(**optimizer_kwargs)
        super().compile(**kwargs)

    def build(self, input_shape) -> None:
        self.penalty_tracker = Metrix(name='penalty')
        self.total_loss_tracker = Metrix(name='total_loss')
        super().build(input_shape)

    def fit(self, x=None, y=None, **kwargs) -> History:
        if x is None:
            x = ops.zeros((1, 1))
        if y is None:
            y = ops.zeros((1, 1))

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

    def post_train_step(self, logs: dict) -> None:
        _ = logs
        if isinstance(self.optimizer, ProxOptimizer):
            penalties = self._get_regularization_losses(prox=True)
            penalty = ops.sum([self._aggregate_additional_loss(loss) for loss in penalties])
            self.penalty_tracker.update_state(penalty)
            self.total_loss_tracker.update_state(logs['loss'] + penalty)
