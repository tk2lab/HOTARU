import tensorflow as tf

from ..util.distribute import ReduceOp
from ..util.distribute import distributed
from .dynamics import DoubleExpMixin
from .variable import HotaruVariableMixin
from .optimizer import ProxOptimizer as Optimizer
from .loss import HotaruLoss as Loss


class HotaruConfigMixin:
    """Model"""

    def set_early_stop(self, *args, **kwargs):
        self.callbacks = [
            tf.keras.callbacks.EarlyStopping("score", *args, **kwargs),
        ]

    def set_optimizer(self, *args, **kwargs):
        self.spatial.optimizer.set(*args, **kwargs)
        self.temporal.optimizer.set(*args, **kwargs)
