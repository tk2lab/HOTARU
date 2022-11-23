import tensorflow as tf

from ..proxmodel import L2
from ..proxmodel import MaxNormNonNegativeL1
from ..proxmodel import NonNegativeL1

dummy_tensor = tf.zeros((1,), tf.float32, name="dummy")
dummy_inputs = tf.keras.Input(tensor=dummy_tensor)


def is_split_variable(x):
    return hasattr(x, "_variable_list") or hasattr(x, "_variables")


class DynamicInputLayer(tf.keras.layers.Layer):
    """Dynamic Input Layer"""

    def __init__(self, nk, nx, **kwargs):
        super().__init__(**kwargs)
        self._nk = self.add_weight("nk", (), tf.int32, trainable=False)
        self._val = self.add_weight("val", (nk, nx), tf.float32)

    @property
    def val(self):
        return self.val_tensor().numpy()

    @val.setter
    def val(self, val):
        if tf.is_tensor(val):
            nk = tf.shape(val)[0]
            self._nk.assign(nk)
            self._val[:nk].assign(val)
        else:
            batch = 100
            nk = len(val)
            self._nk.assign(nk)
            for k in range(0, nk, batch):
                e = min(nk, k + batch)
                self._val[k:e].assign(val[k:e])

    @property
    def regularizer(self):
        return self._regularizer

    @regularizer.setter
    def regularizer(self, regularizer):
        self._regularizer = regularizer
        prox = regularizer.prox
        if is_split_variable(self._val):
            for v in self._val:
                v.prox = prox
        else:
            self._val.prox = prox
        self._val.penalty = lambda: self.regularizer(self.val_tensor())

    def clear(self, nk):
        self._nk.assign(nk)
        self._val.assign(tf.zeros_like(self._val))

    def val_tensor(self):
        return self._val[: self._nk]

    def call(self, inputs):
        return self.val_tensor()

    def get_config(self):
        config = super().get_config()
        config.update(dict(
            nk=self._val.shape[0],
            nx=self._val.shape[1],
        ))
        return config


class DynamicL2InputLayer(DynamicInputLayer):
    """"""

    def __init__(self, nk, nx, l, **kwargs):
        super().__init__(nk, nx, **kwargs)
        self.regularizer = L2(l)

    def get_config(self):
        config = super().get_config()
        config.update(dict(
            l=self._regularizer.l.numpy(),
        ))
        return config


class DynamicNonNegativeL1InputLayer(DynamicInputLayer):
    """"""

    def __init__(self, nk, nx, l, **kwargs):
        super().__init__(nk, nx, **kwargs)
        self.regularizer = NonNegativeL1(l)

    def get_config(self):
        config = super().get_config()
        config.update(dict(
            l=self._regularizer.l.numpy(),
        ))
        return config

class DynamicMaxNormNonNegativeL1InputLayer(DynamicInputLayer):
    """"""

    def __init__(self, nk, nx, l, axis=-1, **kwargs):
        super().__init__(nk, nx, **kwargs)
        self.regularizer = MaxNormNonNegativeL1(l, axis=axis)

    def get_config(self):
        config = super().get_config()
        config.update(dict(
            l=self._regularizer.l.numpy(),
            axis=self._regularizer.axis,
        ))
        return config
