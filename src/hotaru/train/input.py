import tensorflow as tf

from ..proxmodel import L2
from ..proxmodel import NonNegativeL1
from ..proxmodel import MaxNormNonNegativeL1


dummy_tensor = tf.zeros((1,), tf.float32, name="dummy")
dummy_inputs = tf.keras.Input(tensor=dummy_tensor)


def is_split_variable(x):
    return hasattr(x, "_variable_list") or hasattr(x, "_variables")


class DynamicInputLayer(tf.keras.layers.Layer):
    """Dynamic Input Layer"""

    def __init__(self, nk, nx, regularizer=None, **kwargs):
        super().__init__(**kwargs)
        self._nk = self.add_weight("nk", (), tf.int32, trainable=False)
        self._val = self.add_weight("val", (nk, nx), tf.float32)
        self._regularizer = regularizer
        if regularizer:
            prox = regularizer.prox
            if is_split_variable(self._val):
                for v in self._val:
                    v.prox = prox
            else:
                self._val.prox = prox
            self.add_loss(lambda: self.regularizer(self.val_tensor()))

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

    def clear(self, nk):
        self._nk.assign(nk)
        self._val.assign(tf.zeros_like(self._val))

    def val_tensor(self):
        return self._val[: self._nk]

    def call(self, inputs):
        return self.val_tensor()


class DynamicL2InputLayer(DynamicInputLayer):
    """"""

    def __init__(self, nk, nx, l, **kwargs):
        Regularizer = L2
        name = kwargs.get("name", "input")
        regularizer = Regularizer(l)
        super().__init__(nk, nx, regularizer, **kwargs)


class DynamicNonNegativeL1InputLayer(DynamicInputLayer):
    """"""

    def __init__(self, nk, nx, l, **kwargs):
        Regularizer = NonNegativeL1
        name = kwargs.get("name", "input")
        regularizer = Regularizer(l)
        super().__init__(nk, nx, regularizer, **kwargs)


class DynamicMaxNormNonNegativeL1InputLayer(DynamicInputLayer):
    """"""

    def __init__(self, nk, nx, l, axis=-1, **kwargs):
        Regularizer = MaxNormNonNegativeL1
        name = kwargs.get("name", "input")
        regularizer = Regularizer(l, axis=axis)
        super().__init__(nk, nx, regularizer, **kwargs)
