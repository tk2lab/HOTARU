import numpy as np
from keras import ops
from keras.losses import MeanSquaredError

from hotaru.optimizers import L1Regularizer
from hotaru.optimizers import ProxModel
from hotaru.optimizers import ProxOptimizer


class MyModel(ProxModel):
    def build(self, input_shape):
        *_, size = input_shape
        self.w = self.add_weight(shape=(size,), regularizer=L1Regularizer(1.0))
        super().build(input_shape)

    def call(self, x):
        return ops.sum(self.w * x, axis=-1)


def test_prox():
    model = MyModel()
    model.compile(
        loss=MeanSquaredError(),
        optimizer=ProxOptimizer(learning_rate=0.01, nesterov=1.0),
    )

    n_data = 100
    rng = np.random.default_rng()
    w = rng.uniform(size=(10,))
    x = rng.uniform(size=(n_data, 10))
    y = (w * x).sum(axis=-1) + 0.01 * rng.uniform(size=(n_data,))

    model.fit(x, y, steps_per_epoch=100, epochs=100)
    print(w)
    print(model.w.numpy())
