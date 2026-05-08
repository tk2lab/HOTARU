import numpy as np
from jax.debug import print as jprint
from keras import Model
from keras.utils import PyDataset


class MyDataset(PyDataset):
    def on_epoch_end(self):
        pass

    def __len__(self) -> int:
        return 10

    def __getitem__(self, index: int):
        _ = index
        return np.array(index), np.ones(3)


class MyModel(Model):
    def build(self, input_shape):
        print('build', input_shape)
        super().build(input_shape)

    def fit(self, *args, **kwargs):
        print('fit', self.built)
        self.build(10)
        return super().fit(*args, **kwargs)

    def train_step(self, state, *args, **kwargs):
        jprint('train: {}', args)
        self(*args, **kwargs)
        return {}, state

    def call(self, *args, **kwargs):
        jprint('call: {}', args)
        return args


data = MyDataset()
model = MyModel()
model.compile()
print('compile')
model.fit(data, shuffle=False)
