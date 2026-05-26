from typing import Self

from keras import saving
from keras.layers import Layer as KerasLayer

from ..saving import Config


class Layer(KerasLayer):
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        saving.register_keras_serializable()(cls)

    @classmethod
    def get(cls, x: Layer | Config, /) -> Self:
        match x:
            case Config() as config:
                return cls.from_config(config)
            case cls() as obj:
                return obj
            case _:
                raise ValueError()
