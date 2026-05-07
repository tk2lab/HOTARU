from os import PathLike as OriginalPathLike
from typing import Self
from typing import cast

from keras.saving import deserialize_keras_object
from keras.saving import register_keras_serializable
from keras.saving import serialize_keras_object

Config = dict
PathLike = OriginalPathLike | str

Config = dict


class Serializable:
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        register_keras_serializable()(cls)

    def get_config(self) -> Config:
        return {}

    @classmethod
    def from_config(cls, config: Config, /) -> Self:
        return cls(**config)

    def serialize(self) -> Config:
        return serialize_keras_object(self)

    @classmethod
    def get(cls, x: Serializable | Config, /) -> Self:
        match x:
            case Config() as config:
                return cast(Self, deserialize_keras_object(config))
            case cls() as obj:
                return obj
            case _:
                raise ValueError()
