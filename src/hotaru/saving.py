from contextlib import nullcontext
from dataclasses import dataclass
from dataclasses import fields
from functools import wraps
from importlib import import_module
from os import PathLike as OriginalPathLike
from typing import Self
from typing import cast
from typing import dataclass_transform

from h5py import File
from h5py import Group
from keras import saving

Config = dict
PathLike = OriginalPathLike | str

Config = dict


class Serializable:
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        saving.register_keras_serializable()(cls)

    def get_config(self) -> Config:
        return {}

    @classmethod
    def from_config(cls, config: Config, /) -> Self:
        return cls(**config)

    @classmethod
    def get(cls, x: Serializable | Config, /) -> Self:
        match x:
            case Config() as config:
                return cls.from_config(config)
            case cls() as obj:
                return obj
            case _:
                raise ValueError()

    def serialize(self) -> Config:
        return saving.serialize_keras_object(self)


def cacheable(func):
    @wraps(func)
    def wrap(
        *args,
        cache_path: PathLike | None = None,
        force: bool = False,
        **kwargs,
    ):
        try:
            if force or (cache_path is None):
                raise Exception()
            cache = saving.load_weights(cache_path)
            out = func(*args, cache=cache, **kwargs)
        except Exception:
            out = None

        if out is None:
            out = func(*args, **kwargs)
            if cache_path is not None:
                saving.save_weights(out, cache_path)
        return out

    return wrap
