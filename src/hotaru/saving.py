from dataclasses import dataclass
from dataclasses import fields
from functools import wraps
from hashlib import md5
from json import dumps as json_dump
from os import PathLike as OriginalPathLike
from typing import Self
from typing import cast
from typing import dataclass_transform

from h5py import File
from keras.saving import deserialize_keras_object
from keras.saving import register_keras_serializable
from keras.saving import serialize_keras_object

Config = dict
PathLike = OriginalPathLike | str

Config = dict


@dataclass_transform()
@dataclass
class Data:
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        dataclass(cls)

    def save(self, path: PathLike, attrs: dict) -> None:
        with File(path, 'w') as db:
            for f in fields(self):
                ds = db.create_dataset(f.name, data=getattr(self, f.name))
            for k, v in attrs.items():
                ds.attrs[k] = v

    @classmethod
    def load(cls, path: PathLike) -> tuple[Self, dict]:
        with File(path, 'r') as db:
            obj = cls(**{f.name: db[f.name][...] for f in fields(cls)})
            attrs = dict(db.attrs)
            return obj, attrs


def cached_getter(cls: type[Data]):
    def decolator(func):
        @wraps(func)
        def wrap(*args, cache_path: PathLike | None = None, **kwargs):
            try:
                if cache_path is None:
                    raise Exception()
                out, _attrs = cls.load(cache_path)
            except Exception:
                out = None

            if out is None:
                out = func(*args, **kwargs)
                if cache_path is not None:
                    out.save(cache_path, {})

            return out

        return wrap

    return decolator


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
