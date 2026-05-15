from __future__ import annotations  # req: Python < 3.14 (PEP 563)

from dataclasses import dataclass
from dataclasses import fields
from functools import wraps
from importlib import import_module
from os import PathLike as OriginalPathLike
from pathlib import Path
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

    def __post_init__(self):
        self.attrs = {}

    def save(self, path: PathLike) -> None:
        with File(path, 'w') as db:
            for f in fields(self):
                ds = db.create_dataset(f.name, data=getattr(self, f.name))
            for k, v in self.attrs.items():
                ds.attrs[k] = v
            ds.attrs['module'] = self.__module__
            ds.attrs['class'] = self.__class__.__name__

    @classmethod
    def load(cls, path: PathLike) -> Self:
        with File(path, 'r') as db:
            attrs = dict(db.attrs)
            module = attrs.pop('module', None)
            clsname = attrs.pop('class', None)
            if module and clsname:
                cls = getattr(import_module(module), clsname)
            obj = cls(**{f.name: db[f.name][...] for f in fields(cls)})
            obj.attrs.update(attrs)
            return obj


class DataDict(dict):
    def save(self, path: PathLike) -> None:
        path = Path(path)
        for k, v in self.items():
            v.save(path.with_stem(f'{path.stem}_{k}'))

    @classmethod
    def load(cls, path: PathLike, names: list[str]) -> Self:
        path = Path(path)
        return cls({k: Data.load(path.with_stem(f'{path.stem}_{k}')) for k in names})


def cached_getter(cls: type[Data] | type[DataDict]):
    def decolator(func):
        @wraps(func)
        def wrap(
            *args,
            cache_path: PathLike | None = None,
            names: list[str] | None = None,
            force: bool = False,
            **kwargs,
        ):
            try:
                if force or (cache_path is None):
                    raise Exception()
                if issubclass(cls, DataDict) and names is not None:
                    out = cls.load(cache_path, names)
                elif issubclass(cls, Data):
                    out = cls.load(cache_path)
                else:
                    raise ValueError()
            except Exception:
                out = None

            if out is None:
                out = func(*args, **kwargs)
                if cache_path is not None:
                    if issubclass(cls, DataDict) and names is not None:
                        out = cls(zip(names, out, strict=True))
                    out.save(cache_path)

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
