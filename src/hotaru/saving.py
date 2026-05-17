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

    def save(self, path: PathLike | Group) -> None:
        cm = nullcontext(path) if isinstance(path, Group) else File(path, 'w')
        with cm as db:
            for f in fields(self):
                ds = db.create_dataset(f.name, data=getattr(self, f.name))
            for k, v in self.attrs.items():
                ds.attrs[k] = v
            ds.attrs['module'] = self.__module__
            ds.attrs['class'] = self.__class__.__name__

    @classmethod
    def load(cls, path: PathLike, group: str | None = None) -> Self:
        with File(path, 'r') as db:
            if group is not None:
                db = db[group]
            attrs = dict(db.attrs)
            module = attrs.pop('module', None)
            clsname = attrs.pop('class', None)
            if module and clsname:
                cls = getattr(import_module(module), clsname)
            obj = cls(**{f.name: db[f.name][...] for f in fields(cls)})
            obj.attrs.update(attrs)
            return obj


def cached_getter(cls: type[Data]):
    def decolator(func):
        @wraps(func)
        def wrap(
            *args,
            cache_path: PathLike | None = None,
            names: tuple[str] | None = None,
            force: bool = False,
            **kwargs,
        ):
            try:
                if force or (cache_path is None):
                    raise Exception()
                if names is None:
                    out = cls.load(cache_path)
                else:
                    out = [cls.load(cache_path, name) for name in names]
            except Exception:
                out = None

            if out is None:
                out = func(*args, **kwargs)
                if cache_path is not None:
                    if names is None:
                        out.save(cache_path)
                    else:
                        with File(cache_path, 'w') as db:
                            for name, outi in zip(names, out, strict=True):
                                outi.save(db.create_group(name))
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
