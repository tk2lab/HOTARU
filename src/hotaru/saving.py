import base64
import hashlib
import inspect
import json
from functools import wraps
from os import PathLike as OriginalPathLike
from typing import Self

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


def auto_save_config(version=0.1, exclude=()):
    def decolator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            sig = inspect.signature(func)
            bound_args = sig.bind(*args, **kwargs)
            bound_args.apply_defaults()
            config = dict(bound_args.arguments)
            kwargs = config.pop('kwargs')
            config = {**config, **kwargs}

            path = config.pop('path')
            force = config.pop('force', False)
            extra = {key: config.pop(key) for key in exclude}

            config_str = json.dumps(to_str({**config, 'version': version}), sort_keys=True)
            config_bytes = hashlib.sha256(config_str.encode('utf-8')).digest()
            config_hash = base64.urlsafe_b64encode(config_bytes[:6]).decode('utf-8').rstrip('=')
            path = path / config_hash

            config_path = path / 'config.json'
            if force or not config_path.exists():
                path.mkdir(exist_ok=True, parents=True)
                func(path, **config, **extra)
                config_path.write_text(config_str)

            return path

        return wrapper

    return decolator


def to_str(x):
    match x:
        case dict():
            return {k: to_str(v) for k, v in x.items()}
        case list() | tuple():
            return [to_str(v) for v in x]
        case _:
            return str(x)


def make_link(dst_path, dst, target):
    target = target.absolute().relative_to(dst_path.absolute(), walk_up=True)
    dst_path = dst_path / dst
    dst_path.unlink(missing_ok=True)
    dst_path.symlink_to(target, target_is_directory=True)
