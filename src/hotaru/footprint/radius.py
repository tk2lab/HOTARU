from typing import Self

import numpy as np

from ..saving import Config


class Radius(np.ndarray):
    @classmethod
    def get(cls, x: Radius | Config) -> Self:
        match x:
            case cls() as radius:
                return radius
            case Config() as config:
                return cls(**config)
            case _:
                raise ValueError()

    def __new__(cls, **kwargs):
        match kwargs:
            case {'kind': 'logscale', 'min': min, 'max': max, 'num': num}:
                array = np.geomspace(min, max, num, dtype='float32')
            case {'kind': 'linear', 'min': min, 'max': max, 'num': num}:
                array = np.linspace(min, max, num, dtype='float32')
            case {'kind': 'list', 'val': val}:
                array = np.array(val, 'float32')
            case _:
                raise ValueError()
        return np.asanyarray(array).view(cls)

    def __array_finalize__(self, obj) -> None:
        pass
