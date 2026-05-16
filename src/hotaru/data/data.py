from dataclasses import dataclass
from logging import getLogger
from typing import Self

import numpy as np

from ..saving import Config
from ..saving import PathLike
from ..typing import Array
from ..typing import Shape
from .imgs import apply_mask
from .imgs import load_imgs

logger = getLogger(__name__)


@dataclass
class MovieData:
    imgs: Array
    mask: Array[np.bool]
    hz: float

    @property
    def shape(self) -> Shape:
        return self.imgs.shape

    @property
    def num_frames(self) -> int:
        return self.imgs.shape[0]

    @property
    def width(self) -> int:
        return self.imgs.shape[2]

    @property
    def height(self) -> int:
        return self.imgs.shape[1]

    @classmethod
    def get(cls, x: MovieData | Config, /) -> Self:
        match x:
            case cls() as obj:
                return obj
            case Config() as config:
                return cls.load(**config)
            case _:
                raise ValueError()

    @classmethod
    def load(cls, path: PathLike, hz: float, **kwargs) -> Self:
        imgs = load_imgs(path, **kwargs.pop('imgs', {}))
        imgs, mask = apply_mask(imgs, **kwargs.pop('mask', {'kind': 'nomask'}))
        return cls(imgs, mask, hz, **kwargs)
