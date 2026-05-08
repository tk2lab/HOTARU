from dataclasses import dataclass
from logging import getLogger
from pathlib import Path

import numpy as np
from tifffile import TiffFile
from tifffile import imread
from tifffile import memmap

from ..saving import Config
from ..saving import PathLike
from ..typing import Array
from ..typing import Shape

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
    def get(cls, x: MovieData | Config, /) -> MovieData:
        match x:
            case cls() as obj:
                return obj
            case Config() as config:
                return cls.load(**config)
            case _:
                raise ValueError()

    @classmethod
    def load(cls, path: PathLike, hz: float, **kwargs) -> MovieData:
        path = Path(path)
        if (kind := kwargs.get('kind')) is None:
            match path.suffix:
                case '.npy':
                    kind = 'npy'
                case '.tif' | '.tiff':
                    kind = 'tif'
                case '.raw':
                    kind = 'raw'
                case _:
                    raise ValueError('unknown file type: {path.suffix}')
        match (kind, kwargs):
            case ('npy', {}):
                imgs = np.load(path, mmap_mode='r')
            case ('tif', {}):
                path_fix = path.with_stem(f'{path.stem}_fix')
                if path_fix.exists():
                    path = path_fix
                with TiffFile(path) as tif:
                    data = tif.series[0]
                    if data.dataoffset is None:
                        imgs = memmap(path_fix, shape=data.shape, dtype=data.dtype)
                        for i, pi in enumerate(data):
                            if pi is None:
                                raise ValueError('invalid tiff file')
                            imgs[i] = pi.asarray()
                    else:
                        imgs = data.asarray(out='memmap')
            case ('raw', {'dtype': dtype, 'endian': endian, 'height': height, 'width': width}):
                dtype = np.dtype(dtype).newbyteorder(endian)
                data = np.memmap(path, dtype, 'r')
                imgs = data.reshape(-1, height, width)
            case _:
                raise ValueError(f'unkown file type: {kind}')

        imgs, mask = apply_mask(imgs, **kwargs.get('mask', {'kind': 'nomask'}))
        return MovieData(imgs, mask, hz)


def apply_mask(imgs, **kwargs):
    kind = kwargs.get('kind')
    if kind == 'nomask':
        return imgs, np.ones(imgs.shape[-2:], np.bool)

    path = Path(kwargs.get('path', 'mask.tif'))
    match path.suffix:
        case '.npy':
            kind = 'npy'
        case '.tif' | '.tiff':
            kind = 'tif'
        case '.raw':
            kind = 'raw'
        case _:
            raise ValueError('unknown file type: {path.suffix}')

    match kind:
        case 'npy':
            mask = np.load(path) > 0
        case 'tif':
            mask = imread(path) > 0
        case _:
            raise RuntimeError('bad file type: {maskfile}')

    if mask is not None:
        my = np.where(np.any(mask, axis=1))[0]
        mx = np.where(np.any(mask, axis=0))[0]
        x0, y0, w, h = mx[0], my[0], mx[-1] - mx[0] + 1, my[-1] - my[0] + 1
        imgs = imgs[:, y0 : y0 + h, x0 : x0 + w]
        mask = mask[y0 : y0 + h, x0 : x0 + w]

    return imgs, mask
