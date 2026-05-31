from logging import getLogger
from pathlib import Path

import av
import numpy as np
import tifffile
import zarr
from tqdm import tqdm

from ..saving import PathLike
from ..typing import Array
from ..typing import Shape

logger = getLogger(__name__)


def load_imgs(path: PathLike, **kwargs) -> Array | zarr.Array:
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
            tif = tifffile.imread(path, return_as='zarr')
            imgs = zarr.open(tif, mode='r')
            if not isinstance(imgs, zarr.Array):
                raise ValueError()
        case ('raw', {'dtype': dtype, 'height': height, 'width': width}):
            data = np.memmap(path, np.dtype(dtype), 'r')
            imgs = data.reshape(-1, height, width)
        case _:
            raise ValueError(f'unkown file type: {kind}')
    return imgs


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
            mask = tifffile.imread(path) > 0
        case _:
            raise RuntimeError('bad file type: {maskfile}')

    if mask is not None:
        my = np.where(np.any(mask, axis=1))[0]
        mx = np.where(np.any(mask, axis=0))[0]
        x0, y0, w, h = mx[0], my[0], mx[-1] - mx[0] + 1, my[-1] - my[0] + 1
        imgs = imgs[:, y0 : y0 + h, x0 : x0 + w]
        mask = mask[y0 : y0 + h, x0 : x0 + w]

    return imgs, mask


def to_movie(
    outfile: PathLike,
    imgs: Array,
    shape: Shape,
    fps: float,
    fmt: str = 'yuv420p',
    bit_rate: int = 8_000_000,
    **kwargs,
) -> None:
    n, h, w = shape
    ypad, xpad = 0, 0
    if h % 2 == 1:
        h += 1
        ypad = 1
    if w % 2 == 1:
        w += 1
        xpad = 1
    with av.open(outfile, 'w') as output:
        stream = output.add_stream(kwargs.get('codec', 'h264'), int(fps))
        stream.pix_fmt = fmt
        stream.bit_rate = bit_rate
        stream.height = h
        stream.width = w
        for img in tqdm(imgs, total=n, ncols=150, desc=f'make {outfile}'):
            img = np.pad(img, ((0, ypad), (0, xpad), (0, 0)))
            frame = av.VideoFrame.from_ndarray(img, format='rgba')
            packet = stream.encode(frame)
            output.mux(packet)
