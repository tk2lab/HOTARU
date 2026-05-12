from logging import getLogger
from pathlib import Path

import av
import numpy as np
import tifffile
import zarr
from tqdm import tqdm

from ..saving import PathLike
from ..typing import Array

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
        case ('raw', {'dtype': dtype, 'endian': endian, 'height': height, 'width': width}):
            dtype = np.dtype(dtype).newbyteorder(endian)
            data = np.memmap(path, dtype, 'r')
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


def to_movie(outfile, imgs, shape, fps, fmt='yuv420p', bit_rate=8_000_000, **kwargs):
    with av.open(outfile, 'w') as output:
        stream = output.add_stream(kwargs.get('codec', 'h264'), int(fps))
        stream.pix_fmt = fmt
        stream.bit_rate = bit_rate
        stream.height = shape[1]
        stream.width = shape[2]
        for img in tqdm(imgs, total=shape[0]):
            frame = av.VideoFrame.from_ndarray(img, format='rgba')
            packet = stream.encode(frame)
            output.mux(packet)
