import pathlib

import numpy as np
from tifffile import (
    TiffFile,
    memmap,
)


def load_imgs(path, in_type=None):
    path = pathlib.Path(path)

    if in_type is None:
        in_type = path.suffix[1:]

    if in_type == "npy":
        return np.load(path, mmap_mode="r")

    if in_type == "raw":
        # info = path.with_suffix(".raw.info")
        # h, w, dtype, endian = info.read_text().replace("\n", "").split(".")
        h, w, dtype, endian = path.suffixes
        h, w = int(h), int(w)
        endian = "<" if endian == "l" else ">"
        dtype = np.dtype(dtype).newbyteorder(endian)
        return np.memmap(path, dtype, "r").reshape(-1, h, w)

    if in_type == "tif" or in_type == "tiff":
        fix_path = path.with_stem(f"{path.stem}_fix")
        if fix_path.exists():
            path = fix_path
        with TiffFile(path) as tif:
            imgs = tif.series[0]
            if imgs.dataoffset is not None:
                mmap = tif.asarray(out="memmap")
            else:
                mmap = memmap(fix_path, shape=imgs.shape, dtype=imgs.dtype)
                for i, pi in enumerate(imgs):
                    mmap[i] = pi.asarray()
        return mmap

    raise RuntimeError(f"{path} is not imgs file")
