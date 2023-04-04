import pathlib

import numpy as np

# import tensorflow as tf
import tifffile


def load_imgs(path, in_type=None):
    path = pathlib.Path(path)

    if in_type is None:
        in_type = path.suffix[1:]

    if in_type == "npy":
        return np.load(path, mmap_mode="r")

    if in_type == "raw":
        with open(f"{path}.info", "r") as fp:
            info = fp.readline().replace("\n", "")
        dtype, h, w, endian = info.split(",")
        h, w = int(h), int(w)
        endian = "<" if endian == "l" else ">"
        dtype = np.dtype(dtype).newbyteorder(endian)
        return np.memmap(path, dtype, "r").reshape(-1, h, w)

    if in_type == "tif" or in_type == "tiff":
        fix_path = path.with_stem(path.stem + "_fix")
        if fix_path.exists():
            path = fix_path
        with tifffile.TiffFile(path) as tif:
            imgs = tif.series[0]
            if imgs.dataoffset is None:
                mmap = tifffile.memmap(fix_path, shape=imgs.shape, dtype=imgs.dtype)
                for i, pi in enumerate(imgs):
                    mmap[i] = pi.asarray()
            else:
                mmap = tif.asarray(out="memmap")
        return mmap

    raise RuntimeError(f"{path} is not imgs file")


def dataset(imgs):
    nt, h, w = imgs.shape
    imgs = tf.data.Dataset.from_generator(
        imgs.__iter__,
        output_signature=tf.TensorSpec([h, w], dtype=tf.float32),
    )
    imgs.shape = nt, h, w
    return imgs
