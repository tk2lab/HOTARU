import numpy as np
import tensorflow as tf
import tifffile


class ImageStack:
    """"""

    def __init__(self, path, in_type=None):
        # path = ensure_local_file(path)
        if in_type is None:
            in_type = path.suffix[1:]

        if in_type == "npy":
            self._imgs = np.load(path, mmap_mode="r")
        elif in_type == "tif" or in_type == "tiff":
            imgs = tifffile.TiffFile(path).series[0]
            if imgs.dataoffset:
                self._imgs = imgs.asarray(out="memmap")
            else:
                self._imgs = imgs
        elif in_type == "raw":
            with open(f"{path}.info", "r") as fp:
                info = fp.readline().replace("\n", "")
            dtype, h, w, endian = info.split(",")
            h, w = int(h), int(w)
            endian = "<" if endian == "l" else ">"
            dtype = np.dtype(dtype).newbyteorder(endian)
            self._imgs = np.memmap(path, dtype, "r").reshape(-1, h, w)
        else:
            raise RuntimeError(f"{path} is not imgs file")

    @property
    def shape(self):
        return self._imgs.shape

    def dataset(self):
        nt, h, w = self._imgs.shape
        if hasattr(self._imgs[0], "asarray"):
            wrap = lambda x: x.asarray()
        else:
            wrap = lambda x: x
        imgs = tf.data.Dataset.from_generator(
            lambda: (wrap(img) for img in self._imgs),
            output_signature=tf.TensorSpec([h, w], dtype=tf.float32),
        )
        imgs.shape = nt, h, w
        return imgs
