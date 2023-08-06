from pathlib import Path

import numpy as np
from tifffile import (
    TiffFile,
    imread,
    memmap,
)


def load_imgs(**cfg):
    match cfg["type"]:
        case "npy":
            imgs = np.load(cfg["file"], mmap_mode="r")
        case "raw":
            dtype = np.dtype(cfg["dtype"]).newbyteorder(cfg["endian"])
            data = np.memmap(cfg["file"], dtype, "r")
            imgs = data.reshape(-1, cfg["height"], cfg["width"])
        case "tif":
            imgsfile = Path(cfg["file"])
            imgsfile_fix = imgsfile.with_stem(f"{imgsfile.stem}_fix")
            if imgsfile_fix.exists():
                imgsfile = imgsfile_fix
            with TiffFile(imgsfile) as tif:
                imgs = tif.series[0]
                if imgs.dataoffset is None:
                    imgs = memmap(imgsfile_fix, shape=imgs.shape, dtype=imgs.dtype)
                    for i, pi in enumerate(imgs):
                        imgs[i] = pi.asarray()
                else:
                    imgs = imgs.asarray(out="memmap")
        case _:
            raise RuntimeError(f"{imgsfile} is not imgs file")
    return imgs, cfg["hz"]


def apply_mask(imgs, **cfg):
    match cfg["type"]:
        case "nomask":
            mask = None
        case "tif":
            mask = imread(cfg["file"])
        case "npy":
            mask = np.load(cfg["file"])
        case _:
            raise RuntimeError("bad file type: {maskfile}")

    if mask is not None:
        my = np.where(np.any(mask, axis=1))[0]
        mx = np.where(np.any(mask, axis=0))[0]
        x0, y0, w, h = mx[0], my[0], mx[-1] - mx[0] + 1, my[-1] - my[0] + 1
        imgs = imgs[:, y0 : y0 + h, x0 : x0 + w]
        mask = mask[y0 : y0 + h, x0 : x0 + w]

    return imgs, mask
