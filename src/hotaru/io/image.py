from pathlib import Path
from logging import getLogger

import numpy as np
from tifffile import (
    TiffFile,
    imread,
    memmap,
)

logger = getLogger(__name__)


def load_imgs(**cfg):
    logger.debug("load_imgs: %s", cfg)
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
                data = tif.series[0]
                if data.dataoffset is None:
                    logger.debug("create fixed tiff: %s", imgsfile_fix)
                    imgs = memmap(imgsfile_fix, shape=data.shape, dtype=data.dtype)
                    for i, pi in enumerate(data):
                        imgs[i] = pi.asarray()
                else:
                    imgs = data.asarray(out="memmap")
            logger.debug("mount tiff as memmap: %s", imgsfile_fix)
        case _:
            raise RuntimeError(f"{imgsfile} is not imgs file")
    return imgs, cfg["hz"]


def apply_mask(imgs, **cfg):
    logger.debug("apply_mask: %s", cfg)
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
        logger.debug("data size: (%d, %d, %d)", *imgs.shape)

    nt, h, w = imgs.shape
    nx = -1 if mask is None else h * w
    logger.info("imgs, mask: (%d, %d, %d), %d", nt, h, w, nx)
    return imgs, mask
