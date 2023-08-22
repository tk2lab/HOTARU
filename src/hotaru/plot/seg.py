import numpy as np
from PIL import Image

from ..cui.common import load
from .common import to_image


def seg_max_image(cfg, stage, base=0, showbg=False):
    if stage == 0:
        segs, stats = load(cfg, "make", stage)
    else:
        segs, stats = load(cfg, "clean", stage)
    nk = np.count_nonzero(stats.kind == "cell")
    fp = segs[:nk]
    fp = np.maximum(0, (fp - base) / (1 - base)).max(axis=0)
    fpimg = to_image(fp, "Greens")
    fpimg = Image.fromarray(fpimg)
    if showbg:
        bg = segs[nk:].max(axis=0)
        bgimg = to_image(bg, "Reds")
        bgimg[:, :, 3] = 128
        fpimg = Image.fromarray(bgimg)
        fpimg.paset(bgimg, (0, 0), bgimg)
    return fpimg


def segs_image(cfg, stage, select=slice(None), mx=None, hsize=20, pad=5):
    def s(i):
        return pad + i * (size + pad)

    def e(i):
        return (i + 1) * (size + pad)

    if stage == 0:
        segs, stats = load(cfg, "make", stage)
    else:
        segs, stats = load(cfg, "clean", stage)
    nk = np.count_nonzero(stats.kind == "cell")
    fp = segs[:nk]
    fp = segs[select]
    nk = fp.shape[0]

    if mx is None:
        mx = int(np.floor(np.sqrt(nk)))
    my = (nk + mx - 1) // mx

    size = 2 * hsize + 1
    segs = np.pad(segs, ((0, 0), (hsize, hsize), (hsize, hsize)))
    clip = np.zeros(
        (my * size + pad * (my + 1), mx * size + pad * (mx + 1), 4), np.uint8
    )

    for x in range(mx + 1):
        st = x * (size + pad)
        en = st + pad
        clip[:, st:en] = [0, 0, 0, 255]
    for y in range(my + 1):
        st = y * (size + pad)
        en = st + pad
        clip[st:en] = [0, 0, 0, 255]

    ys = stats.y.to_numpy()
    xs = stats.x.to_numpy()
    for i, (y, x) in enumerate(zip(ys, xs)):
        j, k = divmod(i, mx)
        clip[s(j) : e(j), s(k) : e(k)] = to_image(
            segs[i, y : y + size, x : x + size], "Greens",
        )
    return Image.fromarray(clip)
