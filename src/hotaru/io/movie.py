import av
import numpy as np
import matplotlib.pyplot as plt

from ..utils import get_progress


def gen_movie(filename, imgs, shape, fps, pbar=None, **kwargs):
    pbar = get_progress(pbar)
    pbar = pbar.session("movie")
    pbar.set_count(shape[0])
    with av.open(filename, "w") as output:
        stream = output.add_stream(kwargs.get("codec", "h264"), fps)
        stream.bit_rate = kwargs.get("bit_rate", 8_000_000)
        stream.pix_fmt = kwargs.get("fmt", "yuv420p")
        stream.height = shape[1]
        stream.width = shape[2]
        for img in imgs:
            frame = av.VideoFrame.from_ndarray(img, format="rgba")
            packet = stream.encode(frame)
            output.mux(packet)
            if pbar is not None:
                pbar.update(1)


def gen_normalize_movie(filename, data, **kwargs):
    imgs, mask, hz, avgx, avgt, std0, min0, max0, min1, max1 = data
    nt, h, w = imgs.shape
    def gen_frame():
        for t in range(nt):
            img0 = (imgs[t] - min0) / (max0 - min0)
            img1 = (imgs[t] - avgx - avgt[t]) / std0
            img1 = (img1 - min1) / (max1 - min1)
            img = np.concatenate([img0, img1], axis=1)
            cimg = plt.get_cmap("Greens")(img)
            if mask is not None:
                cimg[~mask] = 0
            yield (255 * cimg).astype(np.uint8)
    gen_movie(filename, gen_frame(), (nt, h, 2 * w), hz, **kwargs)
