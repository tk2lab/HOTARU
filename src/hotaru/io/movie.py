from logging import getLogger

import av
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageFont, ImageDraw

logger = getLogger(__name__)


def gen_movie(filename, imgs, shape, fps, **kwargs):
    logger.info("%s: %s %s %d", "pbar", "start", "movie", shape[0])
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
            logger.info("%s: %s %d", "pbar", "update", 1)
        logger.info("%s: %s", "pbar", "close")


def gen_normalize_movie(outfile, data, **kwargs):
    imgs, mask, hz, avgx, avgt, std0, min0, max0, min1, max1 = data
    nt, h, w = imgs.shape
    def gen_frame():
        for t in range(nt):
            img0 = (imgs[t] - min0) / (max0 - min0)
            img1 = (imgs[t] - avgx - avgt[t]) / std0
            img1 = (img1 - min1) / (max1 - min1)
            img = np.concatenate([img0, img1], axis=1)
            img = (255 * plt.get_cmap("Greens")(img)).astype(np.uint8)
            if mask is not None:
                img[~mask] = 0
            yield (255 * img).astype(np.uint8)
    gen_movie(outfile, gen_frame(), (nt, h, 2 * w), hz, **kwargs)


def gen_result_movie(outfile, data, footprints, spikes, dynamics, t0, t1, **kwargs):
    calcium = np.array(dynamics(spikes))
    print(calcium.shape)
    def gen_frame():
        for t in range(t0, t1):
            img0 = (data.select(t) - data.min1) / (data.max1 - data.min1)

            img1 = (calcium[:, t, np.newaxis, np.newaxis] * footprints).sum(axis=0)
            img1 = (img1 - data.min1) / (data.max1 - data.min1)

            img = np.concatenate([img0, img1], axis=1)
            img = (255 * plt.get_cmap("Greens")(img)).astype(np.uint8)

            img = Image.fromarray(img)
            draw = ImageDraw.Draw(img)
            draw.text((10, 10), "Normalized original movie", "red", font=font)
            draw.text((w + 10, 10), "Reconstructed movie", "red", font=font)

            yield np.array(img)

    font = ImageFont.truetype("/usr/share/fonts/truetype/ttf-bitstream-vera/VeraBd.ttf", 24)
    nk, _ = calcium.shape
    _, h, w = footprints.shape
    footprints = footprints[:nk]
    gen_movie(outfile, gen_frame(), (t1 - t0, h, 2 * w), data.hz, **kwargs)
