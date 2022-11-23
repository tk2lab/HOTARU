import numpy as np
import tensorflow as tf
from tqdm.auto import tqdm
from matplotlib.font_manager import fontManager
from matplotlib.pyplot import get_cmap
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont

from ..io.mpeg import MpegStream
from ..io.numpy import load_numpy
from ..io.pickle import load_pickle
from ..io.tfrecord import load_tfrecord
from ..train.dynamics import SpikeToCalcium
from ..util.dataset import unmasked


def make_mpeg(model, outfile, has_truth=False):
    cmap = get_cmap("Greens")
    font = ImageFont.truetype(fontManager.findfont("DejaVu Sans"), size=20)

    imgs = model.raw_imgs
    avgx, avgt, std = model.stats
    avg = avgt.mean()
    nt, h, w = imgs.shape
    mask = model.mask
    hz = model.hz

    if has_truth:
        a0 = load_numpy("truth/a0.npy")
        a0 = a0 > 0.5
        v0 = load_numpy("truth/v0.npy")
        v0 -= v0.min(axis=1, keepdims=True)
        v0 /= smax - smin
        v0 *= 2.0

    val = model.footprint.val
    nk = val.shape[0]
    a2 = np.empty((nk, h, w))
    a2[:, mask] = val
    u2 = model.spike.val
    with tf.device("CPU"):
        v2 = model.spike_to_calcium(u2).numpy()
    v2 -= v2.min(axis=1, keepdims=True)

    if has_truth:
        mpeg = MpegStream(3 * w, h, hz, outfile)
    else:
        mpeg = MpegStream(2 * w, h, hz, outfile)
    with mpeg:
        for t, d in tqdm(enumerate(imgs.as_numpy_iterator()), total=nt):
            val = (d - avg) / std
            val = (val + 3.0) / 10.0
            imgo = np.clip(val, 0.0, 1.0)
            val = (a2 * v2[:, t, None, None]).sum(axis=0)
            val = (val + 3.0) / 10.0
            img2 = np.clip(val, 0.0, 1.0)
            if has_truth:
                val = (a0 * v0[:, t, None, None]).sum(axis=0),
                img0 = np.clip(val, 0.0, 1.0)
                img = np.concatenate([imgo, img0, img2], axis=1)
            else:
                img = np.concatenate([imgo, img2], axis=1)
            img = (255 * cmap(img)).astype(np.uint8)
            img = Image.fromarray(img)
            draw = ImageDraw.Draw(img)
            draw.text((10, 10), "orig", font=font)
            if has_truth:
                draw.text((10 + w, 10), "true", "black", font=font)
                draw.text((10 + 2 * w, 10), "HOTARU", "black", font=font)
            else:
                draw.text((10 + w, 10), "HOTARU", font=font)
            mpeg.write(np.asarray(img))
