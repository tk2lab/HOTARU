import numpy as np
from plotly.colors import get_colorscale
from matplotlib.pyplot import get_cmap


def cmap(colorscale, value):
    mask = np.zeros_like(value, bool)
    out = np.empty_like(value, str)
    for thr, color in get_colorscale(colorscale):
        cond = mask & (value < thr)
        out[cond] = color
        mask |= cond
    return out


def to_image(data, cmap):
    return (255 * get_cmap(cmap)(data)).astype(np.uint8)
