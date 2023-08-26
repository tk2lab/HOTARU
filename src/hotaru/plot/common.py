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


def add_jitter(df):
    radius = np.sort(np.unique(df.radius))
    rscale = np.log(np.min(radius[1:] / radius[:-1]))
    df["ri"] = df.radius * np.exp(rscale * (np.random.uniform(size=df.shape[0]) - 0.5))
    df["lri"] = np.log2(df.ri)
    return df
