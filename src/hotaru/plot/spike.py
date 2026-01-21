import numpy as np
import pandas as pd
import plotly.express as px
from PIL import Image
from PIL import ImageDraw

from ..cui.common import load
from ..spike import get_dynamics
from .common import to_image


def to_csv(cfg, stage, path, *, tsel=None, ksel=None):
    u, _, _, stats = load(cfg, 'temporal', stage)
    u = u[stats.query('kind == "cell"').spkid, :]

    dynamics = get_dynamics(cfg.dynamics)
    pad = dynamics.size - 1
    v = dynamics(u)
    u = u[:, pad:]

    if tsel is None:
        tsel = np.arange(u.shape[1])
    if ksel is None:
        ksel = np.arange(u.shape[0])

    udata = {'frame': tsel, 'time': tsel / cfg.data.imgs.hz}
    vdata = {'frame': tsel, 'time': tsel / cfg.data.imgs.hz}
    for k in ksel:
        udata[f'cell{k:04}'] = u[k, tsel]
        vdata[f'cell{k:04}'] = v[k, tsel]
    pd.DataFrame(udata).to_csv(path / 'spike.csv')
    pd.DataFrame(vdata).to_csv(path / 'calcium.csv')
    #_, _, _, stats = load(cfg, 'temporal', stage)
    #stats = stats.query("kind == 'cell'")
    ## u = u[stats.udense <= thr_udense]
    #return _spike_image(u, tsel, ksel, width, lines)


def spike_image(cfg, stage, tsel=slice(None), ksel=slice(None), width=3, lines=(), thr_udense=1.0):
    pad = get_dynamics(cfg.dynamics).size - 1
    u, _, _, stats = load(cfg, 'temporal', stage)
    u = u[stats.query('kind == "cell"').spkid, pad:]
    #_, _, _, stats = load(cfg, 'temporal', stage)
    #stats = stats.query("kind == 'cell'")
    # u = u[stats.udense <= thr_udense]
    return _spike_image(u, tsel, ksel, width, lines)


def _spike_image(u, tsel=slice(None), ksel=slice(None), width=3, lines=()):
    u /= u.max(axis=1, keepdims=True)
    u = u[ksel, tsel]
    nk, nt = u.shape
    img = to_image(u, 'Reds')
    img = Image.fromarray(img)
    draw = ImageDraw.Draw(img)
    draw.line(
        (int(0.8 * nt), (nk - 10), int(0.9 * nt), (nk - 10)),
        fill=(0, 0, 0),
        width=width,
    )
    for k0, k1, color, width in lines:
        if k0 < 0:
            k0 = nk + k0
        if k1 < 0:
            k1 = nk + k1
        draw.line((0, k0, 0, k1), fill=color, width=width)
    return img, nt, nk


def spike_stats_fig(cfg, stages, **kwargs):
    kwargs.setdefault('template', 'none')
    kwargs.setdefault('margin', {'l': 40, 'r': 10, 't': 20, 'b': 35})

    dfs = []
    for stage in stages:
        _, _, stats, _ = load(cfg, 'temporal', stage)
        stats['epoch'] = stage
        dfs.append(stats)
    stats = pd.concat(dfs, axis=0)
    fig = px.scatter(
        stats,
        x='signal',
        y='udense',
        opacity=0.3,
        facet_col='epoch',
        labels={'signal': 'signal intensity', 'udense': 'spike density'},
    )
    fig.update_layout(
        **kwargs,
    )
    return fig
