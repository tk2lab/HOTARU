import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from PIL import Image

from ..cui.common import load
from ..footprint import get_radius
from .common import to_image


def seg_max_image(cfg, stage, base=0, showbg=False):
    if stage == 0:
        segs = load(cfg, "make", stage)
        stats = load(cfg, "init", stage)
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

    size = 2 * hsize + 1

    if stage == 0:
        segs = load(cfg, "make", stage)
        stats = load(cfg, "init", stage)
    else:
        segs, stats = load(cfg, "clean", stage)
    stats = stats.query("kind == 'cell'")
    nk = stats.shape[0]
    fp = segs[:nk]
    ys = stats.y.to_numpy()
    xs = stats.x.to_numpy()

    fp = fp[select]
    ys = ys[select]
    xs = xs[select]
    nk = fp.shape[0]

    if mx is None:
        mx = int(np.floor(np.sqrt(nk)))
    my = (nk + mx - 1) // mx

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

    for i, (y, x) in enumerate(zip(ys, xs)):
        j, k = divmod(i, mx)
        clip[s(j) : e(j), s(k) : e(k)] = to_image(
            segs[i, y : y + size, x : x + size], "Greens",
        )
    return Image.fromarray(clip)


def jitter(r, radius, scale):
    dscale = np.log((radius[1:] / radius[:-1]).min())
    jitter = np.exp(scale * dscale * np.random.randn(r.size))
    return r * jitter


def footprint_stats_fig(cfg, stages, usefind=False, **kwargs):
    kwargs.setdefault("template", "none")
    kwargs.setdefault("margin", dict(l=40, r=10, t=20, b=35))
    kwargs.setdefault("showlegend", False)

    radius = get_radius(cfg.radius)
    rmin, rmax = radius[0], radius[-1]

    fig = go.Figure().set_subplots(2, len(stages), row_heights=(1, 2))
    vmax = 0
    for i, stage in enumerate(stages):
        if i == 0:
            intensity = "intensity"
            if usefind:
                peakval = load(cfg, "find", 0)
                ri = peakval.r
                cond = ri > 0
                rs = peakval.radius[ri[cond]]
                vs = peakval.v[cond]
                vmax = vs.max()
                fig.add_trace(
                    go.Scattergl(
                        x=jitter(rs, peakval.radius, 0.3),
                        y=vs,
                        mode="markers",
                        marker=dict(opacity=0.2, size=1, color="blue"),
                        name="all pixels",
                    ),
                    col=i + 1,
                    row=2,
                )
        else:
            intensity = "firmness"
        print(intensity)
        stats, _ = load(cfg, "evaluate", stage)
        cell = stats.query("kind == 'cell'")
        v = cell[intensity]
        vmax = max(v.max(), vmax)
        fig.add_trace(
            go.Scattergl(
                x=cell.radius, #jitter(rs),
                y=v,
                mode="markers",
                marker=dict(opacity=0.3, size=5, color="green"),
                name="all pixels",
            ),
            col=i + 1,
            row=2,
        )
        r, c = np.unique(cell.radius, return_counts=True)
        print(r)
        print(c)
        fig.add_trace(
            go.Bar(
                x=np.log(r),
                y=c,
                marker_color="green",
            ),
            col=i + 1,
            row=1,
        )
        fig.update_xaxes(
            showticklabels=False,
            #tickmode="array",
            #tickvals=[np.log(2), np.log(4), np.log(8)],
            #ticktext=["2", "4", "8"],
            range=[np.log(rmin), np.log(rmax)],
            col=i + 1,
            row=1,
        )
        print(rmin, rmax)
        fig.update_xaxes(
            title_text="radius",
            type="log",
            tickmode="array",
            tickvals=[3, 6, 12],
            ticktext=["3", "6", "12"],
            autorange=False,
            range=[np.log10(rmin), np.log10(rmax)],
            col=i + 1,
            row=2,
        )
        if i == 0:
            fig.update_yaxes(
                title_text="intensity",
                col=1,
                row=2,
                range=(0, 1.05 * vmax),
            )
            vmax = 0
    fig.update_yaxes(
        title_text="fimness",
        col=2,
        row=2,
    )
    for i in range(1, len(stats)):
        fig.update_yaxes(
            range=(0, 1.05 * vmax),
            col=i + 1,
            row=2,
        )
    fig.update_yaxes(
        title_text="count",
        col=1,
        row=1,
    )
    fig.update_layout(**kwargs)
    return fig
