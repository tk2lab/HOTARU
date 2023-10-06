import numpy as np
import plotly.graph_objects as go
from PIL import Image

from ..cui.common import load
from ..footprint import get_radius
from .common import to_image


def seg_max_image(cfg, stage, base=0.5, showbg=False, thr_udense=1.0):
    if stage == 0:
        segs = load(cfg, "make", stage)
    else:
        segs, _ = load(cfg, "clean", stage)
    stats, _ = load(cfg, "evaluate", stage)
    return _seg_max_image(segs, stats, base, showbg, thr_udense)


def _seg_max_image(segs, stats, base=0.5, showbg=False, thr_udense=1.0):
    #stats.loc[stats.udense > thr_udense, "kind"] = "background"
    cell = stats.query("kind == 'cell'").segid.to_numpy()
    bg = stats.query("kind == 'background'").segid.to_numpy()
    fp = segs[cell]
    fp = np.maximum(0, (fp - base) / (1 - base)).max(axis=0)
    fpimg = to_image(fp, "Greens")
    fpimg = Image.fromarray(fpimg)
    if showbg:
        bg = segs[bg].max(axis=0)
        bgimg = to_image(bg, "Reds")
        bgimg[:, :, 3] = 128
        fpimg = Image.fromarray(bgimg)
        fpimg.paset(bgimg, (0, 0), bgimg)
    return fpimg


def seg_max_fig(cfg, stage, base=0.5, showbg=False, width=600, thr_udense=1.0):
    if stage == 0:
        segs = load(cfg, "make", stage)
        stats = load(cfg, "init", stage)
    else:
        _, segs = load(cfg, "clean", stage)
        stats = load(cfg, "evaluate", stage)
    return _seg_max_fig(segs, stats, base, showbg, width, thr_udense)


def _seg_max_fig(segs, stats, base=0.5, showbg=False, width=600, thr_udense=1.0):
    img = _seg_max_image(segs, stats, base, showbg, thr_udense)
    #stats.loc[stats.udense > thr_udense, "kind"] = "background"
    cell = stats.query("kind == 'cell'")
    fig = go.Figure()
    fig.add_trace(
        go.Image(z=img),
    )
    fig.add_trace(
        go.Scatter(
            x=cell.x,
            y=cell.y,
            text=cell.index,
            mode="text",
            textfont=dict(size=5, color="red"),
        ),
    )
    w, h = img.size
    fig.update_xaxes(
        showticklabels=False,
        domain=[0, 1],
        autorange=False,
        range=(0, w),
    )
    fig.update_yaxes(
        showticklabels=False,
        domain=[0, 1],
        autorange=False,
        range=(h, 0),
    )
    fig.update_layout(
        width=width,
        height=int(width * h / w),
        margin=dict(l=0, r=0, b=0, t=0),
    )
    return fig


def bg_sum_image(cfg, stage):
    if stage == 0:
        segs = load(cfg, "make", stage)
    else:
        segs, _ = load(cfg, "clean", stage)
    stats, _ = load(cfg, "evaluate", stage)
    st = stats.query("kind == 'background'")
    bg = segs[st.index]
    bg *= st.bmax.to_numpy()[:, np.newaxis, np.newaxis]
    bgsum = bg.sum(axis=0)
    bgsum /= bgsum.max()
    img = to_image(bgsum, "Reds")
    return Image.fromarray(img)


def segs_image(cfg, stage, select=None, mx=None, hsize=20, pad=5, thr_udense=1.0):
    if stage == 0:
        segs = load(cfg, "make", stage)
        stats = load(cfg, "init", stage)
    else:
        segs, stats = load(cfg, "clean", stage)
    return _segs_image(segs, stats, select, mx, hsize, pad, thr_udense)


def _segs_image(segs, stats, select=None, mx=None, hsize=20, pad=5, thr_udense=1.0):
    def s(i):
        return pad + i * (size + pad)

    def e(i):
        return (i + 1) * (size + pad)

    size = 2 * hsize + 1

    stats.loc[stats.udense > thr_udense, "kind"] = "background"
    stats = stats.query("kind == 'cell'")

    nk = stats.shape[0]
    fp = segs[stats.index]
    ys = stats.y.to_numpy()
    xs = stats.x.to_numpy()

    if select is None:
        select = range(nk)
    else:
        fp = fp[select]
        ys = ys[select]
        xs = xs[select]
        nk = fp.shape[0]

    if mx is None:
        mx = int(np.floor(np.sqrt(nk)))
    my = (nk + mx - 1) // mx

    fp = np.pad(fp, ((0, 0), (hsize, hsize), (hsize, hsize)))
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

    for i, (img, y, x) in enumerate(zip(fp, ys, xs)):
        u, w = divmod(i, mx)
        clip[s(u) : e(u), s(w) : e(w)] = to_image(
            img[y : y + size, x : x + size],
            "Greens",
        )
    return Image.fromarray(clip)


def jitter(r):
    radius = np.sort(np.unique(r))
    rscale = np.log((radius[1:] / radius[:-1]).min())
    jitter = np.exp(rscale * (np.random.uniform(size=r.size) - 0.5))
    return r * jitter


def footprint_stats_fig(cfg, stages, usefind=False, **kwargs):
    kwargs.setdefault("template", "none")
    kwargs.setdefault("margin", dict(l=40, r=10, t=20, b=35))
    kwargs.setdefault("showlegend", False)

    radius = get_radius(cfg.radius.filter)
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
                        x=jitter(rs),
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
        stats, _ = load(cfg, "evaluate", stage)
        cell = stats.query("kind == 'cell'")
        v = cell[intensity]
        print(intensity, v)
        vmax = max(v.max(), vmax)
        fig.add_trace(
            go.Scattergl(
                x=cell.radius if usefind else jitter(cell.radius),
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
            # tickmode="array",
            # tickvals=[np.log(2), np.log(4), np.log(8)],
            # ticktext=["2", "4", "8"],
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
    print(vmax)
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
