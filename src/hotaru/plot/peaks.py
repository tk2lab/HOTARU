import numpy as np
import plotly.graph_objects as go
import plotly.io as pio

from ..cui.common import load

pio.kaleido.scope.mathjax = None


def jitter(r, radius, scale):
    dscale = np.log((radius[1:] / radius[:-1]).min())
    jitter = np.exp(scale * dscale * np.random.randn(r.size))
    return r * jitter


def multi_peak_stats_fig(cfg, stage, stats, rmin, rmax):
    stats, _ = load(cfg, "evaluate", stage)
    fig = go.Figure().set_subplots(2, len(stats), row_heights=(1, 2))
    vmax = 0
    for i, peaks in enumerate(stats):
        cell = peaks.query("kind == 'cell'")
        if i == 0:
            intensity = "intensity"
        else:
            intensity = "firmness"
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
    return fig


def peak_stats_fig(cfg, stage, rmin=None, rmax=None, peakval=None):
    if stage == 0:
        stats = load(cfg, "init", stage)
    else:
        stats, _ = load(cfg, "evaluate", stage)
    if rmin is None:
        rmin = cfg.init.args.min_radius
    if rmax is None:
        rmax = cfg.init.args.max_radius
    peakval = None
    if stage == 0:
        try:
            peakval = load(cfg, "find", stage)
        except FileNotFoundError:
            pass

    fig = go.Figure().set_subplots(2, 1, row_heights=(1, 2))
    if peakval is None:
        intensity = "firmness"
        vmax = stats[intensity].max()
    else:
        intensity = "intensity"
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
            col=1,
            row=2,
        )
        """
        cs = [(ri == i).sum() for i in np.arange(radius.size)]
        fig.add_trace(
            go.Bar(
                x=np.log(radius),
                y=cs,
            ),
            col=1,
            row=1,
        )
        """
    cell = stats.query("kind == 'cell'")
    fig.add_trace(
        go.Scattergl(
            x=cell.radius, #jitter(rs),
            y=cell[intensity],
            mode="markers",
            marker=dict(opacity=0.3, size=5, color="green"),
            name="all pixels",
        ),
        col=1,
        row=2,
    )
    r, c = np.unique(cell.radius, return_counts=True)
    fig.add_trace(
        go.Bar(
            x=np.log(r),
            y=c,
        ),
        col=1,
        row=1,
    )
    fig.update_xaxes(
        showticklabels=False,
        #tickmode="array",
        #tickvals=[np.log(2), np.log(4), np.log(8)],
        #ticktext=["2", "4", "8"],
        range=[np.log(rmin), np.log(rmax)],
        col=1,
        row=1,
    )
    fig.update_xaxes(
        title_text="radius",
        type="log",
        tickmode="array",
        tickvals=[3, 6, 12],
        ticktext=["3", "6", "12"],
        autorange=False,
        range=[np.log10(rmin), np.log10(rmax)],
        col=1,
        row=2,
    )
    fig.update_yaxes(
        title_text=intensity,
        range=(0, 1.05 * vmax),
        col=1,
        row=2,
    )
    fig.update_yaxes(
        title_text="count",
        col=1,
        row=1,
    )
    return fig


def peak_stats_trace(peaks, peakval=None, label=""):

    radius = np.sort(np.unique(peaks.radius))
    dscale = np.log((radius[1:] / radius[:-1]).min())
    data = []
    if peakval is None:
        intensity = "firmness"
        vmax = peaks[intensity].max()
    else:
        intensity = "intensity"
        ri = peakval.r
        cond = ri > 0
        rs = peakval.radius[ri[cond]]
        vs = peakval.v[cond]
        vmax = vs.max()
        data.append(
            go.Scattergl(
                x=jitter(rs),
                y=vs,
                mode="markers",
                marker=dict(opacity=0.01, color="blue"),
                name="all pixels",
            )
        )
    cell = peaks[peaks.kind == "cell"]
    bg = peaks[peaks.kind == "background"]
    data += [
        go.Scattergl(
            x=jitter(cell.radius),
            y=cell[intensity],
            mode="markers",
            marker=dict(size=10, symbol="star", color="green", opacity=0.1),
            name="cell",
        ),
        go.Scattergl(
            x=jitter(bg.radius),
            y=bg[intensity],
            mode="markers",
            marker=dict(size=10, symbol="pentagon", color="red", opacity=0.5),
            name="background",
        ),
    ]
    return data
