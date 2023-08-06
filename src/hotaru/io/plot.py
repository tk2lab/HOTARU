import numpy as np
import plotly.graph_objects as go
import plotly.io as pio

from hotaru.filter import gaussian_laplace
from hotaru.footprint import get_radius

pio.kaleido.scope.mathjax = None


def plot_simgs(simgs, scale=1, margin=30):
    simgs = np.stack(simgs)
    h, w = simgs[0].shape
    smin = simgs.min(axis=(1, 2), keepdims=True)
    smax = simgs.max(axis=(1, 2), keepdims=True)
    simgs = (simgs - smin) / (smax - smin)
    simgs = np.pad(simgs, ((0, 0), (0, 0), (0, 1)))
    simgs = np.concatenate(simgs, axis=1)
    simgs = np.pad(simgs, ((1, 1), (0, 1)))
    kwargs = dict(
        colorscale="Greens",
        showscale=False,
    )
    margin = dict(t=margin, b=margin, l=margin, r=margin)
    return plot_clip(simgs, h + 1, w + 1, scale, margin, **kwargs)


def plot_gl(data, idx, scale=1, margin=30, **radius):
    radius = get_radius(**radius)
    gl = gaussian_laplace(data.select(idx), radius, -2)
    gl = np.pad(gl, ((0, 0), (0, 1), (0, 0), (0, 1)))
    nt, h, nr, w = gl.shape
    gl = gl.reshape(nt * h, nr * w)
    gl = np.pad(gl, ((1, 0), (1, 0)))
    kwargs = dict(
        colorscale="Picnic",
        colorbar=dict(thicknessmode="pixels", thickness=10, xpad=0, ypad=0),
        zmin=-1,
        zmax=1,
    )
    margin = dict(t=margin, b=margin, l=margin, r=margin + 40)
    return plot_clip(gl, h, w, scale, margin, **kwargs)


def plot_peak_stats(peakval, peaks):
    def jitter(r):
        jitter = np.exp(0.2 * dscale * np.random.randn(r.size))
        return r * jitter

    dscale = np.log(peakval.radius[1] / peakval.radius[0])
    r = peakval.radius[peakval.r.ravel()]
    v = peakval.v.ravel()
    fig = go.Figure(
        data=[
            go.Scattergl(
                x=jitter(r),
                y=v,
                mode="markers",
                marker_opacity=0.1,
                name="all pixels",
            ),
            go.Scattergl(
                x=jitter(peaks.r),
                y=peaks.v,
                mode="markers",
                marker=dict(size=10, symbol="star", opacity=0.5),
                name="select",
            ),
        ],
        layout=go.Layout(
            template="none",
            width=1000,
            height=600,
            legend=dict(
                x=0.01,
                y=0.99,
                xanchor="left",
                yanchor="top",
            ),

        ),
    )
    fig.update_xaxes(
        title="radius",
        type="log",
    )
    fig.update_yaxes(
        title="intensity",
        range=(0, 1.05 * v.max()),
    )
    return fig


def plot_seg_max(seg, scale=1, margin=30):
    segmax = seg.max(axis=0)
    h, w = segmax.shape
    segmax = np.pad(segmax, ((1, 1), (1, 1)))
    kwargs = dict(
        colorscale="Greens",
        showscale=False,
    )
    margin = dict(t=margin, b=margin, l=margin, r=margin)
    return plot_clip(segmax, h + 1, w + 1, scale, margin, **kwargs)


def plot_seg(peaks, seg, mx=None, hsize=20, scale=1, margin=30):
    def s(i):
        return 1 + i * (size + 1)

    def e(i):
        return (i + 1) * (size + 1)

    n, h, w = seg.shape
    size = 2 * hsize + 1
    if mx is None:
        mx = int(np.floor(np.sqrt(n)))
    my = (n + mx - 1) // mx
    seg = np.pad(seg, ((0, 0), (hsize, hsize), (hsize, hsize)))
    clip = np.zeros((my * size + (my + 1), mx * size + (mx + 1)), np.float32)
    for i in range(n):
        j, k = divmod(i, mx)
        y = peaks.y[i]
        x = peaks.x[i]
        clip[s(j) : e(j), s(k) : e(k)] = seg[i, y : y + size, x : x + size]

    kwargs = dict(
        colorscale="Greens",
        showscale=False,
    )
    margin = dict(t=margin, b=margin, l=margin, r=margin)
    return plot_clip(clip, size + 1, size + 1, scale, margin, **kwargs)


def plot_calcium(trace, seg, hz, scale=0.3):
    nk, nt = trace.shape
    trace = trace / seg.sum(axis=(1, 2))[:, np.newaxis]
    trace *= -scale
    trace += np.arange(nk)[:, np.newaxis]
    fig = go.Figure(
        data=[
            go.Scatter(
                x=np.arange(nt) / hz,
                y=trace[k],
                mode="lines",
                line=dict(color="green", width=1),
            )
            for k in range(nk)
        ],
        layout=go.Layout(
            template="none",
            width=nt,
            height=nk * 5,
            showlegend=False,
        ),
    )
    fig.update_xaxes(
        title="time (sec)",
        showline=False,
    )
    fig.update_yaxes(
        title="cell ID",
        autorange="reversed",
        tickmode="linear",
        tick0=1,
        dtick=100,
        zeroline=False,
    )
    return fig


def plot_spike(spike, hz, scale=1, margin=30):
    nk, nt = spike.shape
    spike = spike / spike.max(axis=1, keepdims=True)
    fig = go.Figure(
        data=go.Heatmap(
            x=np.arange(nt) / hz,
            y=np.arange(nk) + 1,
            z=spike,
            colorscale="Reds",
            showscale=False,
        )
    )
    fig.update_xaxes(
        title="time (sec)",
        showline=False,
    )
    fig.update_yaxes(
        title="cell ID",
        autorange="reversed",
        tickmode="linear",
        tick0=1,
        dtick=100,
        zeroline=False,
    )
    return fig


def xgrid_data(h, w, size):
    return [
        go.Scatter(
            x=[x, x],
            y=[0, h - 1],
            mode="lines",
            line_color="black",
            line_width=1,
        )
        for x in range(0, w, size)
    ]


def ygrid_data(h, w, size):
    return [
        go.Scatter(
            x=[0, w - 1],
            y=[y, y],
            mode="lines",
            line_color="black",
            line_width=1,
        )
        for y in range(0, h, size)
    ]


def plot_clip(clip, xsize, ysize, scale, margin, **kwargs):
    h, w = clip.shape
    fig = go.Figure(
        data=[
            go.Heatmap(
                z=clip,
                **kwargs,
            ),
        ]
        + xgrid_data(h, w, xsize)
        + ygrid_data(h, w, ysize),
        layout=go.Layout(
            legend_visible=False,
            width=scale * w + margin["l"] + margin["r"],
            height=scale * h + margin["t"] + margin["b"],
            margin=margin | dict(autoexpand=False),
        ),
    )
    fig.update_xaxes(
        autorange=False,
        range=(-0.5, w - 0.5),
        showticklabels=False,
    )
    fig.update_yaxes(
        scaleanchor="x",
        autorange=False,
        range=(h - 0.5, -0.5),
        showticklabels=False,
    )
    return fig
