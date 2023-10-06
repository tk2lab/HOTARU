import numpy as np
import plotly.graph_objects as go
import plotly.io as pio

from hotaru.filter import gaussian_laplace
from hotaru.footprint import get_radius

pio.kaleido.scope.mathjax = None


def plot_imgs_subplots(imgs, labels, width=500, pad=0.05):
    margin = dict(l=10, r=10, t=20, b=10)
    num = len(imgs)
    xdomainsize = (1 - (num - 1) * pad) / num
    xstride = xdomainsize + pad
    fig = go.Figure().set_subplots(1, len(imgs), subplot_titles=labels)
    for i, (img, label) in enumerate(zip(imgs, labels)):
        h, w = img.shape
        smin = img.min()
        smax = img.max()
        img = (img - smin) / (smax - smin)
        fig.add_trace(
            go.Heatmap(z=img, colorscale="Greens", showscale=False),
            col=i + 1,
            row=1,
        )
        fig.update_xaxes(
            autorange=False,
            range=(-0.5, w - 0.5),
            showticklabels=False,
            domain=[i * xstride, i * xstride + xdomainsize],
            col=i + 1,
            row=1,
        )
        fig.update_yaxes(
            scaleanchor="x",
            autorange=False,
            domain=[0, 1],
            range=(h, 0),
            showticklabels=False,
            col=i + 1,
            row=1,
        )
    mod_w = num * w * (1 + (num -1) * pad)
    mod_width = width - margin["l"] - margin["r"]
    fig.update_layout(
        annotations=[dict(font=dict(size=11)) for _ in range(num)],
        width=width,
        height=np.ceil(margin["t"] + margin["b"] + mod_width * h / mod_w),
        margin=margin,
    )
    return fig


def plot_simgs(simgs, labels, scale=1, margin=30, label=None):
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
    data = [go.Heatmap(z=simgs, **kwargs)]
    fig = plot_clip(data, simgs.shape, h + 1, w + 1, scale, margin)
    for i, label in enumerate(labels):
        fig.add_annotation(x=10 + i * (w + 1), y=10, text=label, showarrow=False)
    return fig


def plot_gl(data, radius, idx, scale=1, margin=30, label=""):
    radius = get_radius(radius)
    imgs = data.select(idx)
    gl = gaussian_laplace(imgs, radius, 0)
    gl = np.transpose(gl, (1, 2, 0, 3))
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
    data = [go.Heatmap(z=gl, **kwargs)]
    return plot_clip(data, gl.shape, h, w, scale, margin)


def plot_peak_stats(peaks, peakval=None, label=""):
    def jitter(r):
        jitter = np.exp(0.2 * dscale * np.random.randn(r.size))
        return r * jitter

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
                marker=dict(opacity=0.03, color="blue"),
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
            marker=dict(size=10, symbol="star", color="lime", opacity=0.5),
            name="cell",
        ),
        go.Scattergl(
            x=jitter(bg.radius),
            y=bg[intensity],
            mode="markers",
            marker=dict(size=10, symbol="pentagon", color="gray", opacity=0.5),
            name="background",
        ),
    ]

    fig = go.Figure(
        data=data,
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
        title=intensity,
        range=(0, 1.05 * vmax),
    )
    return fig


def plot_seg_max(
    footprints, peaks, base=0.5, plot_bg=True, scale=1, margin=30, label=""
):
    nk = np.count_nonzero(peaks.kind == "cell")
    seg = footprints[:nk]
    bg = footprints[nk:]

    margin = dict(t=margin, b=margin, l=margin, r=margin)
    nk, h, w = seg.shape

    seg = np.maximum(0, (seg - base) / (1 - base))
    segmax = seg.max(axis=0)
    segmax = np.pad(segmax, ((1, 1), (1, 1)))
    data = [
        go.Heatmap(z=segmax, colorscale="Greens", showscale=False, opacity=0.8),
    ]
    if plot_bg and bg.shape[0] > 0:
        bgmax = bg.max(axis=0)
        bgmax = np.pad(bgmax, ((1, 1), (1, 1)))
        data += [
            go.Heatmap(z=bgmax, colorscale="Reds", showscale=False, opacity=0.2),
        ]
    return plot_clip(data, segmax.shape, h + 1, w + 1, scale, margin)


def plot_seg(seg, peaks, mx=None, hsize=20, scale=1, margin=30, label=""):
    def s(i):
        return 1 + i * (size + 1)

    def e(i):
        return (i + 1) * (size + 1)

    peaks = peaks[peaks.kind == "cell"]
    nk = peaks.shape[0]
    seg = seg[:nk]

    size = 2 * hsize + 1
    if mx is None:
        mx = int(np.floor(np.sqrt(nk)))
    my = (nk + mx - 1) // mx
    seg = np.pad(seg, ((0, 0), (hsize, hsize), (hsize, hsize)))
    clip = np.zeros((my * size + (my + 1), mx * size + (mx + 1)), np.float32)

    peaks = peaks[peaks.kind == "cell"]
    ys = np.array(peaks.y)
    xs = np.array(peaks.x)
    for i, (y, x) in enumerate(zip(ys, xs)):
        j, k = divmod(i, mx)
        clip[s(j) : e(j), s(k) : e(k)] = seg[i, y : y + size, x : x + size]

    kwargs = dict(
        colorscale="Greens",
        showscale=False,
    )
    margin = dict(t=margin, b=margin, l=margin, r=margin)
    data = [go.Heatmap(z=clip, **kwargs)]
    return plot_clip(data, clip.shape, size + 1, size + 1, scale, margin)


def plot_calcium(trace, seg, hz, scale=0.3, margin=30, label=""):
    nk, nt = trace.shape
    trace = trace / seg.sum(axis=(1, 2))[:, np.newaxis]
    trace *= -scale
    trace += np.arange(nk)[:, np.newaxis]
    margin = dict(t=margin, b=60 + margin, l=50 + margin, r=margin)
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
            width=nt + margin["l"] + margin["r"],
            height=nk * 3 + margin["t"] + margin["b"],
            margin=margin | dict(autoexpand=False),
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
        tickmode="array",
        tickvals=[1] + list(range(100, nk, 100)) + [nk],
        zeroline=False,
    )
    return fig


def plot_spike(spike, hz, diff, time, scale=1, margin=30, label=""):
    spmax = spike.max(axis=1, keepdims=True)
    if time is not None:
        spike = spike[:, slice(*time)]
    spike = spike / np.where(spmax > 0, spmax, 1)
    nk, nt = spike.shape
    margin = dict(t=margin, b=60 + margin, l=50 + margin, r=margin)
    if time is None:
        time = np.arange(nt)
    else:
        time = np.arange(*time)
    fig = go.Figure(
        data=go.Heatmap(
            x=(time - diff) / hz,
            y=np.arange(nk) + 1,
            z=spike,
            colorscale="Reds",
            showscale=False,
        ),
        layout=go.Layout(
            template="none",
            width=nt + margin["l"] + margin["r"],
            height=nk * 3 + margin["t"] + margin["b"],
            margin=margin | dict(autoexpand=False),
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
        tickmode="array",
        tickvals=[1] + list(range(100, nk, 100)) + [nk],
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


def plot_clip(data, shape, ysize, xsize, scale, margin):
    h, w = shape
    fig = go.Figure(
        data + xgrid_data(h, w, xsize) + ygrid_data(h, w, ysize),
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
