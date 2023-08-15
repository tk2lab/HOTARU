import numpy as np
import plotly.graph_objects as go


def img_fig(title, colorscale="greens", scattercolor="red", zmin=None, zmax=None):
    heat = go.Heatmap(
        colorscale=colorscale,
        zmin=zmin,
        zmax=zmax,
        colorbar=dict(orientation="h", y=0, yanchor="top", thickness=10),
    )
    scatter = go.Scatter(mode="markers", marker_color=scattercolor)
    layout = go.Layout(
        title=title,
        xaxis=dict(visible=False, autorange=False),
        yaxis=dict(visible=False, autorange=False),
        width=600,
        height=570,
        margin=dict(t=25, b=45, l=0, r=0),
    )
    return go.Figure([heat, scatter], layout)


def update_img_fig(fig, img, x, y, height=500):
    h, w = img.shape
    fig.data[0].z = img
    fig.data[0].zmin = img.min()
    fig.data[0].zmax = img.max()
    fig.data[1].x = x
    fig.data[1].y = y
    fig.layout.width = height * w / h
    fig.layout.height = height + 70
    fig.layout.xaxis.range = [0, w]
    fig.layout.yaxis.range = [h, 0]
    return fig


def scatter_fig(title, xlabel, ylabel):
    scatter = go.Scatter(mode="markers", showlegend=False)
    layout = go.Layout(
        title=title,
        xaxis_title=xlabel,
        yaxis_title=ylabel,
        width=600,
        height=570,
    )
    return go.Figure([scatter, scatter], layout)


def update_scatter_fig(fig, istd, icor, x, y):
    fig.data[0].x = istd[y, x]
    fig.data[0].y = icor[y, x]
    return fig


def radius_fig(title, xlabel, ylabel):
    fig = scatter_fig(title, xlabel, ylabel)
    fig.data[0].marker.maxdisplayed = 10000
    fig.data[0].marker.opacity = 0.3
    fig.data[1].marker.opacity = 0.3
    fig.layout.xaxis.type = "log"
    return fig


def update_radius_fig(fig, peakval, peaks):
    dr = peakval.radius[1] / peakval.radius[0] - 1
    v0 = peakval.v.ravel()
    r0 = np.array(peakval.radius)[peakval.r.ravel()]
    jitter0 = 1 + 0.2 * dr * np.random.randn(v0.size)
    jitter = 1 + 0.2 * dr * np.random.randn(peaks.r.size)
    fig.data[0].x = jitter0 * r0
    fig.data[0].y = v0
    fig.data[1].x = jitter * peaks.r
    fig.data[1].y = peaks.v
    return fig


def circle_fig(title):
    scatter = go.Scatter(mode="markers", marker=dict(color="green", sizeref=10.0))
    layout = go.Layout(
        title=title,
        xaxis=dict(visible=False, autorange=False),
        yaxis=dict(visible=False, autorange=False),
        width=600,
        height=570,
        margin=dict(t=25, b=45, l=0, r=0),
    )
    return go.Figure([scatter], layout)


def update_circle_fig(fig, peakval, peaks, height=500):
    h, w = peakval.v.shape
    fig.data[0].x = peaks.x
    fig.data[0].y = peaks.y
    fig.data[0].marker.opacity = 0.5 * peaks.v / peaks.v.max()
    fig.data[0].marker.size = peaks.r
    fig.data[0].marker.sizeref = 0.25 * h / 500
    fig.data[0].marker.sizemode = "diameter"
    fig.layout.width = height * w / h
    fig.layout.height = height + 70
    fig.layout.xaxis.range = [0, w]
    fig.layout.yaxis.range = [h, 0]
    return fig
