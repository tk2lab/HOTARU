import numpy as np
import plotly.graph_objects as go


def heat_fig(title, colorscale="greens", zmin=None, zmax=None):
    heat = go.Heatmap(
        colorscale=colorscale, zmin=zmin, zmax=zmax,
        colorbar=dict(orientation="h", y=0, yanchor="top", thickness=10),
    )
    scatter = go.Scatter(mode="markers")
    layout = go.Layout(
        title=title,
        xaxis=dict(visible=False, autorange=False),
        yaxis=dict(visible=False, autorange=False),
        width=600,
        height=570,
        margin=dict(t=25, b=45, l=0, r=0),
    )
    return go.Figure([heat, scatter], layout)


def circle_fig(title):
    scatter = go.Scatter(mode="markers", marker=dict(color="green"))
    layout = go.Layout(
        title=title,
        #xaxis=dict(visible=False, autorange=False),
        #yaxis=dict(visible=False, autorange=False),
        xaxis=dict(autorange=False),
        yaxis=dict(autorange=False),
        width=600,
        height=570,
        margin=dict(t=25, b=45, l=0, r=0),
    )
    return go.Figure([scatter], layout)


def update_img(fig, h, w):
    fig.layout.width = 500 * w / h
    fig.layout.xaxis.range = [0, w]
    fig.layout.yaxis.range = [h, 0]
    return fig


def scatter_fig(title, xlabel, ylabel):
    scatter = go.Scatter(mode="markers")
    layout = go.Layout(
        title=title,
        xaxis_title=xlabel,
        yaxis_title=ylabel,
        width=600,
        height=570,
    )
    return go.Figure([scatter], layout)


def bar_fig(title, xlabel, ylabel):
    bar = go.Bar()
    layout = go.Layout(
        title=title,
        xaxis_title=xlabel,
        yaxis_title=ylabel,
        width=600,
        height=570,
    )
    return go.Figure([bar], layout)
