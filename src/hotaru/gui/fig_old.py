import plotly.graph_objects as go


def heat_fig(title, colorscale="greens", scattercolor="red", zmin=None, zmax=None):
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


def spike_fig(title, hz, colorscale="reds"):
    spike = go.Heatmap(
        dx=1 / hz,
        colorscale=colorscale,
        zmin=0.0,
        zmax=1.0,
        # colorbar=dict(orientation="h", y=0, yanchor="top", thickness=10),
    )
    layout = go.Layout(
        title=title,
        xaxis=dict(title="time"),
        yaxis=dict(title="ID"),
        width=1000,
        height=570,
        margin=dict(t=25, b=45, l=0, r=0),
    )
    return go.Figure([spike], layout)


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


def update_img(fig, h, w, height=500):
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
