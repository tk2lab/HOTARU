import dash.dcc as dcc
import dash.html as html
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from dash import (
    Input,
    Output,
    Patch,
    State,
    callback,
)
from omegaconf import OmegaConf

from ..main import (
    make,
    reduce,
)
from .graph import (
    img_fig,
    update_img_fig,
)
from .progress import Progress


def spike_fig():
    fig = go.Figure()
    return fig


def update_spike_fig(fig, spike):
    return fig


def create_init_div(cfg_store):
    footprint_graph = dcc.Graph(figure=img_fig("cell candidates"))
    spike_graph = dcc.Graph(figure=spike_fig())

    status_div = html.Div(
        style=dict(
            display="grid",
            gridTemplateColumns="auto 100px 1fr",
        ),
        children=[
            button := dbc.Button("INIT"),
            session := html.Div(),
            progress := dbc.Progress(value="0"),
        ],
    )

    graph_div = html.Div(
        style=dict(
            display="grid",
            columnGap="10px",
            gridTemplateColumns="auto auto",
        ),
        children=[
            footprint_graph,
            spike_graph,
        ],
    )

    div = html.Div(
        style=dict(
            width="1200px",
        ),
        children=[
            status_div,
            graph_div,
        ],
    )

    @callback(
        Output(footprint_graph, "figure"),
        Input(button, "n_clicks"),
        State(cfg_store, "data"),
        background=True,
        interval=100,
        running=[
            (Output(button, "disabled"), True, False),
        ],
        progress=[
            Output(session, "children"),
            Output(progress, "value"),
            Output(progress, "max"),
            Output(progress, "label"),
        ],
        prevent_initial_call=True,
    )
    def get_init(set_progress, n_cilcks, cfg):
        pbar = Progress(set_progress)
        _cfg = OmegaConf.create(cfg)
        peaks = reduce(_cfg, pbar)
        footprint = make(_cfg, pbar)
        footprint_fig = update_img_fig(Patch(), footprint.max(axis=0), peaks.x, peaks.y)
        pbar.session("finish")
        return footprint_fig

    return div
