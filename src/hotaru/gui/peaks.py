import dash.dcc as dcc
import dash.html as html
import dash_bootstrap_components as dbc
from dash import (
    Input,
    Output,
    Patch,
    State,
    callback,
)
from omegaconf import OmegaConf

from ..main import (
    find,
    reduce,
)
from .graph import (
    circle_fig,
    radius_fig,
    update_circle_fig,
    update_radius_fig,
)
from .progress import Progress


def create_peaks_div(cfg_store):
    circle_graph = dcc.Graph(figure=circle_fig("cell candidates"))
    radius_graph = dcc.Graph(figure=radius_fig("", "radius", "indtensity"))

    status_div = html.Div(
        style=dict(
            display="grid",
            gridTemplateColumns="auto 100px 1fr",
        ),
        children=[
            button := dbc.Button("PEAKS"),
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
            circle_graph,
            radius_graph,
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
        Output(circle_graph, "figure"),
        Output(radius_graph, "figure"),
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
    def get_peaks(set_progress, n_cilcks, cfg):
        pbar = Progress(set_progress)
        _cfg = OmegaConf.create(cfg)
        peakval = find(_cfg, pbar)
        peaks = reduce(_cfg, pbar)
        circle_fig = update_circle_fig(Patch(), peakval, peaks)
        radius_fig = update_radius_fig(Patch(), peakval, peaks)
        pbar.session("finish")
        return circle_fig, radius_fig

    return div
