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

from ..footprint.find import simple_peaks
from ..main import stats
from .graph import (
    img_fig,
    scatter_fig,
    update_img_fig,
    update_scatter_fig,
)
from .progress import Progress


def create_stats_div(cfg_store):
    gaussian = dcc.Slider(
        min=0,
        max=10,
        step=0.1,
        value=1,
        marks={i: str(i) for i in range(11)},
    )
    maxpool = dcc.Slider(
        min=3,
        max=11,
        step=2,
        value=3,
    )

    std_graph = dcc.Graph(figure=img_fig("std"))
    cor_graph = dcc.Graph(figure=img_fig("cor"))
    std_cor_graph = dcc.Graph(figure=scatter_fig("", "std", "cor"))

    status_div = html.Div(
        style=dict(
            # width="1200px",
            display="grid",
            gridTemplateColumns="auto 100px 1fr",
        ),
        children=[
            get_stats := dbc.Button("STATS"),
            session := html.Div(),
            progress := dbc.Progress(value="0"),
        ],
    )

    control_div = html.Div(
        style=dict(
            # width="1200px",
            display="grid",
            gridTemplateColumns="auto 1fr",
        ),
        children=[
            dbc.Label("Gaussian"),
            gaussian,
            dbc.Label("MaxPool"),
            maxpool,
        ],
    )

    graph_div = html.Div(
        style=dict(
            # width="1200px",
            display="grid",
            columnGap="10px",
            gridTemplateColumns="auto auto auto",
        ),
        children=[
            std_graph,
            cor_graph,
            std_cor_graph,
        ],
    )

    stats_div = html.Div(
        style=dict(
            width="1200px",
        ),
        children=[
            status_div,
            control_div,
            graph_div,
        ],
    )

    @callback(
        Output(std_graph, "figure"),
        Output(cor_graph, "figure"),
        Output(std_cor_graph, "figure"),
        Input(get_stats, "n_clicks"),
        Input(gaussian, "value"),
        Input(maxpool, "value"),
        State(cfg_store, "data"),
        background=True,
        interval=100,
        running=[
            (Output(get_stats, "disabled"), True, False),
            (Output(gaussian, "disabled"), True, False),
            (Output(maxpool, "disabled"), True, False),
        ],
        progress=[
            Output(session, "children"),
            Output(progress, "value"),
            Output(progress, "max"),
            Output(progress, "label"),
        ],
        prevent_initial_call=True,
    )
    def get_stats(set_progress, n_cilcks, gaussian, maxpool, cfg):
        pbar = Progress(set_progress)
        _cfg = OmegaConf.create(cfg)
        _stats = stats(_cfg, pbar)
        y, x = simple_peaks(_stats.icor, gaussian, maxpool, pbar.session("peaks"))
        std_fig = update_img_fig(Patch(), _stats.istd, x, y)
        cor_fig = update_img_fig(Patch(), _stats.icor, x, y)
        std_cor_fig = update_scatter_fig(Patch(), _stats.istd, _stats.icor, x, y)
        return std_fig, cor_fig, std_cor_fig

    return stats_div
