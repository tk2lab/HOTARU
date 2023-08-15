import dash.html as html
import dash_bootstrap_components as dbc
import diskcache
from dash import (
    Dash,
    DiskcacheManager,
)

from .config import create_config_div
from .init import create_init_div
from .peaks import create_peaks_div
from .stats import create_stats_div

"""
from ..footprint.find import simple_peaks
from ..main import (
    data,
    find,
    footprint,
    get_frame,
    make,
    reduce,
    spike,
    stats,
)
from .fig import (
    bar_fig,
    circle_fig,
    heat_fig,
    scatter_fig,
    spike_fig,
    update_img,
)
from .ui import (
    Collapse,
    ThreadButton,
    two_column,
)
"""


def gui(cfg, *args, **kwargs):
    kwargs.setdefault("external_stylesheets", [dbc.themes.BOOTSTRAP])
    kwargs.setdefault("title", "HOTARU")

    cache = diskcache.Cache(f"{cfg.outdir}/server_cache")
    manager = DiskcacheManager(cache)
    app = Dash(__name__, *args, **kwargs, background_callback_manager=manager)

    app.layout = html.Div(
        children=[
            cfg_div := create_config_div(cfg),
            dbc.Tabs(
                children=[
                    dbc.Tab(label="STATS", children=create_stats_div(cfg_div.store)),
                    dbc.Tab(label="PEAKS", children=create_peaks_div(cfg_div.store)),
                    dbc.Tab(label="INIT", children=create_init_div(cfg_div.store)),
                ],
            ),
        ]
    )

    return app
