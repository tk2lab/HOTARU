import os
import uuid
import pathlib
import threading
from functools import cached_property

import hydra
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import (
    Dash,
    Input,
    Output,
    State,
    dcc,
    html,
    no_update,
)
import dash_bootstrap_components as dbc
from plotly.subplots import make_subplots

from .filter.stats import calc_stats, Stats, ImageStats
from .filter.laplace import gaussian_laplace, gaussian_laplace_multi
from .io.image import load_imgs
from .io.mask import get_mask
from .footprint.find import find_peak, find_peak_batch, PeakVal
from .footprint.reduce import reduce_peak, reduce_peak_block


def get_ui(cfg):

    stats = html.Div([
        dcc.Store(id="load-submitted"),
        dcc.Store(id="load-finished"),
        dcc.Interval(id="load-interval", interval=500, disabled=True),
        dbc.Button("Open/Close stats", id="stats-collapse-button"),
        dbc.Collapse([
            html.H2("Simple image stats"),
            html.Div([
                html.Div([
                    dbc.Label("Image file path", html_for="imgs_path"),
                    dbc.Input(id="imgs_path", type="text", value=cfg.data.imgs),
                ]),
                html.Div([
                    dbc.Label("Mask type", html_for="imgs_mask"),
                    dbc.Input(id="imgs_mask", type="text", value=cfg.data.mask),
                ]),
                html.Div([
                    dbc.Label("Frequency", html_for="imgs_hz"),
                    dbc.Input(id="imgs_hz", type="number", value=cfg.data.hz),
                ]),
            ], style=dict(display="flex")),
            html.Div([
                dbc.Button("LOAD", id="load", n_clicks=0),
                dbc.Progress(id="load-progress"),
            ], style=dict(width="1200px", display="grid", gridTemplateColumns="auto 1fr")),
            html.Div([
                dcc.Graph(id="maxImage"),
                dcc.Graph(id="stdImage"),
                dcc.Graph(id="corImage"),
            ], id="stats", style=dict(display="none")),
        ], id="stats-collapse"),
    ])

    frame = html.Div([
        dbc.Button("Open/Close frame", id="frame-collapse-button"),
        dbc.Collapse([
            html.H2("Check frames (optional)"),
            html.Div([
                dbc.Label("Frame", html_for="frame"),
                dcc.Slider(id="frame", updatemode="drag", min=0, max=0, step=1, value=0),
                dbc.Label("Min/Max", html_for="imgs_minmax"),
                dcc.RangeSlider(id="imgs_minmax", min=-1, max=1, step=0.1, value=[-1, 1]),
                dbc.Label("Filter's Radius", html_for="radius"),
                dcc.Slider(id="radius", min=0, max=5, step=0.1, value=2, marks={i: str(2**i) for i in range(6)}),
            ], style=dict(width="1200px", display="grid", gridTemplateColumns="auto 1fr")),
            html.Div([
                dcc.Graph(id="image"),
                dcc.Graph(id="filtered"),
                dcc.Graph(id="hist"),
            ], id="imagegraph", style=dict(display="none")),
        ], id="frame-collapse"),
    ])

    peak = html.Div([
        dcc.Store(id="peak-submitted"),
        dcc.Store(id="peak-finished"),
        dcc.Interval(id="peak-interval", interval=500, disabled=True),
        dbc.Button("Open/Close peak", id="peak-collapse-button"),
        dbc.Collapse([
            html.H2("Find peaks"),
            html.Div([
                dbc.Label("Radius range", html_for="radius-range"),
                dcc.RangeSlider(id="radius-range", min=0, max=5, step=0.1, value=[1, 4], marks={i: str(2**i) for i in range(6)}),
                dbc.Label("Num radius", html_for="nradius"),
                dcc.Slider(id="nradius", min=1, max=30, step=1, value=11),
                dbc.Button("PEAK", id="peak", n_clicks=0),
                dbc.Progress(id="peak-progress"),
                dbc.Label("Distance threshold", html_for="thr"),
                dcc.Slider(id="thr", min=0.5, max=3, step=0.1, value=1.5),
            ], style=dict(width="1200px", display="grid", gridTemplateColumns="auto 1fr")),
            html.Div([
                dcc.Graph(id="circle"),
                dcc.Graph(id="radius-intensity"),
            ], id="peakgraph", style=dict(display="none")),
        ], id="peak-collapse"),
    ])

    footprint = html.Div([
        dcc.Store(id="make-submitted"),
        dcc.Store(id="make-finished"),
        dcc.Interval(id="make-interval", interval=500, disabled=True),
        html.H2("Make initial candidates"),
        html.Div([
            dbc.Button("MAKE", id="make", n_clicks=0),
            dbc.Progress(id="make-progress"),
            dbc.Label("Cell", html_for="cell-select"),
            dcc.Slider(id="cell-select", updatemode="drag", min=0, max=0, step=1, value=0),
        ], style=dict(width="1200px", display="grid", gridTemplateColumns="auto 1fr")),
        html.Div([
            dcc.Graph(id="cell-single"),
            dcc.Graph(id="cell-all"),
        ], id="makegraph", style=dict(display="none")),
    ])

    return html.Div([stats, frame, peak, footprint])
