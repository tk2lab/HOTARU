import dash_bootstrap_components as dbc
import numpy as np
from dash import (
    dcc,
    html,
)


class UI:
    def __init__(self, cfg):
        self.cfg = cfg
        self.stylesheet = dbc.themes.BOOTSTRAP
        self.collapses = ["stats", "frame", "peak", "make"]

        stats = html.Div(
            [
                dcc.Store(id="load-submitted"),
                dcc.Store(id="load-finished"),
                dcc.Interval(id="load-interval", interval=500, disabled=True),
                dbc.Button("Open/Close stats", id="stats-collapse-button"),
                dbc.Collapse(
                    [
                        html.Div(
                            [
                                html.Div(
                                    [
                                        dbc.Label(
                                            "Image file path", html_for="imgs_path"
                                        ),
                                        dbc.Input(
                                            id="imgs_path",
                                            type="text",
                                            value=cfg.data.imgs,
                                        ),
                                    ]
                                ),
                                html.Div(
                                    [
                                        dbc.Label("Mask type", html_for="imgs_mask"),
                                        dbc.Input(
                                            id="imgs_mask",
                                            type="text",
                                            value=cfg.data.mask,
                                        ),
                                    ]
                                ),
                                html.Div(
                                    [
                                        dbc.Label("Frequency", html_for="imgs_hz"),
                                        dbc.Input(
                                            id="imgs_hz",
                                            type="number",
                                            value=cfg.data.hz,
                                        ),
                                    ]
                                ),
                            ],
                            style=dict(display="flex"),
                        ),
                        html.Div(
                            [
                                dbc.Button("LOAD", id="load", n_clicks=0),
                                dbc.Progress(id="load-progress"),
                                dbc.Label("Gaussian", html_for="gaussian"),
                                dcc.Slider(
                                    id="gaussian",
                                    min=0,
                                    max=10,
                                    step=0.1,
                                    value=1,
                                    marks={i: str(i) for i in range(11)},
                                ),
                                dbc.Label("MaxPool", html_for="maxpool"),
                                dcc.Slider(
                                    id="maxpool", min=3, max=11, step=2, value=3
                                ),
                            ],
                            style=dict(
                                width="1200px",
                                display="grid",
                                gridTemplateColumns="auto 1fr",
                            ),
                        ),
                        html.H2("Conventional cell candidates detection"),
                        html.Div(
                            [
                                dcc.Graph(id="stdImage"),
                                dcc.Graph(id="corImage"),
                                dcc.Graph(id="std-cor"),
                            ],
                            id="stats",
                            style=dict(display="none"),
                        ),
                    ],
                    id="stats-collapse",
                ),
            ]
        )

        frame = html.Div(
            [
                dbc.Button("Open/Close frame", id="frame-collapse-button"),
                dbc.Collapse(
                    [
                        html.H2("Check frames (optional)"),
                        html.Div(
                            [
                                dbc.Label("Frame", html_for="frame"),
                                dcc.Slider(
                                    id="frame",
                                    updatemode="drag",
                                    min=0,
                                    max=0,
                                    step=1,
                                    value=0,
                                ),
                                dbc.Label("Min/Max", html_for="imgs_minmax"),
                                dcc.RangeSlider(
                                    id="imgs_minmax",
                                    min=-1,
                                    max=1,
                                    step=0.1,
                                    value=[-1, 1],
                                ),
                                dbc.Label("Filter's Radius", html_for="radius"),
                                dcc.Slider(
                                    id="radius",
                                    min=0,
                                    max=5,
                                    step=0.1,
                                    value=2,
                                    marks={i: str(2**i) for i in range(6)},
                                ),
                            ],
                            style=dict(
                                width="1200px",
                                display="grid",
                                gridTemplateColumns="auto 1fr",
                            ),
                        ),
                        html.Div(
                            [
                                dcc.Graph(id="image"),
                                dcc.Graph(id="filtered"),
                                dcc.Graph(id="hist"),
                            ],
                            id="imagegraph",
                            style=dict(display="none"),
                        ),
                    ],
                    id="frame-collapse",
                ),
            ]
        )

        peak = html.Div(
            [
                dcc.Store(id="peak-submitted"),
                dcc.Store(id="peak-finished"),
                dcc.Interval(id="peak-interval", interval=500, disabled=True),
                dbc.Button("Open/Close peak", id="peak-collapse-button"),
                dbc.Collapse(
                    [
                        html.H2("Find peaks"),
                        html.Div(
                            [
                                dbc.Label("Radius range", html_for="radius-range"),
                                dcc.RangeSlider(
                                    id="radius-range",
                                    min=0,
                                    max=5,
                                    step=0.1,
                                    value=[1, 4],
                                    marks={i: str(2**i) for i in range(6)},
                                ),
                                dbc.Label("Num radius", html_for="nradius"),
                                dcc.Slider(
                                    id="nradius", min=1, max=50, step=1, value=11
                                ),
                                dbc.Button("PEAK", id="peak", n_clicks=0),
                                dbc.Progress(id="peak-progress"),
                                dbc.Label("Radius cut range", html_for="radius-range2"),
                                dcc.RangeSlider(
                                    id="radius-range2",
                                    min=0,
                                    max=10,
                                    step=1,
                                    value=[1, 9],
                                    marks={
                                        i: f"{2 ** r:.2f}"
                                        for i, r in enumerate(np.linspace(1, 5, 11))
                                    },
                                ),
                                dbc.Label("Distance threshold", html_for="thr"),
                                dcc.Slider(
                                    id="thr", min=0.5, max=3, step=0.1, value=2.0
                                ),
                            ],
                            style=dict(
                                width="1200px",
                                display="grid",
                                gridTemplateColumns="auto 1fr",
                            ),
                        ),
                        html.Div(
                            [
                                dcc.Graph(id="circle"),
                                dcc.Graph(id="radius-intensity"),
                            ],
                            id="peakgraph",
                            style=dict(display="none"),
                        ),
                    ],
                    id="peak-collapse",
                ),
            ]
        )

        footprint = html.Div(
            [
                dcc.Store(id="make-submitted"),
                dcc.Store(id="make-finished"),
                dcc.Interval(id="make-interval", interval=500, disabled=True),
                dcc.Store(id="spike-submitted"),
                dcc.Store(id="spike-finished"),
                dcc.Interval(id="spike-interval", interval=500, disabled=True),
                dbc.Button("Open/Close main", id="make-collapse-button"),
                dbc.Collapse(
                    [
                        html.H2("Main loop"),
                        html.Div(
                            [
                                dbc.Button("INIT FOOTPRINTS", id="make", n_clicks=0),
                                dbc.Progress(id="make-progress"),
                                dbc.Button("UPDATE SPIKES", id="spike", n_clicks=0),
                                dbc.Progress(id="spike-progress"),
                                dbc.Button("UPDATE FOOTPRINTS", id="cell", n_clicks=0),
                                dbc.Progress(id="cell-progress"),
                                dbc.Label("Select", html_for="cell-select"),
                                dcc.Slider(
                                    id="cell-select",
                                    updatemode="drag",
                                    min=0,
                                    max=0,
                                    step=1,
                                    value=0,
                                ),
                            ],
                            style=dict(
                                width="1200px",
                                display="grid",
                                gridTemplateColumns="auto 1fr",
                            ),
                        ),
                        html.Div(
                            [
                                dcc.Graph(id="cell-all"),
                                dcc.Graph(id="cell-single"),
                            ],
                            id="makegraph",
                            style=dict(display="none"),
                        ),
                    ],
                    id="make-collapse",
                ),
            ]
        )

        self.layout = html.Div([stats, frame, peak, footprint])
