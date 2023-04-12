import dash_bootstrap_components as dbc
import numpy as np
from dash import (
    dcc,
    html,
)


def Input(name, label, kind, *args, **kwargs):
    return html.Div(
        children=[
            dbc.Label(label, html_for=name),
            kind(id=name, *args, **kwargs),
        ],
    )


def Div(name, style, *children):
    return html.Div(
        id=name,
        style=style,
        children=children,
    )


def ThreadButton(name, label):
    return html.Div(
        children=[
            dcc.Store(id=f"{name}-submitted"),
            dcc.Store(id=f"{name}-finished"),
            dcc.Interval(id=f"{name}-interval", interval=500, disabled=True),
            dbc.Button(label, id=name, n_clicks=0),
        ]
    )


def Collapse(name, *children):
    return html.Div(
        children=[
            dbc.Button("Open/Close frame", id=f"{name}-collapse-button"),
            dbc.Collapse(id=f"{name}-collapse", children=children),
        ]
    )


class UI:
    def __init__(self, cfg):
        self.cfg = cfg
        self.stylesheet = dbc.themes.BOOTSTRAP
        self.title = "HOTARU"
        self.collapses = ["stats", "frame", "peak", "make"]
        self.layout = html.Div(
            [self.load, self.stats, self.frame, self.peak, self.main]
        )

    @property
    def load(self):
        return Div(
            "load-div1",
            None,
            Div(
                "load-div2",
                dict(display="flex"),
                Input(
                    "imgs_path",
                    "Image file path",
                    dbc.Input,
                    type="text",
                    value=self.cfg.data.imgs,
                ),
                Input(
                    "imgs_mask",
                    "Mask type",
                    dbc.Input,
                    type="text",
                    value=self.cfg.data.mask,
                ),
                Input(
                    "imgs_hz",
                    "Frequency",
                    dbc.Input,
                    type="number",
                    value=self.cfg.data.hz,
                ),
            ),
            Div(
                "load-block",
                dict(
                    width="1200px",
                    display="grid",
                    gridTemplateColumns="auto 1fr",
                ),
                ThreadButton("load", "LOAD"),
                dbc.Progress(id="load-progress"),
            ),
        )

    @property
    def stats(self):
        return Collapse(
            "stats",
            html.H2("Conventional cell candidates detection"),
            html.Div(
                [
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
                    dcc.Slider(id="maxpool", min=3, max=11, step=2, value=3),
                ],
                style=dict(
                    width="1200px",
                    display="grid",
                    gridTemplateColumns="auto 1fr",
                ),
            ),
            html.Div(
                [
                    dcc.Graph(id="stdImage"),
                    dcc.Graph(id="corImage"),
                    dcc.Graph(id="std-cor"),
                ],
                id="stats",
                style=dict(display="none"),
            ),
        )

    @property
    def frame(self):
        return Collapse(
            "frame",
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
        )

    @property
    def peak(self):
        return Collapse(
            "peak",
            html.H2("Find peaks"),
            Div(
                "peak-input",
                dict(
                    width="1200px",
                    display="grid",
                    gridTemplateColumns="auto 1fr",
                ),
                Input(
                    "radius-range",
                    "Radius range",
                    dcc.RangeSlider,
                    min=0,
                    max=5,
                    step=0.1,
                    value=[1, 4],
                    marks={i: str(2**i) for i in range(6)},
                ),
                Input(
                    "nradius", "Num radius", dcc.Slider, min=1, max=50, step=1, value=11
                ),
                ThreadButton("peak", "PEAK"),
                dbc.Progress(id="peak-progress"),
                dbc.Label("Radius cut range", html_for="radius-range2"),
                dcc.RangeSlider(
                    id="radius-range2",
                    min=0,
                    max=10,
                    step=1,
                    value=[1, 9],
                    marks={
                        i: f"{2 ** r:.2f}" for i, r in enumerate(np.linspace(1, 5, 11))
                    },
                ),
                dbc.Label("Distance threshold", html_for="thr"),
                dcc.Slider(id="thr", min=0.5, max=3, step=0.1, value=2.0),
            ),
            html.Div(
                [
                    dcc.Graph(id="circle"),
                    dcc.Graph(id="radius-intensity"),
                ],
                id="peakgraph",
                style=dict(display="none"),
            ),
        )

    @property
    def main(self):
        return Collapse(
            "make",
            html.H2("Main loop"),
            html.Div(
                [
                    ThreadButton("make", "INIT FOOTPRINTS"),
                    dbc.Progress(id="make-progress"),
                    ThreadButton("spike", "INIT SPIKES"),
                    dbc.Progress(id="spike-progress"),
                    ThreadButton("update", "UPDATE"),
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
            Div(
                "makegraph",
                dict(display="none"),
                dcc.Graph(id="cell-all"),
                dcc.Graph(id="cell-single"),
            ),
        )
