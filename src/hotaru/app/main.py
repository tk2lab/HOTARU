import time
from threading import Thread

import dash_bootstrap_components as dbc
import diskcache
import numpy as np
import plotly.graph_objects as go
from dash import (
    Dash,
    Input,
    Output,
    State,
    callback,
    ctx,
    dcc,
    html,
    no_update,
)

from ..jax.filter.laplace import gaussian_laplace
from ..jax.model import Model


class Progress:
    def reset(self, total):
        self.total = total
        self.n = 0

    def update(self, n):
        self.n += n

    @property
    def value(self):
        return self.n / self.total


def ThreadButton(label, func, *state):
    div = html.Div(
        children=[
            button := dbc.Button(label),
            pbar := dbc.Progress(),
            interval := dcc.Interval(interval=100, disabled=True),
        ]
    )
    jobs = [None, None]

    @callback(
        Output(interval, "disabled"),
        Output(pbar, "value"),
        Input(button, "n_clicks"),
        Input(interval, "n_intervals"),
        *state,
        prevent_initial_call=True,
    )
    def on_click(nc, ni, *state):
        if ctx.triggered_id == button.id:
            pbar = Progress()
            thread = Thread(target=func, args=state, kwargs=dict(pbar=pbar))
            jobs[:] = thread, pbar
            thread.start()
            return False, no_update
        else:
            thread, pbar = jobs
            if thread.is_alive():
                return no_update, pbar.value
            else:
                return True, 100

    div.finish = Input(interval, "disabled")
    return div


def Collapse(name, *children):
    div = html.Div(
        children=[
            button := dbc.Button(f"Open/Close {name}"),
            collapse := dbc.Collapse(children=children),
        ]
    )

    @callback(
        Output(collapse, "is_open"),
        Input(button, "n_clicks"),
        State(collapse, "is_open"),
        prevent_initial_call=True,
    )
    def toggle_stats(n, is_open):
        return not is_open

    return div


class MainApp(Dash):
    def __init__(self, cfg, *args, **kwargs):
        kwargs.setdefault("external_stylesheets", [dbc.themes.BOOTSTRAP])
        kwargs.setdefault("title", "HOTARU")
        super().__init__(__name__, *args, **kwargs)

        model = Model(cfg)

        def load_callback(path, mask, hz, pbar):
            print("load")
            model.load_imgs(path, mask, hz)
            print("load done")
            if not model.load_stats():
                model.calc_stats(pbar)

        load_div = Collapse(
            "load",
            dbc.Label("Image file path"),
            imgs_path := dbc.Input(type="text", value=cfg.data.imgs),
            dbc.Label("Mask type"),
            imgs_mask := dbc.Input(type="text", value=cfg.data.mask),
            dbc.Label("Frequency"),
            imgs_hz := dbc.Input(type="number", value=cfg.data.hz),
            load := ThreadButton(
                "LOAD",
                load_callback,
                State(imgs_path, "value"),
                State(imgs_mask, "value"),
                State(imgs_hz, "value"),
            ),
        )

        stats_div = Collapse(
            "stats",
            html.H2("Conventional cell candidates detection"),
            html.Div(
                children=[
                    dbc.Label("Gaussian"),
                    gaussian := dcc.Slider(
                        min=0,
                        max=10,
                        step=0.1,
                        value=1,
                        marks={i: str(i) for i in range(11)},
                    ),
                    dbc.Label("MaxPool"),
                    maxpool := dcc.Slider(min=3, max=11, step=2, value=3),
                ],
                style=dict(
                    width="1200px",
                    display="grid",
                    gridTemplateColumns="auto 1fr",
                ),
            ),
            stats_graph := dbc.Row(
                children=[
                    dbc.Col(std_image := dcc.Graph()),
                    dbc.Col(cor_image := dcc.Graph()),
                    dbc.Col(std_cor := dcc.Graph()),
                ],
            ),
        )

        ui_width = 490

        def layout():
            if hasattr(model, "shape"):
                h, w = model.shape
            else:
                h, w = 100, 100
            width = ui_width + 10
            height = ui_width * h / w + 50
            return dict(
                margin=dict(l=0, r=0, t=0, b=0),
                width=width,
                height=height,
            )

        def image_layout():
            if hasattr(model, "shape"):
                h, w = model.shape
            else:
                h, w = 100, 100
            width = ui_width + 10
            height = ui_width * h / w + 50
            return dict(
                **layout(),
                xaxis=dict(visible=False, range=(0, w), domain=[0, 1 - 10 / width]),
                yaxis=dict(visible=False, range=(h, 0), domain=[50 / height, 1]),
            )

        colorbar = dict(orientation="h", yanchor="bottom", y=0, thickness=10)

        @callback(
            Output(std_image, "figure"),
            Output(cor_image, "figure"),
            Output(std_cor, "figure"),
            load.finish,
            Input(gaussian, "value"),
            Input(maxpool, "value"),
        )
        def plot_stats(finished, gauss, maxpool):
            if hasattr(model, "istats"):
                y, x = model.simple_peaks(gauss, maxpool)
                imin, imax, istd, icor = model.istats
                vstd = istd[y, x]
                vcor = icor[y, x]
            else:
                istd, icor, x, y, vstd, vcor = None, None, [], [], [], []

            heat_std = go.Heatmap(z=istd, colorscale="greens", colorbar=colorbar)
            heat_cor = go.Heatmap(z=icor, colorscale="greens", colorbar=colorbar)
            peaks = go.Scatter(
                x=x, y=y, mode="markers", marker=dict(color="red", opacity=0.3)
            )
            std_cor = go.Scatter(
                x=vstd, y=vcor, mode="markers", marker=dict(color="red", opacity=0.5)
            )

            stdfig = go.Figure(
                [peaks, heat_std],
                dict(title=dict(x=0.01, y=0.99, text="std"), **image_layout()),
            )

            corfig = go.Figure(
                [peaks, heat_cor],
                dict(title=dict(x=0.01, y=0.99, text="cor"), **image_layout()),
            )

            scatter = go.Figure(
                [std_cor],
                dict(xaxis=dict(title="std"), yaxis=dict(title="cor"), **layout()),
            )

            return stdfig, corfig, scatter

        frame_div = Collapse(
            "frame",
            html.H2("Check frames (optional)"),
            html.Div(
                children=[
                    dbc.Label("Frame"),
                    frame := dcc.Slider(
                        updatemode="drag",
                        min=0,
                        max=0,
                        step=1,
                        value=0,
                    ),
                    dbc.Label("Min/Max"),
                    minmax := dcc.RangeSlider(
                        min=-1,
                        max=1,
                        step=0.1,
                        value=[-1, 1],
                    ),
                    dbc.Label("Filter's Radius"),
                    radius := dcc.Slider(
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
            frame_graph := dbc.Row(
                children=[
                    dbc.Col(norm_img := dcc.Graph()),
                    dbc.Col(filt_img := dcc.Graph()),
                    dbc.Col(hist_img := dcc.Graph()),
                ],
            ),
        )

        @callback(
            Output(frame, "max"),
            Output(frame, "marks"),
            Output(frame, "value"),
            Output(minmax, "min"),
            Output(minmax, "max"),
            Output(minmax, "marks"),
            load.finish,
            prevent_initial_call=True,
        )
        def update_frame(finish):
            if not finish:
                return (no_update,) * 6
            nt = model.nt
            fmarks = {i: f"{i}" for i in range(0, nt + 1, nt // 10)}
            m, M = model.min, model.max
            mmarks = {
                float(v): f"{v:.1f}" for v in np.arange(np.ceil(m), np.ceil(M), 1)
            }
            return nt - 1, fmarks, 0, m, M, mmarks

        @callback(
            Output(norm_img, "figure"),
            Output(filt_img, "figure"),
            Output(hist_img, "figure"),
            load.finish,
            Input(frame, "value"),
            Input(radius, "value"),
            Input(minmax, "value"),
        )
        def plot_frame(finish, t, radius, minmax):
            radius = np.power(2, radius)
            vmin, vmax = minmax
            if hasattr(model, "stats"):
                img = model.frame(t)
                log = gaussian_laplace(img[None, ...], radius)[0]
                logr = log.ravel()
            else:
                img, log, logr = None, None, []

            gcount, ghist = np.histogram(logr, bins=np.linspace(-0.1, 1.0, 100))

            heat_img = go.Heatmap(
                z=img, zmin=vmin, zmax=vmax, colorscale="greens", colorbar=colorbar
            )
            heat_log = go.Heatmap(
                z=log, zmin=-0.4, zmax=0.4, colorscale="Picnic", colorbar=colorbar
            )
            hist = go.Bar(x=ghist, y=gcount, width=ghist[1] - ghist[0])
            imgfig = go.Figure([heat_img], image_layout())
            logfig = go.Figure([heat_log], image_layout())
            histfig = go.Figure(
                [hist],
                dict(
                    xaxis=dict(title="intensity"), yaxis=dict(title="count"), **layout()
                ),
            )
            return imgfig, logfig, histfig

        """
        peak = Collapse(
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
                    "nradius",
                    "Num radius",
                    dcc.Slider,
                    min=1, max=50, step=1, value=11
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
                        i: f"{2 ** r:.2f}"
                        for i, r in enumerate(np.linspace(1, 5, 11))
                    },
                ),
                dbc.Label("Distance threshold", html_for="thr"),
                dcc.Slider(
                    id="thr", min=0.5, max=3, step=0.1, value=2.0
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
        )
        """

        self.layout = html.Div([load_div, stats_div, frame_div])
