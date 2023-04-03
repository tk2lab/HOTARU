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
from .webui import get_ui
from .model import Model
from .utils.progress import SimpleProgress


def collapse_callback(app, name):
    @app.callback(
        [Output(f"{name}-collapse", "is_open")],
        [Input(f"{name}-collapse-button", "n_clicks")],
        [State(f"{name}-collapse", "is_open")],
    )
    def toggle_stats(n, is_open):
        return not is_open,


def check_callback(app, job, name):
    @app.callback(
        [
            Output(f"{name}-progress", "value"),
            Output(f"{name}-finished", "data"),
        ], [
            Input(f"{name}-interval", "n_intervals"),
        ], [
            State(f"{name}-submitted", "data"),
        ],
        prevent_initial_call=True,
    )
    def check(n, submitted):
        uid = submitted["uid"]
        thread, pbar = job.get(uid, (None, None))
        print("check", uid, thread, pbar)
        if uid == "saved" or not thread.is_alive():
            return 100, dict(uid=submitted["uid"])
        else:
            return pbar.value, no_update

    @app.callback(
        [
            Output(f"{name}-interval",  "disabled"),
        ], [
            Input(f"{name}-submitted", "data"),
            Input(f"{name}-finished", "data"),
        ],
        prevent_initial_call=True,
    )
    def start_stop(submitted, finished):
        print(submitted, finished)
        if submitted and (not finished or submitted["uid"] != finished["uid"]):
            return False,
        return True,


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg):
    model = Model()

    app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
    app.layout = get_ui(cfg)
    job = dict()

    ui_width = 490

    collapse_callback(app, "stats")
    collapse_callback(app, "frame")
    collapse_callback(app, "peak")

    @app.callback(
        [
            Output("load-submitted", "data"),
            Output("frame", "max"),
            Output("frame", "marks"),
            Output("frame", "value"),
        ], [
            Input("load", "n_clicks"),
        ], [
            State("imgs_path", "value"),
            State("imgs_mask", "value"),
            State("imgs_hz", "value"),
        ],
        prevent_initial_call=True,
    )
    def on_load(click, path, mask, hz):
        model.load_imgs(path, mask, hz)
        nt, h, w = model.imgs.shape
        marks = {i: f"{i}" for i in range(0, nt + 1, nt // 10)}
        if model.load_stats(): 
            return dict(uid="saved"), nt - 1, marks, 0
        pbar = SimpleProgress(nt)
        thread = threading.Thread(target=model.calc_stats, kwargs=dict(pbar=pbar))
        thread.start()
        job[thread.native_id] = thread, pbar
        return dict(uid=thread.native_id), nt - 1, marks, 0

    check_callback(app, job, "load")

    @app.callback(
        [
            Output("stats", "style"),
            Output("maxImage", "figure"),
            Output("stdImage", "figure"),
            Output("corImage", "figure"),
            Output("imgs_minmax", "min"),
            Output("imgs_minmax", "max"),
            Output("imgs_minmax", "marks"),
        ], [
            Input("load-interval",  "disabled"),
        ],
        prevent_initial_call=True,
    )
    def finish_load(disabled):
        if not disabled:
            return (dict(display="none"),) + (no_update,) * 6
        maxi, stdi, cori = model.istats
        nt, h, w = model.shape
        width = ui_width + 10
        height = ui_width * h / w + 50
        vmarks = {
            int(v): f"{v:.1f}"
            for v in np.arange(np.ceil(model.min), np.ceil(model.max), 1.0)
        }
        layout = dict(
            margin=dict(l=0, r=0, t=0, b=0),
            width=width,
            height=height,
            xaxis=dict(visible=False, constrain="domain", domain=[0, 1]),
            yaxis=dict(visible=False, constrain="domain", domain=[50 / height, 1], scaleanchor="x", autorange="reversed"),
        )
        colorbar = dict(orientation="h", yanchor="bottom", y=0, thickness=10)
        style = dict(display="flex")
        maxData = go.Figure(layout=layout)
        maxData.add_trace(go.Heatmap(z=maxi, colorscale="greens", colorbar=colorbar))
        maxData.update_layout(title=dict(x=0.01, y=0.99, text="max"))
        stdData = go.Figure(layout=layout)
        stdData.add_trace(go.Heatmap(z=stdi, colorscale="greens", colorbar=colorbar))
        stdData.update_layout(title=dict(x=0.01, y=0.99, text="std"))
        corData = go.Figure(layout=layout)
        corData.add_trace(go.Heatmap(z=cori, colorscale="greens", colorbar=colorbar))
        corData.update_layout(title=dict(x=0.01, y=0.99, text="cor"))
        return (
            style,
            maxData,
            stdData,
            corData,
            model.min,
            model.max,
            vmarks,
        )

    @app.callback(
        [
            Output("imagegraph", "style"),
            Output("image", "figure"),
            Output("filtered", "figure"),
            Output("hist", "figure"),
        ],
        [
            Input("frame", "value"),
            Input("radius", "value"),
            Input("imgs_minmax", "value"),
        ],
        prevent_initial_call=True,
    )
    def on_frame(t, radius, minmax):
        radius = np.power(2, radius)
        vmin, vmax = minmax
        style = dict(display="none")
        image = go.Figure()
        filtered = go.Figure()
        hist = go.Figure()
        if hasattr(model, "stats"):
            img = model.frame(t)
            gl = gaussian_laplace(img[None, ...], radius, 4 * np.ceil(radius))[0]
            gcount, ghist = np.histogram(
                gl.ravel(), bins=np.linspace(-0.1, 1.0, 100)
            )
            nt, h, w = model.shape
            width = ui_width + 10
            height = ui_width * h / w + 50
            layout = dict(
                margin=dict(l=0, r=0, t=0, b=0),
                width=width,
                height=height,
                xaxis=dict(visible=False, constrain="domain"),
                yaxis=dict(visible=False, domain=[50 / height, 1], scaleanchor="x", autorange="reversed"),
            )
            colorbar = dict(orientation="h", yanchor="bottom", y=0, thickness=10)
            style = dict(display="flex")
            image = go.Figure(layout=layout)
            image.add_trace(go.Heatmap(z=img, zmin=vmin, zmax=vmax, colorscale="greens", colorbar=colorbar))
            filtered = go.Figure(layout=layout)
            filtered.add_trace(go.Heatmap(z=gl, zmin=-0.4, zmax=0.4, colorscale="Picnic", colorbar=colorbar))
            hist = go.Figure(
                go.Bar(x=ghist, y=gcount, width=ghist[1] - ghist[0]),
                dict(
                    margin=dict(l=0, r=0, t=0, b=0),
                    width=width,
                    height=height,
                ),
            )
        return style, image, filtered, hist

    @app.callback(
        [
            Output("peak-submitted", "data"),
        ], [
            Input("peak", "n_clicks"),
        ], [
            State("radius-range", "value"),
            State("nradius", "value"),
        ],
        prevent_initial_call=True,
    )
    def on_peak(click, radius, rnum):
        if not hasattr(model, "stats"):
            return no_update
        rmin, rmax = radius
        if model.load_peaks(2 ** rmin, 2 ** rmax, rnum):
            return dict(uid="saved"),
        nt, h, w = model.shape
        pbar = SimpleProgress(nt)
        thread = threading.Thread(target=model.calc_peaks, kwargs=dict(pbar=pbar))
        thread.start()
        job[thread.native_id] = thread, pbar
        return dict(uid=thread.native_id),

    check_callback(app, job, "peak")

    @app.callback(
        [
            Output("peakgraph", "style"),
            Output("circle", "figure"),
            Output("radius-intensity", "figure"),
        ], [
            Input("peak-interval", "disabled"),
            Input("thr", "value"),
        ],
        prevent_initial_call=True,
    )
    def finish_peak(disabled, thr):
        if not disabled:
            return dict(display="none"), no_update, no_update
        peaks = reduce_peak_block(model.peakval, model.radius[0], model.radius[-1], thr, 100)
        print(peaks)
        nt, h, w = model.shape
        width = ui_width + 10
        height = ui_width * h / w + 50
        layout = dict(
            margin=dict(l=0, r=0, t=0, b=0),
            width=width,
            height=height,
            xaxis=dict(visible=False, constrain="domain"),
            yaxis=dict(visible=False, domain=[50 / height, 1], scaleanchor="x", autorange="reversed"),
        )
        circle = go.Figure(layout=layout)
        circle.add_trace(go.Scatter(x=peaks.x, y=peaks.y, mode="markers", marker=dict(size=1.5 * (width / w) * (thr * peaks.r), opacity=peaks.v / peaks.v.max()), showlegend=False))
        scatter = go.Figure()
        scatter.add_trace(go.Scatter(x=peaks.r, y=peaks.v, mode="markers", showlegend=False))
        scatter.update_layout(
            xaxis=dict(title="radius", type="log"),
            yaxis=dict(title="intensity", rangemode="tozero"),
        )
        return dict(display="flex"), circle, scatter

    @app.callback(
        [
            Output("make-submitted", "data"),
        ], [
            Input("make", "n_clicks"),
        ], [
            State("thr", "value"),
        ],
        prevent_initial_call=True,
    )
    def on_make(click, thr):
        if not hasattr(model, "stats"):
            return no_update
        if model.load_footprints(thr):
            return dict(uid="saved"),
        print("make", thr)
        nt, h, w = model.shape
        pbar = SimpleProgress(nt)
        thread = threading.Thread(target=model.make_footprints, kwargs=dict(pbar=pbar))
        thread.start()
        job[thread.native_id] = thread, pbar
        return dict(uid=thread.native_id),

    check_callback(app, job, "make")

    @app.callback(
        [
            Output("makegraph", "style"),
            Output("cell-single", "figure"),
            Output("cell-all", "figure"),
            Output("cell-select", "max"),
        ], [
            Input("make-interval", "disabled"),
            Input("cell-select", "value"),
        ],
        prevent_initial_call=True,
    )
    def finish_make(disabled, select):
        if not disabled:
            return dict(display="none"), no_update, no_update, no_update
        nt, h, w = model.shape
        width = ui_width + 10
        height = ui_width * h / w + 50
        layout = dict(
            margin=dict(l=0, r=0, t=0, b=0),
            width=width,
            height=height,
            xaxis=dict(visible=False, constrain="domain"),
            yaxis=dict(visible=False, domain=[50 / height, 1], scaleanchor="x", autorange="reversed"),
        )
        colorbar = dict(orientation="h", yanchor="bottom", y=0, thickness=10)
        single = go.Figure(layout=layout)
        single.add_trace(go.Heatmap(z=model.footprints[select], colorscale="greens", colorbar=colorbar))
        cells = go.Figure(layout=layout)
        cells.add_trace(go.Heatmap(z=model.footprints.max(axis=0), colorscale="greens", colorbar=colorbar))
        return dict(display="flex"), single, cells, model.footprints.shape[0]

    app.run_server(debug=True, port=8888)


if __name__ == "__main__":
    main()
