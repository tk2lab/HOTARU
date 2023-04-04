import pathlib
from threading import Thread

import hydra
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import (
    Input,
    Output,
    State,
    no_update,
)
from plotly.subplots import make_subplots

from ..filter.laplace import gaussian_laplace
from ..model import Model
from ..utils.progress import SimpleProgress
from .app import App
from .graph import Graph
from .ui import UI


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg):
    model = Model()
    ui = UI(cfg)
    graph = Graph(model)

    app = App(__name__, external_stylesheets=[ui.stylesheet])
    app.layout = ui.layout

    for name in ui.collapses:
        app.collapse_callback(name)

    def load_test(path, mask, hz):
        model.load_imgs(path, mask, hz)
        return model.load_stats()

    def load_target():
        pbar = SimpleProgress(model.dt)
        thread = Thread(target=model.calc_stats, kwargs=dict(pbar=pbar))
        return thread, pbar

    app.thread_callback(
        "load",
        load_test,
        load_target,
        Input("imgs_path", "value"),
        Input("imgs_mask", "value"),
        State("imgs_hz", "value"),
    )

    @app.callback(
        Input("load-submitted", "data"),
        Output("frame", "max"),
        Output("frame", "marks"),
        Output("frame", "value"),
        prevent_initial_call=True,
    )
    def update_frame(data):
        if not hasattr(model, "imgs"):
            return no_update, no_update, no_update
        nt = model.nt
        marks = {i: f"{i}" for i in range(0, nt + 1, nt // 10)}
        return nt - 1, marks, 0

    @app.callback(
        Input("load-finished", "data"),
        Input("gaussian", "value"),
        Input("maxpool", "value"),
        Output("stats", "style"),
        Output("stdImage", "figure"),
        Output("corImage", "figure"),
        Output("std-cor", "figure"),
        prevent_initial_call=True,
    )
    def plot_stats(finished, gauss, maxpool):
        print("load", finished, gauss, maxpool)
        if finished != "finished":
            return (dict(display="none"),) + (no_update,) * 6
        return dict(display="flex"), *graph.plot_stats(gauss, maxpool)

    @app.callback(
        Input("std-cor", "figure"),
        Output("imgs_minmax", "min"),
        Output("imgs_minmax", "max"),
        Output("imgs_minmax", "marks"),
        prevent_initial_call=True,
    )
    def update_minmax(update):
        marks = {
            float(v): f"{v:.1f}"
            for v in np.arange(np.ceil(model.min), np.ceil(model.max), 1)
        }
        return model.min, model.max, marks

    @app.callback(
        Input("frame", "value"),
        Input("radius", "value"),
        Input("imgs_minmax", "value"),
        Output("imagegraph", "style"),
        Output("image", "figure"),
        Output("filtered", "figure"),
        Output("hist", "figure"),
        prevent_initial_call=True,
    )
    def plot_frame(t, radius, minmax):
        if not hasattr(model, "stats"):
            return dict(display="none"), no_update, no_update, no_update
        return dict(display="flex"), *graph.plot_frame(t, radius, minmax)

    @app.callback(
        Input("radius-range", "value"),
        Input("nradius", "value"),
        Input("peak", "n_clicks"),
        Output("radius-range2", "max"),
        Output("radius-range2", "marks"),
    )
    def update_radius(radius, rnum, button):
        rmin, rmax = radius
        radius = np.power(2, np.linspace(rmin, rmax, rnum))
        marks = {i: f"{r:.2f}" for i, r in enumerate(radius)}
        return rnum - 1, marks

    def peak_test(radius, rnum):
        if not hasattr(model, "stats"):
            return None
        rmin, rmax = radius
        return model.load_peakval(2**rmin, 2**rmax, rnum)

    def peak_target():
        pbar = SimpleProgress(model.nt)
        thread = Thread(target=model.calc_peakval, kwargs=dict(pbar=pbar))
        return thread, pbar

    app.thread_callback(
        "peak",
        peak_test,
        peak_target,
        Input("radius-range", "value"),
        Input("nradius", "value"),
    )

    @app.callback(
        Input("peak-finished", "data"),
        Input("radius-range2", "value"),
        Input("thr", "value"),
        Output("peakgraph", "style"),
        Output("circle", "figure"),
        Output("radius-intensity", "figure"),
        prevent_initial_call=True,
    )
    def plot_peak(finished, radius, thr):
        if finished != "finished":
            return dict(display="none"), no_update, no_update
        rmin, rmax = radius
        rmin = model.radius[rmin] - 1e-6
        rmax = model.radius[rmax] + 1e-6
        self.model.calc_peaks(rmin, rmax, thr, 100)
        return dict(display="flex"), *graph.plot_peak()

    def make_test():
        if not hasattr(model, "stats"):
            return None
        return model.load_footprints()

    def make_target():
        pbar = SimpleProgress(model.nt)
        thread = Thread(target=model.make_footprints, kwargs=dict(pbar=pbar))
        return thread, pbar

    app.thread_callback(
        "make",
        make_test,
        make_target,
    )

    @app.callback(
        Input("make-finished", "data"),
        Output("makegraph", "style"),
        Output("cell-all", "figure"),
        Output("cell-select", "max"),
        Output("cell-select", "marks"),
        prevent_initial_call=True,
    )
    def finish_make(finished):
        if finished != "finished":
            return dict(display="none"), no_update, no_update, no_update
        peaks = model.peaks
        peaks["id"] = peaks.index
        data = model.footprints
        nc = data.shape[0]
        bg = graph.heatmap(data.max(axis=0))
        circle = go.Scatter(
            x=peaks.x,
            y=peaks.y,
            mode="markers",
            showlegend=False,
            customdata=peaks[["id", "r", "v"]],
            hovertemplate="id:%{customdata[0]}, x:%{x}, y:%{y}, r:%{customdata[1]:.3f}, v:%{customdata[2]:.3f}",
        )
        cells = go.Figure([bg, circle], graph.image_layout)
        marks = {i: f"{i}" for i in range(0, nc + 1, nc // 10)}
        print("plot all cells")
        return dict(display="flex"), cells, model.footprints.shape[0], marks

    @app.callback(
        Input("cell-all", "figure"),
        Input("cell-select", "value"),
        State("make-finished", "data"),
        Output("cell-single", "figure"),
    )
    def select_make(cells, select, finished):
        if finished != "finished":
            return no_update
        print("plot single cell")
        single = go.Figure(
            [graph.heatmap(model.footprints[select])], graph.image_layout
        )
        return (single,)

    app.run_server(debug=True, port=8888)
