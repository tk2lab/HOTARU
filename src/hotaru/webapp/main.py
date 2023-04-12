import numpy as np
from dash import (
    Input,
    Output,
    State,
    no_update,
)

from ..jax.model import Model
from .app import App
from .graph import Graph
from .ui import UI


def MainApp(cfg):
    ui = UI(cfg)
    app = App(__name__, ui)

    model = Model(cfg)
    graph = Graph(model)

    def load_test(path, mask, hz):
        model.load_imgs(path, mask, hz)
        return model.load_stats()

    app.thread_callback(
        "load",
        load_test,
        model.calc_stats,
        Input("imgs_path", "value"),
        Input("imgs_mask", "value"),
        State("imgs_hz", "value"),
    )

    def peak_test(radius, rnum):
        if not hasattr(model, "stats"):
            return None
        rmin, rmax = radius
        return model.load_peakval(2**rmin, 2**rmax, rnum)

    app.thread_callback(
        "peak",
        peak_test,
        model.calc_peakval,
        Input("radius-range", "value"),
        Input("nradius", "value"),
    )

    def make_test():
        if not hasattr(model, "peaks"):
            return None
        return model.load_footprints()

    app.thread_callback(
        "make",
        make_test,
        model.make_footprints,
    )

    @app.callback(
        Input("load-submitted", "data"),
        Output("frame", "max"),
        Output("frame", "marks"),
        Output("frame", "value"),
    )
    def update_frame(data):
        if not hasattr(model, "stats"):
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
    )
    def plot_stats(finished, gauss, maxpool):
        if finished != "finished":
            return (dict(display="none"),) + (no_update,) * 3
        return dict(display="flex"), *graph.plot_stats(gauss, maxpool)

    @app.callback(
        Input("std-cor", "figure"),
        Output("imgs_minmax", "min"),
        Output("imgs_minmax", "max"),
        Output("imgs_minmax", "marks"),
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

    @app.callback(
        Input("peak-finished", "data"),
        Input("radius-range2", "value"),
        Input("thr", "value"),
        Output("peakgraph", "style"),
        Output("circle", "figure"),
        Output("radius-intensity", "figure"),
    )
    def plot_peak(finished, radius, thr):
        if finished != "finished":
            return dict(display="none"), no_update, no_update
        rmin, rmax = radius
        if not model.load_peaks(rmin, rmax, thr):
            model.calc_peaks()
        return dict(display="flex"), *graph.plot_peak()

    @app.callback(
        Input("make-finished", "data"),
        Output("cell-select", "max"),
        Output("cell-select", "marks"),
    )
    def update_select(finished):
        if finished != "finished":
            return no_update, no_update
        nc = model.footprints.shape[0]
        marks = {i: f"{i}" for i in range(0, nc + 1, nc // 10)}
        return nc, marks

    @app.callback(
        Input("make-finished", "data"),
        Output("makegraph", "style"),
        Output("cell-all", "figure"),
    )
    def finish_make(finished):
        if finished != "finished":
            return dict(display="none"), no_update
        return dict(display="flex"), graph.plot_all()

    @app.callback(
        Input("make-finished", "data"),
        Input("cell-select", "value"),
        Output("cell-single", "figure"),
    )
    def select_make(finished, select):
        if finished != "finished":
            return (no_update,)
        return (graph.plot_single(select),)

    return app
