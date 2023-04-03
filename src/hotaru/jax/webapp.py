from threading import Thread
import pathlib

import hydra
import numpy as np
import pandas as pd
from dash import (
    Dash,
    Input,
    Output,
    State,
    no_update,
)
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .webui import get_ui
from .model import Model
from .filter.gaussian import gaussian
from .filter.laplace import gaussian_laplace
from .utils.progress import SimpleProgress
from .callbacks import MyDash
from .layout import Layout
from .filter.pool import max_pool
from .footprint.reduce import reduce_peak, reduce_peak_block


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg):
    model = Model()
    layout = Layout(model)

    app = MyDash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
    app.layout = get_ui(cfg)

    app.collapse_callback("stats")
    app.collapse_callback("frame")
    app.collapse_callback("peak")

    def load_test(path, mask, hz):
        model.load_imgs(path, mask, hz)
        nt, h, w = model.imgs.shape
        return model.load_stats()

    def load_target():
        pbar = SimpleProgress(model.shape[0])
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
    def on_load(data):
        if not hasattr(model, "imgs"):
            return no_update, no_update, no_update
        nt = model.shape[0]
        marks = {i: f"{i}" for i in range(0, nt + 1, nt // 10)}
        return nt - 1, marks, 0

    @app.callback(
        Input("load-finished",  "data"),
        Output("stats", "style"),
        Output("stdImage", "figure"),
        Output("corImage", "figure"),
        Output("std-cor", "figure"),
        Output("imgs_minmax", "min"),
        Output("imgs_minmax", "max"),
        Output("imgs_minmax", "marks"),
        prevent_initial_call=True,
    )
    def finish_load(finished):
        if finished != "finished":
            return (dict(display="none"),) + (no_update,) * 6
        style = dict(display="flex")
        maxi, stdi, cori = model.istats
        #fil_cori = gaussian(cori[None, ...], 1)[0]
        fil_cori = cori
        max_cori = max_pool(fil_cori, (3, 3), (1, 1), "same")
        y, x = np.where(fil_cori == max_cori)
        stdv = stdi[y, x]
        corv = cori[y, x]
        scatter = go.Scatter(x=x, y=y, mode="markers", marker=dict(opacity=0.3), showlegend=False)
        stdfig = go.Figure([layout.heatmap(stdi), scatter])
        stdfig.update_layout(
            title=dict(x=0.01, y=0.99, text="std"),
            **layout.layout,
            **layout.image_layout,
        )
        corfig = go.Figure([layout.heatmap(cori), scatter])
        corfig.update_layout(
            title=dict(x=0.01, y=0.99, text="cor"),
            **layout.layout,
            **layout.image_layout,
        )
        stdcor = go.Figure([go.Scatter(x=stdv, y=corv, mode="markers", showlegend=False)])
        stdcor.update_layout(
            xaxis=dict(title="std"),
            yaxis=dict(title="cor"),
            **layout.layout,
        )
        marks = {
            float(v): f"{v:.1f}"
            for v in np.arange(np.ceil(model.min), np.ceil(model.max), 1)
        }
        return (
            style,
            stdfig,
            corfig,
            stdcor,
            model.min,
            model.max,
            marks,
        )

    @app.callback(
        Input("frame", "value"),
        Input("radius", "value"),
        Input("imgs_minmax", "value"),
        Input("stats", "style"),
        Output("imagegraph", "style"),
        Output("image", "figure"),
        Output("filtered", "figure"),
        Output("hist", "figure"),
        prevent_initial_call=True,
    )
    def on_frame(t, radius, minmax, load):
        if not hasattr(model, "stats"):
            return dist(display="none"), no_update, no_update, no_update
        img = model.frame(t)
        log = gaussian_laplace(img[None, ...], radius)[0]
        radius = np.power(2, radius)
        vmin, vmax = minmax
        gcount, ghist = np.histogram(
            log.ravel(), bins=np.linspace(-0.1, 1.0, 100)
        )
        style = dict(display="flex")
        imgfig = go.Figure([layout.heatmap(img, zmin=vmin, zmax=vmax)], layout.layout)
        logfig = go.Figure([layout.heatmap(log, zmin=-0.4, zmax=0.4, colorscale="Picnic")], layout.layout)
        histfig = go.Figure(
            go.Bar(x=ghist, y=gcount, width=ghist[1] - ghist[0]),
            dict(
                margin=dict(l=0, r=0, t=0, b=0),
                width=layout.width,
                height=layout.height,
            ),
        )
        return style, imgfig, logfig, histfig

    def peak_test(radius, rnum):
        if not hasattr(model, "stats"):
            return None
        rmin, rmax = radius
        return model.load_peaks(2 ** rmin, 2 ** rmax, rnum)

    def peak_target():
        nt = model.shape[0]
        pbar = SimpleProgress(nt)
        thread = Thread(target=model.calc_peaks, kwargs=dict(pbar=pbar))
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
        Input("thr", "value"),
        Output("peakgraph", "style"),
        Output("circle", "figure"),
        Output("radius-intensity", "figure"),
        prevent_initial_call=True,
    )
    def finish_peak(finished, thr):
        if finished != "finished":
            return dict(display="none"), no_update, no_update
        peaks = reduce_peak_block(model.peakval, model.radius[0], model.radius[-1], thr, 100)
        print(peaks)
        nt, h, w = model.shape
        circle = go.Figure(layout=layout.layout)
        circle.add_trace(go.Scatter(x=peaks.x, y=peaks.y, mode="markers", marker=dict(size=2 * (layout.width / w) * peaks.r, opacity=peaks.v / peaks.v.max()), showlegend=False))
        scatter = go.Figure()
        scatter.add_trace(go.Scatter(x=(1 + 0.05 * np.random.randn(h * w)) * model.peakval.r.ravel(), y=model.peakval.val.ravel(), mode="markers", marker=dict(opacity=0.05), showlegend=False))
        scatter.add_trace(go.Scatter(x=peaks.r, y=peaks.v, mode="markers", marker=dict(opacity=0.5), showlegend=False))
        scatter.update_layout(
            xaxis=dict(title="radius", type="log"),
            yaxis=dict(title="intensity", rangemode="tozero"),
        )
        return dict(display="flex"), circle, scatter

    def make_test(thr):
        if not hasattr(model, "stats"):
            return None
        return model.load_footprints(thr)

    def make_target():
        nt, h, w = model.shape
        pbar = SimpleProgress(nt)
        thread = Thread(target=model.make_footprints, kwargs=dict(pbar=pbar))
        return thread, pbar

    app.thread_callback(
        "make",
        make_test,
        make_target,
        State("thr", "value"),
    )

    @app.callback(
        Input("make-finished", "data"),
        Input("cell-select", "value"),
        Output("makegraph", "style"),
        Output("cell-single", "figure"),
        Output("cell-all", "figure"),
        Output("cell-select", "max"),
        prevent_initial_call=True,
    )
    def finish_make(finished, select):
        if finished != "finished":
            return dict(display="none"), no_update, no_update, no_update
        single = layout.heatmap(model.footprints[select])
        cells = layout.heatmap(model.footprints.max(axis=0))
        return dict(display="flex"), single, cells, model.footprints.shape[0]

    app.run_server(debug=True, port=8888)


if __name__ == "__main__":
    main()
