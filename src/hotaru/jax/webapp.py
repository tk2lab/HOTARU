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
    app.collapse_callback("make")

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
        Input("gaussian", "value"),
        Input("maxpool", "value"),
        Output("stats", "style"),
        Output("stdImage", "figure"),
        Output("corImage", "figure"),
        Output("std-cor", "figure"),
        Output("imgs_minmax", "min"),
        Output("imgs_minmax", "max"),
        Output("imgs_minmax", "marks"),
        prevent_initial_call=True,
    )
    def finish_load(finished, gauss, maxpool):
        print("load", finished, gauss, maxpool)
        if finished != "finished":
            return (dict(display="none"),) + (no_update,) * 6
        style = dict(display="flex")
        maxi, stdi, cori = model.istats
        if gauss > 0:
            fil_cori = gaussian(cori[None, ...], gauss)[0]
        else:
            fil_cori = cori
        max_cori = max_pool(fil_cori, (maxpool, maxpool), (1, 1), "same")
        y, x = np.where(fil_cori == max_cori)
        stdv = stdi[y, x]
        corv = cori[y, x]
        scatter = go.Scatter(x=x, y=y, mode="markers", marker=dict(opacity=0.3), showlegend=False)
        stdfig = go.Figure([layout.heatmap(stdi), scatter], layout.image_layout)
        stdfig.update_layout(
            title=dict(x=0.01, y=0.99, text="std"),
        )
        corfig = go.Figure([layout.heatmap(cori), scatter], layout.image_layout)
        corfig.update_layout(
            title=dict(x=0.01, y=0.99, text="cor"),
        )
        stdcor = go.Figure([go.Scatter(x=stdv, y=corv, mode="markers", showlegend=False)], layout.layout)
        stdcor.update_layout(
            xaxis=dict(title="std"),
            yaxis=dict(title="cor"),
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
        Output("imagegraph", "style"),
        Output("image", "figure"),
        Output("filtered", "figure"),
        Output("hist", "figure"),
        prevent_initial_call=True,
    )
    def on_frame(t, radius, minmax):
        if not hasattr(model, "stats"):
            return dict(display="none"), no_update, no_update, no_update
        radius = np.power(2, radius)
        vmin, vmax = minmax
        img = model.frame(t)
        log = gaussian_laplace(img[None, ...], radius)[0]
        gcount, ghist = np.histogram(
            log.ravel(), bins=np.linspace(-0.1, 1.0, 100)
        )
        style = dict(display="flex")
        imgfig = go.Figure([layout.heatmap(img, zmin=vmin, zmax=vmax)], layout.image_layout)
        logfig = go.Figure([layout.heatmap(log, zmin=-0.4, zmax=0.4, colorscale="Picnic")], layout.image_layout)
        histfig = go.Figure(go.Bar(x=ghist, y=gcount, width=ghist[1] - ghist[0]), layout.layout)
        histfig.update_layout(
            xaxis=dict(title="intensity"),
            yaxis=dict(title="count"),
        )
        return style, imgfig, logfig, histfig

    def peak_test(radius, rnum):
        if not hasattr(model, "stats"):
            return None
        rmin, rmax = radius
        return model.load_peakval(2 ** rmin, 2 ** rmax, rnum)

    def peak_target():
        nt = model.shape[0]
        pbar = SimpleProgress(nt)
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
        prevent_initial_call=True,
    )
    def finish_peak(finished, radius, thr):
        if finished != "finished":
            return dict(display="none"), no_update, no_update
        rmin, rmax = radius
        rmin = model.radius[rmin] - 1e-6
        rmax = model.radius[rmax] + 1e-6
        model.calc_peaks(rmin, rmax, thr, 100)
        dr = model.radius[1] / model.radius[0] - 1
        nt, h, w = model.shape
        peaks = model.peaks.copy()
        nc = peaks.shape[0]
        r = 2 * (layout.width / w) * peaks.r
        v = peaks.v / peaks.v.max()
        peaks["id"] = peaks.index
        cells = go.Scatter(
            x=peaks.x,
            y=peaks.y,
            mode="markers",
            marker=dict(size=r, opacity=v),
            showlegend=False,
            customdata=peaks[["id", "r", "v"]],
            hovertemplate="id:%{customdata[0]}, x:%{x}, y:%{y}, r:%{customdata[1]:.3f}, v:%{customdata[2]:.3f}",
        )
        v0 = model.peakval.val.ravel()
        m = (v0.size + 9999) // 10000
        v0 = v0[::m]
        r0 = model.peakval.r.ravel()[::m]
        circle = go.Figure([cells], layout.image_layout)
        allpeak = go.Scatter(
            x=(1 + 0.3 * dr * np.random.randn(r0.size)) * r0,
            y=v0,
            mode="markers",
            marker=dict(opacity=0.1),
            showlegend=False,
        )
        select = go.Scatter(x=peaks.r, y=peaks.v, mode="markers", marker=dict(opacity=0.5), showlegend=False)
        scatter = go.Figure([allpeak, select], layout.layout)
        scatter.update_layout(
            xaxis=dict(title="radius", type="log"),
            yaxis=dict(title="intensity", rangemode="tozero"),
        )
        print("peak done")
        return dict(display="flex"), circle, scatter

    def make_test():
        if not hasattr(model, "stats"):
            return None
        return model.load_footprints()

    def make_target():
        nt, h, w = model.shape
        pbar = SimpleProgress(nt)
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
        bg = layout.heatmap(data.max(axis=0))
        circle = go.Scatter(
            x=peaks.x,
            y=peaks.y,
            mode="markers",
            showlegend=False,
            customdata=peaks[["id", "r", "v"]],
            hovertemplate="id:%{customdata[0]}, x:%{x}, y:%{y}, r:%{customdata[1]:.3f}, v:%{customdata[2]:.3f}",
        )
        cells = go.Figure([bg, circle], layout.image_layout)
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
        single = go.Figure([layout.heatmap(model.footprints[select])], layout.image_layout)
        return single,

    app.run_server(debug=True, port=8888)


if __name__ == "__main__":
    main()
