import dash_bootstrap_components as dbc
import numpy as np
from dash import (
    Dash,
    Input,
    Output,
    Patch,
    State,
    callback,
    ctx,
    dcc,
    html,
    no_update,
)

from ..jax.filter.laplace import gaussian_laplace
from ..jax.model import Model
from .ui import (
    two_column,
    ThreadButton,
    Collapse,
)
from .fig import (
    heat_fig,
    update_img,
    scatter_fig,
    bar_fig,
    circle_fig,
)


class MainApp(Dash):
    def __init__(self, cfg, *args, **kwargs):
        kwargs.setdefault("external_stylesheets", [dbc.themes.BOOTSTRAP])
        kwargs.setdefault("title", "HOTARU")
        super().__init__(__name__, *args, **kwargs)

        model = Model(cfg)

        load_div = Collapse(
            "load",
            html.Div(
                style=two_column(500),
                children=[
                    dbc.Label("Image file path"),
                    imgs_path := dbc.Input(type="text", value=cfg.data.imgs),
                    dbc.Label("Mask type"),
                    imgs_mask := dbc.Input(type="text", value=cfg.data.mask),
                    dbc.Label("Frequency"),
                    imgs_hz := dbc.Input(type="number", value=cfg.data.hz),
                ],
            ),
            load := ThreadButton(
                "LOAD",
                model.calc_stats,
                State(imgs_path, "value"),
                State(imgs_mask, "value"),
                State(imgs_hz, "value"),
            ),
        )

        stats_div = Collapse(
            "stats",
            html.H2("Conventional cell candidates detection"),
            html.Div(
                style=two_column(1200),
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
            ),
            stats_graph := dbc.Row(
                children=[
                    dcc.Store("stats_data"),
                    dbc.Col(std_graph := dcc.Graph(figure=heat_fig("std"))),
                    dbc.Col(cor_graph := dcc.Graph(figure=heat_fig("cor"))),
                    dbc.Col(std_cor_graph := dcc.Graph(figure=scatter_fig("std-cor", "std", "cor"))),
                ],
            )
        )

        @callback(
            Output("stats_data", "data"),
            load.finish,
            Input(gaussian, "value"),
            Input(maxpool, "value"),
        )
        def update_speaks(finished, gauss, maxpool):
            if not hasattr(model, "istats"):
                return no_update
            y, x = model.simple_peaks(gauss, maxpool)
            imin, imax, istd, icor = model.istats
            model.simplex = x
            model.simpley = y
            model.vstd = istd[y, x]
            model.vcor = icor[y, x]
            return "finish"

        @callback(Output(std_graph, "figure"), Input("stats_data", "data"))
        def plot_std(data):
            fig = Patch()
            fig.data[0].z = model.istats.istd
            fig.data[1].x = model.simplex
            fig.data[1].y = model.simpley
            return update_img(fig, *model.shape)

        @callback(Output(cor_graph, "figure"), Input("stats_data", "data"))
        def plot_cor(data):
            fig = Patch()
            fig.data[0].z = model.istats.icor
            fig.data[1].x = model.simplex
            fig.data[1].y = model.simpley
            return update_img(fig, *model.shape)

        @callback(Output(std_cor_graph, "figure"), Input("stats_data", "data"))
        def plot_std_cor(data):
            fig = Patch()
            fig.data[0].x = model.vstd
            fig.data[0].y = model.vcor
            return fig

        frame_div = Collapse(
            "frame",
            html.H2("Check frames (optional)"),
            html.Div(
                style=two_column(1200),
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
            ),
            frame_graph := dbc.Row(
                children=[
                    dcc.Store("frame_data"),
                    dbc.Col(norm_img := dcc.Graph(figure=heat_fig("norm"))),
                    dbc.Col(filt_img := dcc.Graph(figure=heat_fig("filt", "Picnic", -0.4, 0.4))),
                    dbc.Col(hist_img := dcc.Graph(figure=bar_fig("intensity histogram", "intensity", "count"))),
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
        )
        def update_frame(finish):
            if not hasattr(model, "stats"):
                return (no_update,) * 6
            nt = model.nt
            fmarks = {i: f"{i}" for i in range(0, nt + 1, nt // 10)}
            m, M = np.floor(10 * model.min) / 10, np.ceil(10 * model.max) / 10
            mmarks = {
                float(v): f"{v:.1f}" for v in np.arange(m, M, 1)
            }
            return nt - 1, fmarks, 0, m, M, mmarks

        @callback(
            Output("frame_data", "data"),
            load.finish,
            Input(frame, "value"),
            Input(radius, "value"),
            Input(minmax, "value"),
        )
        def update_frame(finish, t, radius, minmax):
            if not hasattr(model, "stats"):
                return no_update
            radius = np.power(2, radius)
            model.vmin, model.vmax = minmax
            model.cur_img = model.frame(t)
            model.log_img = gaussian_laplace(model.cur_img[None, ...], radius)[0]
            return "finish"

        @callback(Output(norm_img, "figure"), Input("frame_data", "data"))
        def plot_norm(data):
            fig = Patch()
            fig.data[0].z = model.cur_img
            fig.data[0].zmin = model.vmin 
            fig.data[0].zmax = model.vmax
            return update_img(fig, *model.shape)

        @callback(Output(filt_img, "figure"), Input("frame_data", "data"))
        def plot_filt(data):
            fig = Patch()
            fig.data[0].z = model.log_img
            return update_img(fig, *model.shape)

        @callback(Output(hist_img, "figure"), Input("frame_data", "data"))
        def plot_filt(data):
            gcount, ghist = np.histogram(model.log_img.ravel(), bins=np.linspace(-0.1, 1.0, 100))
            fig = Patch()
            fig.data[0].x = ghist
            fig.data[0].y = gcount
            return fig

        def calc_peakval(radius, nradius, pbar):
            rmin, rmax = radius
            return model.calc_peakval(2 ** rmin, 2 ** rmax, nradius, "log", pbar)

        peak_div = Collapse(
            "peak",
            html.H2("Find peaks"),
            html.Div(
                style=two_column(1200),
                children=[
                    dbc.Label("Radius range"),
                    radius := dcc.RangeSlider(
                        min=0,
                        max=5,
                        step=0.1,
                        value=[1, 4],
                        marks={i: str(2**i) for i in range(6)},
                    ),
                    dbc.Label("Num radius"),
                    nradius := dcc.Slider(min=1, max=50, step=1, value=11),
                ],
            ),
            peak := ThreadButton(
                "PEAK",
                calc_peakval,
                State(radius, "value"),
                State(nradius, "value"),
            ),
            html.Div(
                style=two_column(1200),
                children=[
                    dbc.Label("Radius cut range"),
                    radius2 := dcc.RangeSlider(
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
                    thr := dcc.Slider(
                        id="thr", min=0.5, max=3, step=0.1, value=2.0
                    ),
                ],
            ),
            dbc.Row(
                children=[
                    dcc.Store("peak"),
                    dbc.Col(circle := dcc.Graph(figure=circle_fig("cell candidates"))),
                    dbc.Col(radius_intensity := dcc.Graph(figure=scatter_fig("radius-intensity", "radius", "intensity"))),
                ],
            ),
        )

        @callback(
            Output(radius2, "max"),
            Output(radius2, "marks"),
            Input(radius, "value"),
            Input(nradius, "value"),
        )
        def update_radius(radius, rnum):
            rmin, rmax = radius
            radius = np.power(2, np.linspace(rmin, rmax, rnum))
            marks = {i: f"{r:.2f}" for i, r in enumerate(radius)}
            return rnum - 1, marks

        @callback(
            Output("peak", "data"),
            peak.finish,
            State(radius2, "value"),
            State(thr, "value"),
        )
        def update_peak(finished, radius, thr):
            if not hasattr(model, "peakval"):
                return no_update
            model.calc_peaks(*radius, thr)
            return "finish"

        @callback(Output(circle, "figure"), Input("peak", "data"))
        def plot_circle(data):
            fig = Patch()
            fig.data[0].x = model.peaks.x
            fig.data[0].y = model.peaks.y
            fig.data[0].marker.opacity = model.peaks.v
            fig.data[0].marker.opacityref = 1 / model.peaks.v.max()
            fig.data[0].marker.size = model.peaks.r
            #fig.data[0].marker.sizeref = 2 * 500 / model.shape[0]
            fig.data[0].marker.sizemode = "diameter"
            fig.data[0].marker.sizemin = 1
            print(model.peaks)
            return update_img(fig, *model.shape)

        @callback(Output(radius_intensity, "figure"), Input("peak", "data"))
        def plot_radius_intensity(data):
            dr = model.radius[1] / model.radius[0] - 1
            v0 = model.peakval.v.ravel()
            m = (v0.size + 9999) // 10000
            v0 = v0[::m]
            r = (1 + 0.2 * dr * np.random.randn(v0.size))
            r0 = r * model.peakval.r.ravel()[::m]
            fig = Patch()
            fig.data[0].x = r0
            fig.data[0].y = v0
            fig.data[1].x = model.peaks.r
            fig.data[1].y = model.peaks.v
            return fig

        self.layout = html.Div([load_div, stats_div, frame_div, peak_div])
