import dash_bootstrap_components as dbc
import jax
import jax.numpy as jnp
import numpy as np
from omegaconf import OmegaConf
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

from ..footprint.find import find_peaks
from ..main import (
    stats,
    data,
    find,
    reduce,
    make,
    spike,
    footprint,
    clean,
)
from .ui import (
    two_column,
    ConfigInput,
    ThreadButton,
    Collapse,
)
from .fig import (
    heat_fig,
    spike_fig,
    update_img,
    scatter_fig,
    bar_fig,
    circle_fig,
)


def HotaruApp(cfg, *args, **kwargs):

    kwargs.setdefault("external_stylesheets", [dbc.themes.BOOTSTRAP])
    kwargs.setdefault("title", "HOTARU")
    app = Dash(__name__, *args, **kwargs)

    divs = [cfg := dcc.Store("cfg_store", data=OmegaConf.to_container(cfg))]

    load_div = Collapse(
        "load",
        True,
        html.Div(
            style=two_column(500),
            children=[
                dbc.Label("Image file path"),
                imgs_path := ConfigInput(cfg, "data", "imgs", type="text"),
                dbc.Label("Mask type"),
                imgs_mask := ConfigInput(cfg, "data", "mask", type="text"),
                dbc.Label("Frequency"),
                imgs_hz := ConfigInput(cfg, "data", "hz", type="number"),
                dbc.Label("Progress"),
                pbar := dbc.Progress(),
            ],
        ),
    )

    stats_div = Collapse(
        "stats",
        False,
        html.H2("Conventional cell candidates detection"),
        load := ThreadButton("LOAD", stats, pbar),
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
                dbc.Col(std_graph := dcc.Graph(figure=heat_fig("std")), width="auto"),
                dbc.Col(cor_graph := dcc.Graph(figure=heat_fig("cor")), width="auto"),
                dbc.Col(std_cor_graph := dcc.Graph(figure=scatter_fig("std-cor", "std", "cor")), width="auto"),
            ],
        )
    )

    @callback(
        Output(std_graph, "figure"),
        Output(cor_graph, "figure"),
        Output(std_cor_graph, "figure"),
        load.finish,
        Input(gaussian, "value"),
        Input(maxpool, "value"),
        State(cfg, "data"),
        prevent_initial_call=True,
    )
    def plot_std(finish, gaussina, maxpool, cfg):
        cfg = OmegaConf.create(cfg)
        imin, imax, istd, icor = stats(cfg)
        std_fig = Patch()
        std_fig.data[0].z = istd
        std_fig.data[0].zmin = istd.min()
        std_fig.data[0].zmax = istd.max()
        #fig.data[1].x = x
        #fig.data[1].y = y
        cor_fig = Patch()
        cor_fig.data[0].z = icor
        cor_fig.data[0].zmin = icor.min()
        cor_fig.data[0].zmax = icor.max()
        #fig.data[1].x = x
        #fig.data[1].y = y
        scatter = Patch()
        scatter.data[0].x = stats.istd[y, x]
        scatter.data[0].y = stats.icor[y, x]
        return update_img(std_fig, *istd.shape), update_img(cor_fig, *icor.shape), scatter

    divs += [load_div, stats_div]

    frame_div = Collapse(
        "frame",
        False,
        html.H2("Check frames"),
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
            ],
        ),
        frame_graph := dbc.Row(
            children=[
                dbc.Col(norm_img := dcc.Graph(figure=heat_fig("norm")), width="auto"),
                dbc.Col(hist_img := dcc.Graph(figure=bar_fig("intensity histogram", "intensity", "count")), width="auto"),
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
        nt = model().nt
        fmarks = {i: f"{i}" for i in range(0, nt + 1, nt // 10)}
        m = np.floor(10 * model().min) / 10
        M = np.ceil(10 * model().max) / 10
        mmarks = {float(v): f"{v:.1f}" for v in np.arange(m, M, 1)}
        return nt - 1, fmarks, 0, m, M, mmarks

    @callback(
        Output(norm_img, "figure"),
        load.finish,
        Input(frame, "value"),
        Input(minmax, "value"),
        prevent_initial_call=True,
    )
    def plot_norm(data, t, minmax):
        cur_img = model().frame(t)
        cache["cur_img"] = cur_img
        vmin, vmax = minmax
        fig = Patch()
        fig.data[0].z = cur_img
        fig.data[0].zmin = vmin 
        fig.data[0].zmax = vmax
        return update_img(fig, *cur_img.shape)

    @callback(
        Output(hist_img, "figure"),
        Input(norm_img, "figure"),
        prevent_initial_call=True,
    )
    def plot_hist(data):
        cur_img = cache["cur_img"]
        gcount, ghist = np.histogram(cur_img.ravel(), bins=np.linspace(-0.1, 5.0, 100))
        fig = Patch()
        fig.data[0].x = ghist
        fig.data[0].y = gcount
        return fig

    divs += [frame_div]

    def set_filt_params(radius, rnum):
        rmin, rmax = np.power(2, radius)
        cache["radius"] = np.geomspace(rmin, rmax, rnum)

    def exec_filt(pbar):
        cur_img = cache["cur_img"]
        radius = cache["radius"]
        mask = model().mask
        log_img = find_peaks(cur_img[None, ...], mask, radius)
        cache["log_img"] = log_img

    def set_peaks_params(radius, nradius):
        cfg = cache["cfg"]
        cfg.radius.rmin = radius[0]
        cfg.radius.rmax = radius[1]
        cfg.radius.rnum = nradius
        cache["cfg"] = cfg

    find_div = Collapse(
        "peak",
        True,
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
        Collapse(
            "filt",
            False,
            filt := ThreadButton(
                "TEST",
                set_filt_params,
                exec_filt,
                radius,
                nradius,
            ),
            filt_graph := dbc.Row(
                children=[
                    dbc.Col(filt_img := dcc.Graph(figure=heat_fig("filt", "Picnic", "green", -0.6, 0.6))),
                ],
            ),
        ),
        peak := ThreadButton(
            "PEAK",
            set_peaks_params,
            model().base_peaks,
            radius,
            nradius,
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
                dbc.Col(circle := dcc.Graph(figure=circle_fig("cell candidates")), width="auto"),
                dbc.Col(radius_intensity := dcc.Graph(figure=scatter_fig("radius-intensity", "radius", "intensity")), width="auto"),
            ],
        ),
    )

    @callback(
        Output(filt_img, "figure"),
        filt.finish,
        prevent_initial_call=True,
    )
    def plot_filt(data):
        gl, t, y, x, r, v = cache["log_img"]
        nt, nr, h, w = gl.shape
        log_img = np.pad(gl[0, ...], [[0, 0], [0, 0], [2, 2]])
        log_img = np.concatenate(log_img, axis=1)
        fig = Patch()
        fig.data[0].z = log_img
        fig.data[1].x = 2 + (w + 4) * r + x
        fig.data[1].y = y
        return update_img(fig, *log_img.shape, 200)

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
        Input(radius2, "value"),
        Input(thr, "value"),
        prevent_initial_call=True,
    )
    def update_peak(finished, radius2, thr):
        cfg = cache["cfg"]
        cfg.radius.rmin2 = radius2[0]
        cfg.radius.rmax2 = radius2[1]
        cfg.radius.thr = thr
        cache["cfg"] = cfg
        model().reduced_peaks()
        return "finish"

    @callback(
        Output(circle, "figure"),
        Input("peak", "data"),
        prevent_initial_call=True,
    )
    def plot_circle(data):
        h, w = model().shape
        peaks = model().reduced_peaks()
        fig = Patch()
        fig.data[0].x = peaks.x
        fig.data[0].y = peaks.y
        fig.data[0].marker.opacity = 0.5 * peaks.v / peaks.v.max()
        fig.data[0].marker.size = peaks.r
        fig.data[0].marker.sizeref = 0.25 * h / 500
        fig.data[0].marker.sizemode = "diameter"
        return update_img(fig, *model().shape)

    @callback(
        Output(radius_intensity, "figure"),
        Input("peak", "data"),
        prevent_initial_call=True,
    )
    def plot_radius_intensity(data):
        find = model().base_peaks()
        reduce = model().reduced_peaks()
        dr = find.radius[1] / find.radius[0] - 1
        v0 = find.v.ravel()
        r0 = find.r.ravel()
        jitter0 = (1 + 0.2 * dr * np.random.randn(v0.size))
        jitter = (1 + 0.2 * dr * np.random.randn(reduce.r.size))
        fig = Patch()
        fig.data[0].x = jitter0 * r0
        fig.data[0].y = v0
        fig.data[0].marker.maxdisplayed = 10000
        fig.data[0].marker.opacity = 0.3
        fig.data[1].x = jitter * reduce.r
        fig.data[1].y = reduce.v
        fig.data[1].marker.opacity = 0.3
        fig.layout.xaxis.type = "log"
        return fig

    make_div = Collapse(
        "make",
        True,
        html.H2("Initial step"),
        make := ThreadButton(
            "FOOTPRINT",
            lambda: None,
            model().segments,
        ),
        spike := ThreadButton(
            "SPIKE",
            lambda: None,
            lambda pbar: model().spikes(True, pbar),
        ),
        dbc.Row(
            children=[
                dbc.Col(seg_img := dcc.Graph(figure=heat_fig("initial footprints")), width="auto"),
                dbc.Col(spk_img := dcc.Graph(figure=spike_fig("initial spikes", cache["cfg"].data.hz)), width="auto"),
            ],
        ),
    )

    @callback(
        Output(seg_img, "figure"),
        make.finish,
        prevent_initial_call=True,
    )
    def plot_seg(data):
        seg = model().segments().max(axis=0)
        fig = Patch()
        fig.data[0].z = seg
        return update_img(fig, *seg.shape)

    @callback(
        Output(spk_img, "figure"),
        spike.finish,
        prevent_initial_call=True,
    )
    def plot_spk(data):
        spk = model().spikes(True)
        print(spk.shape)
        #spk /= spk.max(axis=0, keepdims=True)
        fig = Patch()
        fig.data[0].z = spk[0]
        return fig

    divs += [find_div, make_div]

    app.layout = html.Div(divs)
    return app
