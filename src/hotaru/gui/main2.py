import dash.dcc as dcc
import dash.html as html
import dash_bootstrap_components as dbc
import numpy as np
from dash import (
    Dash,
    Input,
    Output,
    Patch,
    State,
    callback,
)
from omegaconf import OmegaConf

from ..footprint.find import simple_peaks
from ..main import (
    data,
    find,
    get_frame,
    reduce,
    stats,
)
from .fig import (
    bar_fig,
    circle_fig,
    heat_fig,
    scatter_fig,
    update_img,
)
from .ui import (
    Collapse,
    ThreadButton,
    two_column,
)


def HotaruApp(cfg, *args, **kwargs):
    kwargs.setdefault("external_stylesheets", [dbc.themes.BOOTSTRAP])
    kwargs.setdefault("title", "HOTARU")
    app = Dash(__name__, *args, **kwargs)

    divs = [
        cfg_store := dcc.Store("cfg_store", data=OmegaConf.to_container(cfg)),
    ]

    load_div = Collapse(
        "load",
        True,
        pbar := dbc.Progress(),
        html.Div(
            style=two_column(500),
            children=[
                dbc.Label("Image file path"),
                imgs_path := dbc.Input(type="text", value=cfg.data.imgs.file),
                dbc.Label("Mask type"),
                imgs_mask := dbc.Input(type="text", value=cfg.data.mask.type),
                dbc.Label("Frequency"),
                imgs_hz := dbc.Input(type="number", value=cfg.data.hz),
                dbc.Label("Progress"),
            ],
        ),
        load_btn := ThreadButton("STATS", stats, cfg_store, pbar),
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
                dbc.Label("Radius cut range"),
                radius2 := dcc.RangeSlider(
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
                thr := dcc.Slider(id="thr", min=0.5, max=3, step=0.1, value=2.0),
            ],
        ),
    )

    stats_div = Collapse(
        "stats",
        False,
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
                dbc.Col(std_graph := dcc.Graph(figure=heat_fig("std")), width="auto"),
                dbc.Col(cor_graph := dcc.Graph(figure=heat_fig("cor")), width="auto"),
                dbc.Col(
                    std_cor_graph := dcc.Graph(
                        figure=scatter_fig("std-cor", "std", "cor")
                    ),
                    width="auto",
                ),
            ],
        ),
    )

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
                dbc.Col(
                    hist_img := dcc.Graph(
                        figure=bar_fig("intensity histogram", "intensity", "count")
                    ),
                    width="auto",
                ),
            ],
        ),
    )

    divs += [frame_div]

    @callback(
        Output(std_graph, "figure"),
        Output(cor_graph, "figure"),
        Output(std_cor_graph, "figure"),
        load_btn.finish,
        Input(gaussian, "value"),
        Input(maxpool, "value"),
        State(cfg_store, "data"),
        prevent_initial_call=True,
    )
    def plot_stats(finish, gaussian, maxpool, cfg):
        cfg = OmegaConf.create(cfg)
        imin, imax, istd, icor = stats(cfg)
        y, x = simple_peaks(icor, gaussian, maxpool)
        std_fig = Patch()
        std_fig.data[0].z = istd
        std_fig.data[0].zmin = istd.min()
        std_fig.data[0].zmax = istd.max()
        std_fig.data[1].x = x
        std_fig.data[1].y = y
        cor_fig = Patch()
        cor_fig.data[0].z = icor
        cor_fig.data[0].zmin = icor.min()
        cor_fig.data[0].zmax = icor.max()
        cor_fig.data[1].x = x
        cor_fig.data[1].y = y
        scatter = Patch()
        scatter.data[0].x = istd[y, x]
        scatter.data[0].y = icor[y, x]
        return (
            update_img(std_fig, *istd.shape),
            update_img(cor_fig, *icor.shape),
            scatter,
        )

    @callback(
        Output(frame, "max"),
        Output(frame, "marks"),
        Output(frame, "value"),
        Output(minmax, "min"),
        Output(minmax, "max"),
        Output(minmax, "marks"),
        load_btn.finish,
        State(cfg_store, "data"),
        prevent_initial_call=True,
    )
    def update_frame(finish, cfg):
        cfg = OmegaConf.create(cfg)
        nt = data(cfg).imgs.shape[0]
        imin, imax, istd, icor = stats(cfg)
        fmarks = {i: f"{i}" for i in range(0, nt + 1, nt // 10)}
        m = np.floor(10 * imin.min()) / 10
        M = np.ceil(10 * imax.max()) / 10
        mmarks = {float(v): f"{v:.1f}" for v in np.arange(m, M, 1)}
        return nt - 1, fmarks, 0, m, M, mmarks

    @callback(
        Output(norm_img, "figure"),
        Output(hist_img, "figure"),
        load_btn.finish,
        Input(frame, "value"),
        Input(minmax, "value"),
        State(cfg_store, "data"),
        prevent_initial_call=True,
    )
    def plot_norm(finish, t, minmax, cfg):
        cfg = OmegaConf.create(cfg)
        vmin, vmax = minmax
        img = get_frame(cfg, t)
        norm_fig = Patch()
        norm_fig.data[0].z = img
        norm_fig.data[0].zmin = vmin
        norm_fig.data[0].zmax = vmax
        gcount, ghist = np.histogram(img.ravel(), bins=np.linspace(-0.1, 5.0, 100))
        hist_fig = Patch()
        hist_fig.data[0].x = ghist
        hist_fig.data[0].y = gcount
        return update_img(norm_fig, *img.shape), hist_fig

    Collapse(
        "peak",
        True,
        html.H2("Find peaks"),
        find_btn := ThreadButton("PEAK", find, pbar),
        dbc.Row(
            children=[
                dcc.Store("peak"),
                dbc.Col(
                    circle := dcc.Graph(figure=circle_fig("cell candidates")),
                    width="auto",
                ),
                dbc.Col(
                    radius_intensity := dcc.Graph(
                        figure=scatter_fig("radius-intensity", "radius", "intensity")
                    ),
                    width="auto",
                ),
            ],
        ),
    )

    @callback(
        Output(circle, "figure"),
        Output(radius_intensity, "figure"),
        find_btn.finish,
        State(cfg_store, "data"),
        prevent_initial_call=True,
    )
    def plot_circle(finish, cfg):
        peaks = reduce(cfg)
        fig1 = Patch()
        fig1.data[0].x = peaks.x
        fig1.data[0].y = peaks.y
        fig1.data[0].marker.opacity = 0.5 * peaks.v / peaks.v.max()
        fig1.data[0].marker.size = peaks.r
        fig1.data[0].marker.sizeref = 0.25 * h / 500
        fig1.data[0].marker.sizemode = "diameter"

        find = find(cfg)
        dr = find.radius[1] / find.radius[0] - 1
        v0 = find.v.ravel()
        r0 = find.r.ravel()
        jitter0 = 1 + 0.2 * dr * np.random.randn(v0.size)
        jitter = 1 + 0.2 * dr * np.random.randn(reduce.r.size)
        fig2 = Patch()
        fig2.data[0].x = jitter0 * r0
        fig2.data[0].y = v0
        fig2.data[0].marker.maxdisplayed = 10000
        fig2.data[0].marker.opacity = 0.3
        fig2.data[1].x = jitter * peaks.r
        fig2.data[1].y = peaks.v
        fig2.data[1].marker.opacity = 0.3
        fig2.layout.xaxis.type = "log"
        return update_img(fig1, *shape), fig2

    """
    filt_div = Collapse(
        "filt",
        False,
        filt_btn := ThreadButton("TEST", filt, pbar),
        filt_graph := dbc.Row(
            children=[
                dbc.Col(filt_img := dcc.Graph(figure=heat_fig("filt", "Picnic", "green", -0.6, 0.6))),
            ],
        ),
    )

    @callback(
        Output(filt_img, "figure"),
        filt_btn.finish,
        prevent_initial_call=True,
    )
    def plot_filt(data):
        log_img = find_peaks(cur_img[None, ...], mask, radius)
        gl, t, y, x, r, v = cache["log_img"]
        nt, nr, h, w = gl.shape
        log_img = np.pad(gl[0, ...], [[0, 0], [0, 0], [2, 2]])
        log_img = np.concatenate(log_img, axis=1)
        fig = Patch()
        fig.data[0].z = log_img
        fig.data[1].x = 2 + (w + 4) * r + x
        fig.data[1].y = y
        return update_img(fig, *log_img.shape, 200)

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
    """

    @callback(
        Output(cfg_store, "data"),
        Output(radius2, "max"),
        Output(radius2, "marks"),
        Input(imgs_path, "value"),
        Input(imgs_mask, "value"),
        Input(imgs_hz, "value"),
        Input(radius, "value"),
        Input(nradius, "value"),
        Input(radius2, "value"),
        Input(thr, "value"),
        State(cfg_store, "data"),
    )
    def set_cfg(imgs, mask, hz, radius1, rnum, radius2, thr, cfg):
        rmin, rmax = radius
        rmin2, rmax2 = radius2
        radius = np.geomspace(rmin, rmax, rnum)
        marks = {i: f"{r:.2f}" for i, r in enumerate(radius)}
        cfg = OmegaConf.create(cfg)
        cfg.data.imgs.file = imgs
        cfg.data.mask.type = mask
        cfg.data.hz = hz
        cfg.radius.rmin = rmin
        cfg.radius.rmax = rmax
        cfg.radius.rnum = rnum
        cfg.radius.rmin2 = rmin2
        cfg.radius.rmax2 = rmax2
        cfg.radius.thr = thr
        return OmegaConf.to_container(cfg), rnum - 1, marks

    app.layout = html.Div(divs)
    return app
