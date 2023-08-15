import dash.dcc as dcc
import dash.html as html
import dash_bootstrap_components as dbc
import numpy as np
from dash import (
    Input,
    Output,
    State,
    callback,
)
from omegaconf import OmegaConf


def InputGroup(*components, **kwargs):
    style = kwargs.setdefault("style", {})
    style.update(
        display="grid",
        gridTemplateColumns="auto 1fr",
    )
    return html.Div(
        children=sum([[dbc.Label(k), v] for k, v in components], []),
        **kwargs,
    )


def create_config_div(cfg):
    def Radius1Slider(value, rrange, skip=10, **kwargs):
        return dcc.RangeSlider(
            min=0,
            max=len(rrange) - 1,
            value=[np.searchsorted(rrange, v) for v in value],
            marks={i: f"{rrange[i]:.2f}" for i in range(0, len(rrange), skip)},
            **kwargs,
        )

    radius_range = np.geomspace(2, 32, 41)

    imgs_path = dbc.Input(type="text", value=cfg.data.imgs.file)
    imgs_mask = dbc.Input(type="text", value=cfg.data.mask.type)
    imgs_hz = dbc.Input(type="number", value=cfg.data.hz)
    radius1 = Radius1Slider([cfg.radius.rmin, cfg.radius.rmax], radius_range)
    nradius = dcc.Slider(min=5, max=30, step=1, value=cfg.radius.rnum)
    radius2 = dcc.RangeSlider(
        min=0,
        max=cfg.radius.rnum - 1,
        step=1,
        value=[cfg.radius.rmin2, cfg.radius.rmax2],
    )
    thr = dcc.Slider(id="thr", min=0.5, max=3, step=0.1, value=cfg.radius.thr)
    tau = dcc.RangeSlider(
        min=0,
        max=500,
        step=10,
        value=[1000 * cfg.dynamics.tau1, 1000 * cfg.dynamics.tau2],
        marks={i: str(i) for i in range(0, 501, 100)},
    )
    duration = dcc.Slider(min=0.0, max=3.0, value=cfg.dynamics.duration)
    la = dcc.Slider(min=0.0, max=100.0, value=cfg.penalty.la)
    lu = dcc.Slider(min=0.0, max=100.0, value=cfg.penalty.lu)
    bx = dcc.Slider(min=0.0, max=1.0, value=cfg.penalty.bx)
    bt = dcc.Slider(min=0.0, max=1.0, value=cfg.penalty.bt)

    div = html.Div(
        children=[
            cfg_store := dcc.Store("cfg_store", data=OmegaConf.to_container(cfg)),
            InputGroup(
                ("Image file path", imgs_path),
                ("Mask type", imgs_mask),
                ("Frequency", imgs_hz),
                style=dict(width="500px"),
            ),
            InputGroup(
                ("Radius range (px)", radius1),
                ("Num radius", nradius),
                ("Radius cut range", radius2),
                ("Distance threshold", thr),
                ("Time constants (msec)", tau),
                ("Duration (sec)", duration),
                ("Footprint sparseness", la),
                ("Spike sparseness", lu),
                ("Spatial background", bx),
                ("Temporal background", bt),
                style=dict(width="1200px"),
            ),
        ],
    )

    @callback(
        Output(cfg_store, "data"),
        Input(imgs_path, "value"),
        Input(imgs_mask, "value"),
        Input(imgs_hz, "value"),
        Input(radius1, "value"),
        Input(nradius, "value"),
        Input(radius2, "value"),
        Input(thr, "value"),
        Input(tau, "value"),
        Input(duration, "value"),
        Input(la, "value"),
        Input(lu, "value"),
        Input(bx, "value"),
        Input(bt, "value"),
        State(cfg_store, "data"),
    )
    def set_cfg(
        imgs, mask, hz, radius1, rnum, radius2, thr, tau, duration, la, lu, bx, bt, cfg
    ):
        rmin1, rmax1 = [radius_range[r] for r in radius1]
        rmin2, rmax2 = radius2
        tau1, tau2 = tau
        cfg = OmegaConf.create(cfg)
        cfg.data.imgs.file = imgs
        cfg.data.mask.type = mask
        cfg.data.hz = hz
        cfg.radius.rmin = float(rmin1)
        cfg.radius.rmax = float(rmax1)
        cfg.radius.rnum = rnum
        cfg.radius.rmin2 = rmin2
        cfg.radius.rmax2 = rmax2
        cfg.radius.thr = thr
        cfg.dynamics.tau1 = tau1 / 1000
        cfg.dynamics.tau2 = tau2 / 1000
        cfg.dynamics.duration = duration
        cfg.dynamics.duration = duration
        cfg.penalty.la = la
        cfg.penalty.lu = lu
        cfg.penalty.bx = bx
        cfg.penalty.bt = bt
        return OmegaConf.to_container(cfg)

    @callback(
        Output(radius2, "max"),
        Output(radius2, "marks"),
        Input(radius1, "value"),
        Input(nradius, "value"),
    )
    def set_radius2(radius1, rnum):
        rmin1, rmax1 = radius1
        rmin1 = radius_range[rmin1]
        rmax1 = radius_range[rmax1]
        radius = np.geomspace(rmin1, rmax1, rnum)
        marks = {i: f"{r:.2f}" for i, r in enumerate(radius)}
        return rnum - 1, marks

    div.store = cfg_store
    return div
