from threading import Thread

import dash.dcc as dcc
import dash.html as html
import dash_bootstrap_components as dbc
from dash import (
    Input,
    Output,
    State,
    callback,
    ctx,
    no_update,
)
from omegaconf import OmegaConf


class ProgressProxy:
    def __init__(self):
        self()

    def __call__(self, total=1, desc=None, postfix=None):
        self.desc = desc or ""
        self.postfix = postfix or ""
        self.total = total
        self.n = 0
        return self

    def set_description(self, desc):
        self.desc = desc

    def set_postfix_str(self, postfix):
        self.postfix

    def reset(self, total):
        self.total = total
        self.n = 0

    def update(self, n):
        self.n += n

    @property
    def value(self):
        return self.n / self.total

    @property
    def label(self):
        return f"{self.desc}: {int(100 * self.n / self.total)}% ({self.postfix})"


def two_column(width):
    return dict(
        width=f"{width}px",
        display="grid",
        gridTemplateColumns="auto 1fr",
    )


def ThreadButton(label, func, cfg, pbar):
    div = html.Div(
        children=[
            finish := dcc.Store(label),
            button := dbc.Button(label),
            interval := dcc.Interval(interval=100, disabled=True),
        ],
        style=two_column(1200),
    )
    jobs = [None, None]

    @callback(
        Output(finish, "data"),
        Output(interval, "disabled"),
        Output(pbar, "value"),
        Output(pbar, "label"),
        Input(button, "n_clicks"),
        Input(interval, "n_intervals"),
        State(cfg, "data"),
        prevent_initial_call=True,
    )
    def on_click(n_clicks, n_intervals, cfg):
        if ctx.triggered_id == button.id:
            cfg = OmegaConf.create(cfg)
            pbar = ProgressProxy()
            thread = Thread(target=func, kwargs=dict(cfg=cfg, pbar=pbar))
            jobs[:] = thread, pbar
            thread.start()
            return no_update, False, 0, pbar.label
        elif ctx.triggered_id == interval.id:
            thread, pbar = jobs
            if thread.is_alive():
                return no_update, no_update, pbar.value, pbar.label
            else:
                return "finish", True, 100, pbar.label

    div.finish = Input(finish, "data")
    return div


def Collapse(name, is_open, *children):
    div = html.Div(
        children=[
            button := dbc.Button(f"Open/Close {name}"),
            collapse := dbc.Collapse(children=children, is_open=is_open),
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
