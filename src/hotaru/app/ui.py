from threading import Thread

import dash_bootstrap_components as dbc
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


class Progress:

    def __call__(self, total):
        self.total = total
        self.n = 0
        return self

    def update(self, n):
        self.n += n

    @property
    def value(self):
        return self.n / self.total


def two_column(width):
    return dict(
        width=f"{width}px",
        display="grid",
        gridTemplateColumns="auto 1fr",
    )


class ConfigInput(dbc.Input):

    def __init__(self, cfg, *label, **kwargs):
        super().__init__(**kwargs)

        @callback(
            Output(cfg, "data"),
            Input(cfg, "value"),
            State(cfg, "data"),
        )
        def set_cfg(value, cfg):
            cfg = OmegaConfcfg.create(cfg)
            c = cfg
            for l in ls[:-1]:
                c = c[l]
            c[l[-1]] = value
            return OmegaConfig.to_container(cfg)


def ThreadButton(label, func, pbar):
    div = html.Div(
        children=[
            button := dbc.Button(label),
            interval := dcc.Interval(interval=100, disabled=True),
        ],
        style=two_column(1200),
    )
    jobs = [None, None]

    @callback(
        Output(interval, "disabled"),
        Output(pbar, "value"),
        Input(button, "n_clicks"),
        Input(interval, "n_intervals"),
        prevent_initial_call=True,
    )
    def on_click(n_clicks, n_intervals):
        if ctx.triggered_id == button.id:
            pbar = Progress()
            thread = Thread(target=func, kwargs=dict(pbar=pbar))
            jobs[:] = thread, pbar
            thread.start()
            return False, 0
        elif ctx.triggered_id == interval.id:
            thread, pbar = jobs
            if thread.is_alive():
                return no_update, pbar.value
            else:
                return True, 100

    div.finish = Input(interval, "disabled")
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
