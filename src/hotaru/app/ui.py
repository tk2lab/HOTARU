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


def ConfigStore():
    store = dcc.Store("config")

    @callback(
        Output(store, "data"),
        list(inputs.keys()),
        State(store, "data"),
    )
    def set_val(*args):
        *args, cfg = args


class ConfigInput(dbc.Input):

    def __init__(self, store, *label, **kwargs):
        self.cache = cache
        self.label = label
        for l in label:
            cache = cache[l]
        super().__init__(value=cache, **kwargs)

        @callback(
            Output(store, ""),
            Input(self, "value"),
            State(
        )
        def set_cfg(value):
            ls = self.label
            cfg = self.cache[ls[0]]
            c = cfg
            for l in ls[1:-1]:
                c = c[l]
            c[l[-1]] = value
            self.cache[ls[0]] = cfg
            return no_update


def ThreadButton(label, setter, func, *state):
    div = html.Div(
        children=[
            button := dbc.Button(label),
            pbar := dbc.Progress(),
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
        *(Input(s, "value") for s in state),
        prevent_initial_call=True,
    )
    def on_click(nc, ni, *state):
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
        else:
            setter(*state)
            return True, 0

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
