import time
import diskcache

import dash
import dash_bootstrap_components as dbc
from dash import DiskcacheManager, Input, Output, html, dcc


cache = diskcache.Cache("./cache")
background_callback_manager = DiskcacheManager(cache)

app = dash.Dash(
    __name__,
    background_callback_manager=background_callback_manager,
)

app.layout = html.Div(
    [
        button := dbc.Button("TEST"),
        pbar := html.Div(),
        count := html.Div(),
        out := html.Div(),
    ]
)

@dash.callback(
    Output(out, "children"),
    Input(button, "n_clicks"),
    background=True,
    running=[
        (Output(button, "disabled"), True, False),
    ],
    interval=100,
    progress=Output(pbar, "children"),
    progress_default="FINISH",
    prevent_initial_call=True,
)
def on_click(set_progress, n):
    print("click", n)
    set_progress("0/3")
    time.sleep(1.0)
    set_progress("1/3")
    time.sleep(1.0)
    set_progress("2/3")
    time.sleep(1.0)
    print("finish", n)
    return "FINISH!"

app.run_server(debug=True, port=8888)
