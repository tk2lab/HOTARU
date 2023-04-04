from dash import (
    Dash,
    Input,
    Output,
    State,
    no_update,
    ctx,
)


class MyDash(Dash):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.job = {}

    def callback(self, *args, **kwargs):
        outputs = []
        inputs = []
        states = []
        for a in args:
            if isinstance(a, Output):
                outputs.append(a)
            elif isinstance(a, Input):
                inputs.append(a)
            elif isinstance(a, State):
                states.append(a)
        return super().callback(outputs, inputs, states, **kwargs)

    def collapse_callback(self, name):
        @self.callback(
            Input(f"{name}-collapse-button", "n_clicks"),
            State(f"{name}-collapse", "is_open"),
            Output(f"{name}-collapse", "is_open"),
        )
        def toggle_stats(n, is_open):
            return not is_open,

    def thread_callback(self, name, test_fn, target_fn, *args):
        @self.callback(
            Input(name, "n_clicks"),
            Output(f"{name}-submitted", "data"),
            *args,
            prevent_initial_call=True,
        )
        def on_click(click, *args):
            print("start", ctx.triggered_id, *args)
            if ctx.triggered_id == name:
                ok = test_fn(*args)
                if ok is None:
                    return no_update,
                if ok: 
                    uid = "saved"
                else:
                    thread, pbar = target_fn()
                    thread.start()
                    self.job[thread.native_id] = thread, pbar
                    uid = thread.native_id
            else:
                uid = "reset"
            print(uid)
            return dict(uid=uid),

        @self.callback(
            Input(f"{name}-submitted", "data"),
            Input(f"{name}-interval", "n_intervals"),
            Output(f"{name}-finished", "data"),
            Output(f"{name}-progress", "value"),
            Output(f"{name}-interval", "disabled"),
            prevent_initial_call=True,
        )
        def check(submitted, n):
            uid = submitted["uid"]
            print("check", uid)
            if uid == "reset":
                return "reset", 0, True
            thread, pbar = self.job.get(uid, (None, None))
            if uid == "saved" or not thread.is_alive():
                return "finished", 100, True
            return "continued", pbar.value, False
