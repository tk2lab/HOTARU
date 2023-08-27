import pandas as pd
import plotly.express as px
import plotly.io as pio

from ..cui.common import load

pio.kaleido.scope.mathjax = None


def dense_sig_fig(cfg, stages, thr_sig=0, label="", thr_udense=1.0):
    num = []
    dfs = []
    for stage in stages:
        df, _ = load(cfg, "evaluate", stage)
        df = df.query("kind != 'remove'")
        df.loc[df.udense > thr_udense, "kind"] = "background"
        df["col"] = stage
        df["size"] = 1
        df["symbol"] = 1
        dfs.append(df)
        num.append(df.query("kind == 'cell'").shape[0])
    df = pd.concat(dfs, axis=0)
    fig = px.scatter(
        df,
        x="udense",
        y="signal",
        color="kind",
        size="size",
        size_max=3,
        opacity=0.5,
        facet_col="col",
    )
    for i in range(len(fig.data)):
        fig.data[i].marker.line.width = 0

    for n, stage in zip(num, stages):
        fig.update_annotations(
            selector=dict(text=f"col={stage}"), y=0.8, text=f"{label}{stage}; num={n}"
        )
    fig.update_layout(
        template="none",
        font_size=11,
        showlegend=False,
        width=600,
        height=150,
        margin=dict(l=35, r=10, t=20, b=32),
    )
    return fig


def dense_sig_multi_fig(paths, stages, thr_sig=0, thr_udense=1.0):
    num = []
    dfs = []
    for i, (name, path) in enumerate(paths.items()):
        for j, stage in enumerate(stages[i]):
            if stage == 0:
                pathx = f"{path}/../../{stage:03d}stats.csv"
            else:
                pathx = f"{path}/{stage:03d}stats.csv"
            df = pd.read_csv(pathx)
            df = df.query("kind != 'remove'")
            df.loc[df.udense > thr_udense, "kind"] = "background"
            df["name"] = name
            df["stage"] = j
            df["size"] = 1
            df["symbol"] = 1
            dfs.append(df)
            num.append(df.query("kind == 'cell'").shape[0])
    df = pd.concat(dfs, axis=0)
    fig = px.scatter(
        df,
        x="udense",
        y="signal",
        color="kind",
        size="size",
        size_max=3,
        opacity=0.5,
        facet_col="stage",
        facet_row="name",
    )
    for i in range(len(fig.data)):
        fig.data[i].marker.line.width = 0
    for i, name in enumerate(paths.keys()):
        fig.update_annotation(
            selector=dict(text=f"naem={name}"),
            text=name,
        )
    for j, stage in enumerate(paths.keys()):
        fig.update_annotation(
            selector=dict(text=f"naem={name}"),
            text="",
        )
    fig.update_layout(
        template="none",
        font_size=11,
        showlegend=False,
        width=600,
        height=300,
        margin=dict(l=35, r=10, t=20, b=32),
    )
    return fig
