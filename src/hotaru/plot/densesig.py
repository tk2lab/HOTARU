import pandas as pd
import plotly.express as px
import plotly.io as pio

from ..cui.common import load

pio.kaleido.scope.mathjax = None


def dense_sig_fig(cfg, stages, thr_sig=0, label=""):
    dfs = []
    for stage in stages:
        df, _ = load(cfg, "evaluate", stage)
        df = df.query("kind != 'remove'")
        # fp = np.load(f"{path}/../../000footprint.npy")
        # df["intensity"] = df.signal * df.firmness
        df["col"] = stage
        df["size"] = 1
        df["symbol"] = 1
        dfs.append(df)
    df = pd.concat(dfs, axis=0)
    fig = px.scatter(
        df,
        x="udense",
        y="signal",
        color=df.signal < thr_sig,
        size="size",
        size_max=3,
        opacity=0.5,
        facet_col="col",
    )
    for i in range(len(fig.data)):
        fig.data[i].marker.line.width = 0

    for stage in stages:
        fig.update_annotations(
            selector=dict(text=f"col={stage}"), y=0.8, text=f"{label}{stage}"
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
