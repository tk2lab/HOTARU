import plotly.graph_objects as go
import pandas as pd
import numpy as np

from ..cui.common import load
from .common import add_jitter


def compare_la_fig(paths, labels):
    fig = go.Figure().set_subplots(
        2,
        len(paths),
        row_heights=(1, 5),
        horizontal_spacing=0.05,
        vertical_spacing=0.01,
        #shared_xaxes="columns",
        #shared_yaxes="rows",
    )
    for i, path in enumerate(paths):
        stats = pd.read_csv(path, index_col=0)
        #df = stats.query("kind != 'remove'").copy()
        df = stats
        df = add_jitter(df)
        fig.add_trace(
            go.Histogram(x=df.firmness, marker_color="grey"),
            row=1,
            col=i + 1,
        )
        fig.add_trace(
            go.Scatter(
                y=df.old_udense,
                x=df.firmness,
                mode="markers",
                marker=dict(size=3, color="blue", opacity=0.5),
            ),
            row=2,
            col=i + 1,
        )
        fig.add_annotation(
            x=0.2,
            y=0.1,
            text=labels[i],
            showarrow=False,
            font_size=11,
            row=2,
            col=i + 1,
        )
        fig.update_xaxes(
            showticklabels=False,
            range=(0, 0.6),
            row=1,
            col=i + 1,
        )
        fig.update_yaxes(
            showticklabels=False,
            #range=(0, 0.15),
            row=1,
            col=i + 1,
        )
        fig.update_xaxes(
            range=(0, 0.6),
            tickmode="array",
            tickvals=[0, 0.3, 0.6],
            ticktext=[0, 0.3, 0.6],
            title_text="firmness",
            row=2,
            col=i + 1,
        )
        fig.update_yaxes(
            title_text="spike deinsty" if i ==0 else "",
            showticklabels=i == 0,
            range=(0, 0.15),
            #range=((4 / 5), (21 / 5)),
            #tickmode="array",
            #tickvals=np.log2([3, 6, 12]),
            #ticktext=["3", "6", "12"],
            row=2,
            col=i + 1,
        )
    fig.update_layout(
        template="none",
        font_size=11,
        width=600,
        height=200,
        showlegend=False,
        margin=dict(t=0, r=10, l=40, b=40),
    )
    return fig
