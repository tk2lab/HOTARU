import plotly.graph_objects as go
import numpy as np

from ..cui.common import load
from .common import add_jitter


def run_fig(
    cfg,
    stages,
    thr_f=0.35,
    thr_d=0.11,
    min_val=(0.25, 0),
    max_val=(0.58, 0.16),
    min_r=1,
    max_r=1,
):
    def get_color(x):
        if x.radius < 2.1:
            return "blue"
        elif (
            x.kind == "background"
            or x.radius > 15.9
            or x.old_udense > thr_d
            or x.firmness < thr_f
        ):
            return "red"
        else:
            return "green"

    fig = go.Figure().set_subplots(
        2 * len(stages),
        3,
        column_widths=(5, 5, 1),
        row_heights=(1, 5) * len(stages),
        horizontal_spacing=0.05,
        vertical_spacing=0.01,
        # shared_yaxes=True,
    )
    for i, stage in enumerate(stages):
        stats, _ = load(cfg, "evaluate", stage)
        df = stats.query("kind != 'remove'").copy()
        df = add_jitter(df)
        cl = df.query("kind == 'cell'")
        color = df.apply(get_color, axis=1)
        fig.add_trace(
            go.Histogram(x=df.firmness, marker_color="grey"),
            row=2 * i + 1,
            col=1,
        )
        fig.add_trace(
            go.Histogram(x=df.old_udense, marker_color="grey"),
            row=2 * i + 1,
            col=2,
        )
        fig.add_trace(
            go.Scatter(
                y=df.lri,
                x=df.firmness,
                mode="markers",
                marker=dict(size=3, color=color, opacity=0.5),
            ),
            row=2 * i + 2,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                y=df.lri,
                x=df.old_udense,
                mode="markers",
                marker=dict(size=3, color=color, opacity=0.5),
            ),
            row=2 * i + 2,
            col=2,
        )
        fig.add_trace(
            go.Histogram(y=cl.lri, marker_color="grey"),
            row=2 * i + 2,
            col=3,
        )
        fig.add_hline(y=(1 + min_r / 5), line_color="blue", row=2 * i + 2)
        fig.add_hline(y=(4 - max_r / 5), line_color="red", row=2 * i + 2)
        for r in range(2):
            fig.add_vline(thr_f, line_color="red", col=1, row=2 * i + r + 1)
            fig.add_vline(thr_d, line_color="red", col=2, row=2 * i + r + 1)
            for c in range(2):
                fig.update_xaxes(
                    range=(min_val[r], max_val[r]),
                    row=2 * i + 1,
                    col=1,
                )
                fig.update_xaxes(
                    range=(min_val[r], max_val[r]),
                    row=2 * i + 1,
                    col=2,
                )
        for c in range(3):
            fig.update_xaxes(
                showticklabels=False,
                row=2 * i + 1,
                col=1 + c,
            )
            fig.update_yaxes(
                showticklabels=False,
                row=2 * i + 1,
                col=1 + c,
            )
            fig.update_yaxes(
                title_text="radius" if c == 0 else "",
                title_font_size=11,
                showticklabels=c == 0,
                range=((4 / 5), (21 / 5)),
                tickmode="array",
                tickvals=np.log2([3, 6, 12]),
                ticktext=["3", "6", "12"],
                row=2 * i + 2,
                col=1 + c,
            )
            fig.update_xaxes(
                showticklabels=False,
                row=2 * i + 2,
                col=3,
            )
            fig.update_xaxes(
                showticklabels=False,
                range=(min_val[0], max_val[0]),
                row=2 * i + 1,
                col=1,
            )
            fig.update_xaxes(
                showticklabels=False,
                range=(min_val[1], max_val[1]),
                row=2 * i + 1,
                col=2,
            )
            bottom = i == len(stages) - 1
            fig.update_xaxes(
                showticklabels=bottom,
                range=(min_val[0], max_val[0]),
                title_text="firmness" if bottom else "",
                title_font_size=11,
                row=2 * i + 2,
                col=1,
            )
            fig.update_xaxes(
                showticklabels=bottom,
                range=(min_val[1], max_val[1]),
                title_text="spike density" if bottom else "",
                title_font_size=11,
                row=2 * i + 2,
                col=2,
            )
    fig.add_annotation(
        x=0.02,
        y=0.98,
        text="A",
        xref="paper",
        yref="paper",
        showarrow=False,
        font_size=13,
    )
    fig.add_annotation(
        x=0.02,
        y=0.48,
        text="B",
        xref="paper",
        yref="paper",
        showarrow=False,
        font_size=13,
    )
    fig.update_layout(
        template="none",
        font_size=11,
        width=600,
        height=200 * len(stages),
        showlegend=False,
        margin=dict(t=0, r=0, l=40, b=40),
    )
    return fig
