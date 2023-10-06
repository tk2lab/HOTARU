import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio

pio.kaleido.scope.mathjax = None

# from hotaru.plot.densesig import dense_sig_multi_fig


paths = dict(
    u0="outputs/Kd32/miniature/min31/clip2x2/u0/default/dup9/",
    u40="outputs/Kd32/miniature/min31/clip2x2/u40/default/dup9/",
)
stages = [[0, 1, 17], [0, 1, 14]]
thr_udense = 0.18

xmax = 0.5
ymax = 4.5

fig = go.Figure().set_subplots(2, 3, vertical_spacing=0.03, horizontal_spacing=0.02)
for i, (name, path) in enumerate(paths.items()):
    fig.add_annotation(
        text=name,
        x=0.15,
        y=4,
        xanchor="left",
        showarrow=False,
        col=1,
        row=1 + i,
    )
    for j, stage in enumerate(stages[i]):
        if stage == 0:
            pathx = f"{path}/../../{stage:03d}stats.csv"
        else:
            pathx = f"{path}/{stage:03d}stats.csv"
        df = pd.read_csv(pathx)
        df = df.query("kind != 'remove'")
        if j == 2:
            df.loc[df.udense > thr_udense, "kind"] = "background"
        fig.add_trace(
            go.Scatter(
                x=df.udense,
                y=df.signal,
                mode="markers",
                marker=dict(
                    color=[dict(cell="blue", background="red")[k] for k in df.kind],
                    size=3,
                    opacity=0.2,
                    line_width=0,
                )
            ),
            col=1 + j,
            row=1 + i,
        )
        fig.update_xaxes(
            title="spike density" if i == 1 else "",
            showticklabels=i == 1,
            range=(0, xmax),
            col=1 + j,
            row=1 + i,
        )
        fig.update_yaxes(
            title="signal" if j == 0 else "",
            showticklabels=j == 0,
            range=(0, ymax),
            col=1 + j,
            row=1 + i,
        )
        num = df.query("kind == 'cell'").shape[0]
        fig.add_annotation(
            text=f"epoch={stages[i][j]}, num={num}",
            x=0.2,
            y=3,
            xanchor="left",
            showarrow=False,
            col=1 + j,
            row=1 + i,
        )
fig.update_layout(
    template="none",
    showlegend=False,
    margin=dict(l=50, t=10, b=50, r=10),
    width=600,
    height=300,
    font_size=11,
)

fig.write_image("figs/Kd32-dense-sig.pdf")
