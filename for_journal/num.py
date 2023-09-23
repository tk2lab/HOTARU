import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from myload import load

dummy = go.Figure()
dummy.write_image("dummy.pdf", format="pdf")

la_list = [0.0, 0.02, 0.05, 0.1, 0.15]
lu_list = [0.0, 0.005, 0.01]


def compress_legend(fig):
    group1_base, group2_base = fig.data[0].name.split(",")
    lines_marker_name = []
    for i, trace in enumerate(fig.data):
        part1, part2 = trace.name.split(",")
        if part1 == group1_base:
            lines_marker_name.append(
                dict(
                    line=trace.line.to_plotly_json(),
                    marker=trace.marker.to_plotly_json(),
                    mode=trace.mode,
                    name=part2.lstrip(" "),
                )
            )
        if part2 == group2_base:
            trace["legend"] = "legend2"
            trace["legendgroup"] = "part2"
            trace["name"] = part1
        else:
            trace["name"] = ''
            trace["showlegend"] = False
    for lmn in lines_marker_name:
        lmn["line"]["color"] = "black"
        lmn["marker"]["color"] = "black"
        lmn["legend"] = "legend1"
        lmn["legendgroup"] = "part1"
        fig.add_trace(go.Scatter(y=[None], **lmn))
    fig.update_layout(
        legend=dict(
            title_text="$\\lambda_U$",
            x=0.7,
            y= 0.9,
        ),
        legend2=dict(
            title_text="$\\lambda_A$",
            x=0.4,
            y= 0.9,
        ),
    )


for data, init, clip in [
    ("Kd32", "min31", "clip2x2"),
    ("CA3", "min27", "clip3-100"),
]:
    out = []
    for la in la_list:
        #    for la in [0, 20, 40, 60, 80, 100]:
        for lu in lu_list:
            path = f"outputs/{data}/miniature/{init}/{clip}/su{lu}/sa{la}/pos0"
            for stage in range(30):
                try:
                    df = load(path, stage, df_only=True)
                    out.append((lu, la, stage, np.count_nonzero(df.kind == "cell")))
                except FileNotFoundError:
                    pass
    lu, la, stage, num = zip(*out)
    df = pd.DataFrame(dict(lu=lu, la=la, stage=stage, num=num))
    fig = px.line(
        df,
        x="stage",
        y="num",
        color="la",
        symbol="lu",
        template="none",
    )
    compress_legend(fig)
    fig.update_xaxes(
        autorange=False,
        range=(-0.5, 30.5),
        title_text="epoch",
    )
    fig.update_yaxes(
        autorange=False,
        range=(0, 1.05 * df.num.max()),
        title_text="# candidates",
    )
    fig.update_layout(
        font_size=11,
        width=300,
        height=300,
        margin=dict(l=35, r=5, t=5, b=25),
    )
    pio.full_figure_for_development(fig, warn=False)
    print(fig.data)
    print(fig.layout)
    fig.write_image(f"figs/{data}-num.pdf", engine="kaleido")
