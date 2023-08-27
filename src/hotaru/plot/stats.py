import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio

pio.kaleido.scope.mathjax = None


def plot_stats(paths, imgs, labels, dlabels, **kwargs):
    dfs = []
    for dlabel, path in zip(dlabels, paths):
        df = pd.DataFrame()
        for label, img in zip(labels, imgs):
            img = np.load(f"{path}/{img}.npy")
            df[label] = img.ravel()
        df["Data"] = dlabel
        dfs.append(df)
    df = pd.concat(dfs, axis=0)
    fig = px.scatter(
        df, x=labels[1], y=labels[2], color=labels[0], facet_col="Data", **kwargs
    )
    fig.update_xaxes(
        title_font_size=11,
    )
    fig.update_yaxes(
        title_font_size=11,
    )
    fig.update_layout(
        coloraxis_colorbar_title_font_size=11,
        template="none",
        font_size=11,
        margin=dict(l=50, r=20, t=20, b=50),
        annotations=[
            dict(font=dict(size=11), text=dlabels[i])
            for i in range(len(fig.layout.annotations))
        ],
    )
    return fig
