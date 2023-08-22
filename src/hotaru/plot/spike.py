import pandas as pd
import plotly.exporess as px
from PIL import Image

from .cui.common import load
from .common import to_image


def spike_image(cfg, stage, tselect=slice(None), uselect=slice(None)):
    u, _, _ = load(cfg, "temporal", stage)
    u /= u.max(axis=1, keepdims=True)
    u = u[tselect, uselect]
    imgs = to_image(u, "Reds")
    return Image.fromarray(imgs)


def spike_stats_fig(cfg, stages):
    dfs = []
    for stage in stages:
        stats, _ = load("evaluate", stage)
        stats["epoch"] = stage
        dfs.append(stats)
    stats = pd.concat(dfs, axis=0)
    fig = px.scatter(
        stats,
        x="signal",
        y="udense",
        facet_col="epoch",
        labels=dict(signal="signal intensity", udense="spike density"),
    )
    fig.update_layout(
        template="none",
    )
