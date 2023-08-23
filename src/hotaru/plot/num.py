import numpy as np
import plotly.express as px

from ..cui.common import load


def cell_num_fig(cfg, **kwargs):
    kwargs.setdefault("template", "none")
    kwargs.setdefault("margin", dict(l=40, r=10, t=20, b=35))

    num = []
    for stage in range(1000):
        stats, _ = load(cfg, "evaluate", stage)
        if stats is None:
            break
        num.append(np.count_nonzero(stats.kind == "cell"))

    fig = px.line(x=np.arange(len(num)), y=num)
    fig.update_xaxes(
        title_text="epoch",
        title_font_size=11,
    )
    fig.update_yaxes(
        title_text="Number of cells",
        title_font_size=11,
    )
    fig.update_layout(**kwargs)
    return fig
