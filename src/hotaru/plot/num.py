import numpy as np
import plotly.express as px

from ..cui.common import all_stats


def cell_num_fig(cfg, **kwargs):
    kwargs.setdefault("template", "none")
    kwargs.setdefault("margin", dict(l=40, r=10, t=20, b=35))

    num = [np.count_nonzero(s.kind == "cell") for s in all_stats(cfg)]

    fig = px.line(x=np.arange(len(num)), y=num)
    fig.add_annotation(x=1, y=num[1], ax=2, text="A")
    fig.add_annotation(x=len(num)-1, y=num[-1], text="B")
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
