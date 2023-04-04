import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


class Layout:

    def __init__(self, model, ui_width=490):
        self.model = model
        self.ui_width = ui_width

    def heatmap(self, imgs, **kwargs):
        kwargs.setdefault("colorscale", "greens")
        kwargs.setdefault("colorbar", self.colorbar)
        return go.Heatmap(z=imgs, **kwargs)

    @property
    def width(self):
        return self.ui_width + 10

    @property
    def height(self):
        nt, h, w = self.model.shape
        return self.ui_width * h / w + 50

    @property
    def layout(self):
        return dict(
            margin=dict(l=0, r=0, t=0, b=0),
            width=self.width,
            height=self.height,
        )

    @property
    def image_layout(self):
        nt, h, w = self.model.shape
        return dict(
            **self.layout,
            xaxis=dict(visible=False, range=(0, w), domain=[0, 1 - 10 / self.width]),
            yaxis=dict(visible=False, range=(h, 0), domain=[50 / self.height, 1]),
        )

    @property
    def colorbar(self):
        return dict(orientation="h", yanchor="bottom", y=0, thickness=10)
