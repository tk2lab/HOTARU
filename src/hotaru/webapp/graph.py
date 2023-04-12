import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ..jax.filter.gaussian import gaussian
from ..jax.filter.laplace import gaussian_laplace
from ..jax.filter.pool import max_pool


class Graph:
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
        h, w = self.model.shape
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
        h, w = self.model.shape
        return dict(
            **self.layout,
            xaxis=dict(visible=False, range=(0, w), domain=[0, 1 - 10 / self.width]),
            yaxis=dict(visible=False, range=(h, 0), domain=[50 / self.height, 1]),
        )

    @property
    def colorbar(self):
        return dict(orientation="h", yanchor="bottom", y=0, thickness=10)

    def plot_stats(self, gauss, maxpool):
        mini, maxi, stdi, cori = self.model.istats
        if gauss > 0:
            fil_cori = gaussian(cori[None, ...], gauss)[0]
        else:
            fil_cori = cori
        max_cori = max_pool(fil_cori, (maxpool, maxpool), (1, 1), "same")
        y, x = np.where(fil_cori == max_cori)
        stdv = stdi[y, x]
        corv = cori[y, x]
        style = dict(display="flex")
        df = pd.DataFrame(dict(x=x, y=y))
        stdfig = px.scatter(
            df, x="x", y="y", opacity=0.3, color_discrete_sequence=["red"]
        )
        stdfig.add_trace(self.heatmap(stdi))
        stdfig.update_layout(
            title=dict(x=0.01, y=0.99, text="std"),
            **self.image_layout,
        )
        corfig = px.scatter(
            df, x="x", y="y", opacity=0.3, color_discrete_sequence=["red"]
        )
        corfig.add_trace(self.heatmap(cori))
        corfig.update_layout(
            title=dict(x=0.01, y=0.99, text="cor"),
            **self.image_layout,
        )
        stdcor = go.Figure(
            [
                go.Scatter(
                    x=stdv,
                    y=corv,
                    mode="markers",
                    marker=dict(color="red", opacity=0.5),
                    showlegend=False,
                )
            ],
            self.layout,
        )
        stdcor.update_layout(
            xaxis=dict(title="std"),
            yaxis=dict(title="cor"),
        )
        return stdfig, corfig, stdcor

    def plot_frame(self, t, radius, minmax):
        radius = np.power(2, radius)
        vmin, vmax = minmax
        img = self.model.frame(t)
        log = gaussian_laplace(img[None, ...], radius)[0]
        gcount, ghist = np.histogram(log.ravel(), bins=np.linspace(-0.1, 1.0, 100))
        imgfig = go.Figure([self.heatmap(img, zmin=vmin, zmax=vmax)], self.image_layout)
        logfig = go.Figure(
            [self.heatmap(log, zmin=-0.4, zmax=0.4, colorscale="Picnic")],
            self.image_layout,
        )
        histfig = go.Figure(
            go.Bar(x=ghist, y=gcount, width=ghist[1] - ghist[0]), self.layout
        )
        histfig.update_layout(
            xaxis=dict(title="intensity"),
            yaxis=dict(title="count"),
        )
        return imgfig, logfig, histfig

    def plot_peak(self):
        dr = self.model.radius[1] / self.model.radius[0] - 1
        h, w = self.model.shape
        peaks = self.model.peaks.copy()
        nc = peaks.shape[0]
        peaks["id"] = peaks.index
        peaks["r"] = (1 + 0.2 * dr * np.random.randn(nc)) * peaks.r
        peaks["size"] = 2 * (self.width / w) * peaks.r
        peaks["opacity"] = peaks.v / peaks.v.max()
        m = (h * w + 9999) // 10000
        v0 = self.model.peakval.val.ravel()[::m]
        r0 = (1 + 0.2 * dr * np.random.randn(v0.size)) * self.model.peakval.r.ravel()[
            ::m
        ]
        circle = px.scatter(
            peaks,
            x="x",
            y="y",
            size="size",
            color_discrete_sequence=["green"],
            opacity=peaks.opacity,
            hover_name="id",
            hover_data=dict(size=False, x=True, y=True, r=":.2f", v=":.2f"),
        )
        circle.update_layout(**self.image_layout)
        scatter = px.scatter(
            peaks,
            x="r",
            y="v",
            symbol_sequence=["diamond"],
            color_discrete_sequence=["green"],
            size_max=20,
            opacity=0.5,
            hover_name="id",
            hover_data=dict(size=False, x=True, y=True, r=":.2f", v=":.2f"),
        )
        scatter.add_trace(
            go.Scatter(
                x=r0,
                y=v0,
                mode="markers",
                marker=dict(color="red", opacity=0.2),
                showlegend=False,
            )
        )
        scatter.update_layout(
            xaxis=dict(title="radius", type="log"),
            yaxis=dict(title="intensity", rangemode="tozero"),
            **self.layout,
        )
        print("peak done")
        return circle, scatter

    def plot_all(self):
        data = self.model.footprints
        nc = data.shape[0]
        peaks = self.model.peaks.copy()
        peaks["id"] = peaks.index
        bg = self.heatmap(data.max(axis=0))
        circle = go.Scatter(
            x=peaks.x,
            y=peaks.y,
            mode="markers",
            showlegend=False,
            customdata=peaks[["id", "r", "v"]],
            hovertemplate="id:%{customdata[0]}, x:%{x}, y:%{y}, r:%{customdata[1]:.3f}, v:%{customdata[2]:.3f}",
        )
        return go.Figure([bg, circle], self.image_layout)

    def plot_single(self, select):
        single = go.Figure(
            [self.heatmap(self.model.footprints[select])], self.image_layout
        )
        return single
