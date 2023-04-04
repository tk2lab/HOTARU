import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ..filter.gaussian import gaussian
from ..filter.pool import max_pool


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
        maxi, stdi, cori = self.model.istats
        if gauss > 0:
            fil_cori = gaussian(cori[None, ...], gauss)[0]
        else:
            fil_cori = cori
        max_cori = max_pool(fil_cori, (maxpool, maxpool), (1, 1), "same")
        y, x = np.where(fil_cori == max_cori)
        stdv = stdi[y, x]
        corv = cori[y, x]
        style = dict(display="flex")
        scatter = go.Scatter(x=x, y=y, mode="markers", marker=dict(opacity=0.3), showlegend=False)
        stdfig = go.Figure([self.heatmap(stdi), scatter], self.image_layout)
        stdfig.update_layout(
            title=dict(x=0.01, y=0.99, text="std"),
        )
        corfig = go.Figure([self.heatmap(cori), scatter], self.image_layout)
        corfig.update_layout(
            title=dict(x=0.01, y=0.99, text="cor"),
        )
        stdcor = go.Figure([go.Scatter(x=stdv, y=corv, mode="markers", showlegend=False)], self.layout)
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
        gcount, ghist = np.histogram(
            log.ravel(), bins=np.linspace(-0.1, 1.0, 100)
        )
        imgfig = go.Figure([self.heatmap(img, zmin=vmin, zmax=vmax)], self.image_layout)
        logfig = go.Figure([self.heatmap(log, zmin=-0.4, zmax=0.4, colorscale="Picnic")], self.image_layout)
        histfig = go.Figure(go.Bar(x=ghist, y=gcount, width=ghist[1] - ghist[0]), self.layout)
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
        r = 2 * (self.width / w) * peaks.r
        v = peaks.v / peaks.v.max()
        peaks["id"] = peaks.index
        cells = go.Scatter(
            x=peaks.x,
            y=peaks.y,
            mode="markers",
            marker=dict(size=r, opacity=v),
            showlegend=False,
            customdata=peaks[["id", "r", "v"]],
            hovertemplate="id:%{customdata[0]}, x:%{x}, y:%{y}, r:%{customdata[1]:.3f}, v:%{customdata[2]:.3f}",
        )
        v0 = self.model.peakval.val.ravel()
        m = (v0.size + 9999) // 10000
        v0 = v0[::m]
        r0 = self.model.peakval.r.ravel()[::m]
        circle = go.Figure([cells], self.image_layout)
        allpeak = go.Scatter(
            x=(1 + 0.3 * dr * np.random.randn(r0.size)) * r0,
            y=v0,
            mode="markers",
            marker=dict(opacity=0.1),
            showlegend=False,
        )
        select = go.Scatter(x=peaks.r, y=peaks.v, mode="markers", marker=dict(opacity=0.5), showlegend=False)
        scatter = go.Figure([allpeak, select], self.layout)
        scatter.update_layout(
            xaxis=dict(title="radius", type="log"),
            yaxis=dict(title="intensity", rangemode="tozero"),
        )
        print("peak done")
        return circle, scatter
