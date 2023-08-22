import numpy as np
import jax.numpy as jnp
import plotly.graph_objects as go
import plotly.io as pio

from hotaru.filter import (
    gaussian_laplace,
    max_pool,
)
from hotaru.footprint import get_radius

pio.kaleido.scope.mathjax = None


def plot_gl(data, radius, idx, width):
    radius = get_radius(radius)
    imgs = data.select(idx)

    glrt = gaussian_laplace(imgs, radius, axis=1)
    print(glrt.shape)
    gl_max = max_pool(glrt, (3, 3, 3), (1, 1, 1), "same")
    gl_peak = glrt == gl_max

    glrt = np.array(glrt)
    gl_peak = np.array(gl_peak)
    radius = np.array(radius)

    glrt = glrt[:, 1:-1]
    gl_max = gl_max[:, 1:-1]
    gl_peak = gl_peak[:, 1:-1]
    radius = radius[1:-1]
    t, r, y, x = np.where(gl_peak & (glrt > 0))
    g = glrt[t, r, y, x]
    g /= g.max()
    r = radius[r]

    glrt = glrt[:, ::3]
    radius = radius[::3]
    nt, nr, h, w = glrt.shape
    print(nt, nr, h, w)
    print(np.stack([t, y, x, r, g], axis=1))
    print(y.size)
    #for out in zip(t, y, x, r, g):
    #    print(*out)

    fig = go.Figure().set_subplots(
        nt,
        nr + 2,
        column_widths=[w] * (nr + 2),
        row_heights=[h] * nt,
        column_titles=["image"] + [f"r={r:.1f}" for r in radius] + ["peaks"],
        row_titles=[f"t={t}" for t in idx],
        vertical_spacing=0.01,
        horizontal_spacing=0.01,
        print_grid=True,
        shared_xaxes=True,
        shared_yaxes=True,
    )
    margin = dict(l=15, r=30, t=30, b=15)
    mod_width = width - margin["l"] - margin["r"]
    for j, (img, glr) in enumerate(zip(imgs, glrt)):
        fig.add_trace(
            go.Heatmap(z=img, colorscale="Greens", showscale=False, zmin=-1, zmax=1),
            row=j + 1,
            col=1,
        )
        cond = (t == j)
        yj = y[cond]
        xj = x[cond]
        rj = r[cond]
        gj = g[cond]
        marker = dict(
            symbol="circle",
            color="blue",
            opacity=0.8 * (gj + 0.3) / 1.3,
            line_width=0,
            size=5 * rj * mod_width / (nr + 2) / w,
            sizemode="diameter",
        )
        fig.add_trace(
            go.Scatter(mode="markers", x=xj, y=yj, marker=marker),
            row=j + 1,
            col=nr + 2,
        )
        for i, gl in enumerate(glr):
            fig.add_trace(
                go.Heatmap(z=-gl, colorscale="Picnic", showscale=False, zmin=-1, zmax=1),
                row=j + 1,
                col=i + 2,
            )
    fig.update_xaxes(
        autorange=False,
        range=(0, w),
        showgrid=False,
        showticklabels=False,
    )
    fig.update_yaxes(
        autorange=False,
        range=(h, 0),
        showgrid=False,
        showticklabels=False,
    )
    fig.update_layout(
        showlegend=False,
        annotations=[
            dict(font=dict(size=11)) for _ in range(len(fig.layout.annotations))
        ],
        width=width,
        height=np.ceil(margin["t"] + margin["b"] + mod_width * nt * h / (nr + 2) / w),
        margin=margin,
    )
    return fig
