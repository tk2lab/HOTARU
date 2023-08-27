import numpy as np
import plotly.graph_objects as go
import plotly.io as pio

pio.kaleido.scope.mathjax = None


def plot_data(paths, imgs, labels, dlabels, width, margin, pad):

    num = len(imgs)

    hs, ws = zip(*(np.load(f"{path}/{imgs[0]}.npy").shape for path in paths))
    print(hs, ws)
    aspects = np.array(hs) / np.array(ws)

    fig = go.Figure().set_subplots(
        len(paths),
        len(imgs),
        column_titles=labels,
        row_titles=dlabels,
        row_heights=list(aspects),
        horizontal_spacing=0.01,
        vertical_spacing=0.03,
    )
    for i, path in enumerate(paths):
        for j, img in enumerate(imgs):
            img = np.load(f"{path}/{img}.npy")
            h, w = img.shape
            print(h, w)
            smin = img.min()
            smax = img.max()
            img = (img - smin) / (smax - smin)
            fig.add_trace(
                go.Heatmap(z=img, colorscale="Greens", showscale=False),
                row=i + 1,
                col=j + 1,
            )
            fig.update_xaxes(
                autorange=False,
                range=(0, w),
                showticklabels=False,
                row=i + 1,
                col=j + 1,
            )
            fig.update_yaxes(
                # scaleanchor="x",
                autorange=False,
                range=(h, 0),
                showticklabels=False,
                row=i + 1,
                col=j + 1,
            )
    mod_width = width - margin["l"] - margin["r"]
    fig.update_layout(
        annotations=[dict(font=dict(size=11)) for _ in range(num)],
        width=width,
        height=np.ceil(margin["t"] + margin["b"] + mod_width * aspects.sum() / num),
        margin=margin,
    )
    return fig
