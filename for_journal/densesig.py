import pandas as pd
import plotly.express as px
import plotly.io as pio

pio.kaleido.scope.mathjax = None


lus = [0, 20, 40, 100]
dfs = []
for lu in lus:
    path = f"CA3/miniature/min21/default/u{lu}"
    df = pd.read_csv(f"{path}/000stats.csv")
    df = df.query("kind != 'remove'")
    # fp = np.load(f"{path}/../../000footprint.npy")
    # df["intensity"] = df.signal * df.firmness
    df["lu"] = lu
    df["size"] = 1
    df["symbol"] = 1
    dfs.append(df)
df = pd.concat(dfs, axis=0)
fig = px.scatter(
    df,
    x="udense",
    y="signal",
    color=df.signal < 1,
    size="size",
    size_max=3,
    opacity=0.5,
    facet_col="lu",
)
for i in range(8):
    fig.data[i].marker.line.width = 0

for lu in lus:
    fig.update_annotations(selector=dict(text=f"lu={lu}"), y=0.8, text=f"λU={lu}")
fig.update_layout(
    template="none",
    font_size=11,
    showlegend=False,
    width=600,
    height=150,
    margin=dict(l=35, r=10, t=20, b=32),
)
fig.show()
fig.write_image("../figs/sig-dense.pdf")
