import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio

from myload import load

pio.kaleido.scope.mathjax = None


data, init, clip = "Kd32", "min31", "clip2x2"
#data, init, clip = "CA3small", "min27-16", "clip3-100"
out = []
for la in [0.00, 0.05, 0.10, 0.15]:
#    for la in [0, 20, 40, 60, 80, 100]:
    for lu in [0.0, 0.005, 0.010, 0.015]:
        path = f"outputs/{data}/miniature/{init}/{clip}/su{lu}/sa{la}/pos0"
        for stage in range(30):
            try:
                df = load(path, stage, only_df=True)
                out.append((lu, la, stage, np.count_nonzero(df.kind == "cell")))
            except FileNotFoundError:
                pass
lu, la, stage, num = zip(*out)
df = pd.DataFrame(dict(lu=lu, la=la, stage=stage, num=num))
px.line(df, x="stage", y="num", color="la", symbol="lu").write_image(f"figs/{data}-num.pdf")
