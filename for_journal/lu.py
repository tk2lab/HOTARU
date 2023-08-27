import pandas as pd

from hotaru.plot.complu import compare_lu_fig


paths = [
    "outputs/CA3/miniature/min21/default/u0/000stats.csv",
    "outputs/CA3/miniature/min21/default/u100/000stats.csv",
    "outputs/CA3/miniature/min21/default/u1000/000stats.csv",
]
compare_lu_fig(paths, ["u0", "u100", "u1000"]).write_image(
    "figs/lu.pdf",
)
