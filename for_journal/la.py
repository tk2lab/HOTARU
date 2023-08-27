import pandas as pd

from hotaru.plot.compla import compare_la_fig


paths = [
    "outputs/CA3/miniature/min21/default/u100/a0/dup07-30-8/001stats.csv",
    "outputs/CA3/miniature/min21/default/u100/default/dup07-30-8/001stats.csv",
]
compare_la_fig(paths, ["a0", "a20"]).write_image(
    "figs/la.pdf",
)
