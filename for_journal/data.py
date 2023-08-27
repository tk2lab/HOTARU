from hotaru.plot.data import plot_data
from hotaru.plot.stats import plot_stats


paths = [
    "outputs/Kd32",
    "outputs/Sato1",
    "outputs/CA3",
]
dlabels = [
    "Data 1", "Data 2", "Data 3",
]
imgs = [
    "max", "std", "cor",
]
labels = [
    "Max", "Std", "Cor",
]

args = dict(
    width = 600,
margin = dict(l=10, r=25, t=25, b=10),
pad=10,
)


"""
plot_data(paths, imgs, labels, dlabels, **args).write_image(
    "figs/data.pdf"
)
"""


plot_stats(paths, imgs, labels, dlabels, width=600, height=200).write_image(
    "figs/stats.pdf"
)
