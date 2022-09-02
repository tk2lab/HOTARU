import click
import matplotlib.pyplot as plt

from ..filter.cor import calc_cor
from ..filter.max import calc_max
from ..filter.std import calc_std
from ..io.image import load_data
from .base import run_command


@run_command(
    click.Option(
        ["--imgs-path"],
        type=click.Path(exists=True, dir_okay=False, readable=True),
    ),
)
def stats(obj):
    """Stats"""

    data = load_data(obj.imgs_path)
    t, h, w = data.shape()
    data = data.clipped_dataset(0, h, 0, w).batch(100)

    fw = 1.5
    fh = fw * h / w
    path = obj.out_path("figure", obj.tag, "_stats")

    fig = plt.figure(figsize=(1, 1))
    for i, (l, f) in enumerate(zip("ABC", [calc_max, calc_std, calc_cor])):
        with click.progressbar(length=t, label=f.__name__) as prog:
            image = f(data, prog=prog)
        fig.text(0.0 + i * (0.2 + fw), -0.15, l, fontsize=12)
        ax = fig.add_axes([0.15 + i * (0.2 + fw), -fh, fw, fh])
        ax.imshow(image, cmap="Greens")
        ax.axis("off")
    fig.savefig(f"{path}.pdf", bbox_inches="tight", pad_inches=0.1)
