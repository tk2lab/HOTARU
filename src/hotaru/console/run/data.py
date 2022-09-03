import click

from ...filter.stats import calc_stats
from ...io.image import load_data
from ...io.mask import get_mask
from ...io.mask import get_mask_range
from ...io.tfrecord import save_tfrecord
from ...util.dataset import masked
from ...util.dataset import normalized
from ..base import run_command


@run_command(
    click.Option(
        ["--imgs-path"],
        type=click.Path(exists=True, dir_okay=False, readable=True),
    ),
    click.Option(
        ["--mask-type"],
    ),
    click.Option(
        ["--hz"],
        type=float,
    ),
    click.Option(["--batch"], type=int),
)
def data(obj):
    """Data"""

    imgs = load_data(obj.imgs_path)
    nt, h, w = imgs.shape()

    mask = get_mask(obj.mask_type, h, w)
    y0, y1, x0, x1 = get_mask_range(mask)

    data = imgs.clipped_dataset(y0, y1, x0, x1)
    mask = mask[y0:y1, x0:x1]

    with obj.strategy.scope():
        with click.progressbar(length=nt, label="calc stats") as prog:
            stats = calc_stats(data.batch(obj.batch), mask, prog)
    smin, smax, sstd, avgt, avgx = stats

    normalized_data = normalized(data, sstd, avgt, avgx)
    masked_data = masked(normalized_data, mask)
    out_path = obj.out_path("data", obj.data_tag, "")
    masked_data = click.progressbar(masked_data, length=nt, label="Save")
    with masked_data:
        save_tfrecord(f"{out_path}.tfrecord", masked_data)

    return dict(
        nt=nt,
        y0=y0,
        x0=x0,
        mask=mask,
        smin=smin,
        smax=smax,
        sstd=sstd,
        avgt=avgt,
        avgx=avgx,
    )
