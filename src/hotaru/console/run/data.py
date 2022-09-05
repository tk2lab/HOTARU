import click

from ...filter.stats import calc_stats
from ...io.image import load_data
from ...io.mask import get_mask
from ...io.mask import get_mask_range
from ...util.dataset import masked
from ...util.dataset import normalized
from ..base import command_wrap
from ..base import configure

readable_file = click.Path(exists=True, dir_okay=False, readable=True)


@click.command(context_settings=dict(show_default=True))
@click.option("--tag", type=str, callback=configure, is_eager=True)
@click.option("--imgs-path", type=readable_file)
@click.option("--mask-type", type=str)
@click.option("--hz", type=float)
@click.option("--batch", type=int)
@click.pass_obj
@command_wrap
def data(obj, tag, imgs_path, mask_type, hz, batch):
    """Data"""

    imgs = load_data(imgs_path)
    nt, h, w = imgs.shape()

    mask = get_mask(mask_type, h, w)
    y0, y1, x0, x1 = get_mask_range(mask)

    data = imgs.clipped_dataset(y0, y1, x0, x1)
    mask = mask[y0:y1, x0:x1]
    nx = mask.sum()

    with obj.strategy.scope():
        with click.progressbar(length=nt, label="Stats") as prog:
            stats = calc_stats(data.batch(batch), mask, prog)
    smin, smax, sstd, avgt, avgx = stats

    normalized_data = normalized(data, sstd, avgt, avgx)
    masked_data = masked(normalized_data, mask)
    masked_data = click.progressbar(masked_data, length=nt, label="Save")
    with masked_data:
        obj.save_tfrecord(masked_data, "data", tag, "_data")
    obj.save_numpy(mask, "data", tag, "_mask")
    obj.save_numpy(avgx, "data", tag, "_avgx")
    obj.save_numpy(avgt, "data", tag, "_avgt")

    log = dict(
        nt=int(nt),
        nx=int(nx),
        y0=int(y0),
        x0=int(x0),
        smin=float(smin),
        smax=float(smax),
        sstd=float(sstd),
    )
    return log, "1data", tag, 0
