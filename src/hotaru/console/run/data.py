import click

from ...filter.stats import calc_stats
from ...io.image import load_data
from ...io.mask import get_mask
from ...io.mask import get_mask_range
from ...util.dataset import masked
from ...util.dataset import normalized
from ...util.progress import Progress
from ..base import command_wrap
from ..base import configure
from ..base import readable_file


@click.command(context_settings=dict(show_default=True))
@click.option("--tag", type=str, callback=configure, is_eager=True)
@click.option("--imgs-path", type=readable_file)
@click.option("--hz", type=float)
@click.option("--mask-type", type=str)
@click.option("--tausize", type=int)
@click.option("--batch", type=int)
@click.pass_obj
@command_wrap
def data(obj, tag, imgs_path, hz, mask_type, tausize, batch):
    """Prepare Data."""

    imgs = load_data(imgs_path)
    nt, h, w = imgs.shape()

    mask = get_mask(mask_type, h, w)
    y0, y1, x0, x1 = get_mask_range(mask)

    data = imgs.clipped_dataset(y0, y1, x0, x1)
    mask = mask[y0:y1, x0:x1]
    nx = mask.sum()

    with Progress(length=nt, label="Stats", unit="frame") as prog:
        with obj.strategy.scope():
            stats = calc_stats(data.batch(batch), mask, prog)
    smin, smax, sstd, avgt, avgx = stats

    normalized_data = normalized(data, sstd, avgt, avgx)
    masked_data = masked(normalized_data, mask)
    with Progress(masked_data, length=nt, label="Save", unit="frame") as prog:
        obj.save_tfrecord(prog, tag, 0, "1data", "data")
    obj.save_numpy(mask, tag, 0, "1data", "mask")
    obj.save_numpy(avgx, tag, 0, "1data", "avgx")
    obj.save_numpy(avgt, tag, 0, "1data", "avgt")

    log = dict(
        nt=int(nt),
        nx=int(nx),
        y0=int(y0),
        x0=int(x0),
        smin=float(smin),
        smax=float(smax),
        sstd=float(sstd),
    )
    return log, tag, 0, "1data"
