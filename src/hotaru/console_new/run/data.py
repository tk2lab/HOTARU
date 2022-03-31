import click

from hotaru.image.load import load_data
from hotaru.image.mask import get_mask
from hotaru.image.mask import get_mask_range
from hotaru.image.stats import calc_stats
from hotaru.util.dataset import normalized
from hotaru.util.dataset import masked
from hotaru.util.tfrecord import save_tfrecord

from .base import run_base


@click.command()
@click.option(
    '--imgs-path',
    type=click.Path(exists=True, dir_okay=False, readable=True),
    default='imgs.tif',
    show_default=True,
)
@click.option(
    '--mask-type',
    default='0.pad',
    show_default=True,
)
@click.option(
    '--batch',
    type=int,
    default=100,
    show_default=True,
)
@run_base
def data(obj, imgs_path, mask_type, batch):
    '''Data'''

    imgs = load_data(imgs_path)
    nt, h, w = imgs.shape()

    mask = get_mask(mask_type, h, w)
    y0, y1, x0, x1 = get_mask_range(mask)

    data = imgs.clipped_dataset(y0, y1, x0, x1)
    mask = mask[y0:y1, x0:x1]

    stats = calc_stats(data.batch(batch), mask, nt, obj.verbose)
    smin, smax, sstd, avgt, avgx = stats

    normalized_data = normalized(data, sstd, avgt, avgx)
    masked_data = masked(normalized_data, mask)
    out_path = obj.out_path('data')
    save_tfrecord(f'{out_path}.tfrecord', masked_data, nt, obj.verbose)

    return dict(
        nt=nt, y0=y0, x0=x0, mask=mask,
        smin=smin, smax=smax, sstd=sstd, avgt=avgt, avgx=avgx,
    )
