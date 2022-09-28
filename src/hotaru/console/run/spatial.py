import click
import numpy as np
import tensorflow as tf
import pandas as pd

from ...footprint.evaluate import normalize_footprint
from ...footprint.evaluate import evaluate_footprint
from ...footprint.evaluate import clean_footprint
from ...evaluate.utils import calc_area
from ...evaluate.utils import calc_overwrap
from ..base import command_wrap
from ..base import configure
from ..base import penalty_options
from ..base import optimizer_options
from ..base import early_stop_options
from ..base import radius_options
from ..progress import Progress


@click.command(context_settings=dict(show_default=True))
@click.option("--tag", type=str, callback=configure, is_eager=True)
@click.option("--temporal-tag", type=str)
@click.option("--temporal-stage", type=int)
@click.option("--threshold-denseness", type=float)
@penalty_options
@optimizer_options
@early_stop_options
@click.option("--batch", type=int)
@click.option("--epochs", type=int)
@radius_options
@click.option("--threshold-region", type=float)
@click.option("--storage-saving", is_flag=True)
@click.pass_obj
@command_wrap
def spatial(obj, tag, temporal_tag, temporal_stage, threshold_denseness, penalty, optimizer, early_stop, batch, epochs, radius, threshold_region, storage_saving):
    """Update Footprint from Spike."""

    if temporal_tag is None:
        temporal_tag = tag

    if temporal_stage is None:
        temporal_stage = 1

    if temporal_tag != tag:
        stage = 1
    else:
        if temporal_stage == 999:
            stage = 999
        else:
            stage = temporal_stage + 1

    if storage_saving:
        stage = 999

    prev = dict(tag=temporal_tag, stage=temporal_stage, kind="2temporal")
    curr = dict(tag=tag, stage=stage, kind="1spatial")

    data_tag = obj.data_tag(**prev)
    data = obj.data(data_tag)
    mask = obj.mask(data_tag)
    nt = obj.nt(data_tag)
    dynamics = obj.used_dynamics(**prev)

    spike = obj.spike(**prev)
    localt = obj.localt(**prev)

    info = obj.info(**prev)
    info.query("(kind == 'cell') or (kind == 'local')", inplace=True)
    info["old_kind"] = info.kind
    info["old_id"] = info.id
    cell = info.query("kind == 'cell'")
    local = info.query("kind == 'local'")
    old_nk = cell.shape[0]
    old_nl = local.shape[0]

    model = obj.model(data_tag, info.shape[0])
    model.set_double_exp(**dynamics)
    model.set_penalty(**penalty)
    model.spatial.optimizer.set(**optimizer)
    model.set_early_stop(**early_stop)

    cell_to_local = cell.denseness > threshold_denseness
    localt_new = model.spike_to_calcium(spike[cell_to_local]).numpy()
    localt = np.concatenate([localt, localt_new], axis=0)
    spike = spike[~cell_to_local]
    info.loc[cell.index[cell_to_local], "kind"] = "local"
    cell = info.query("kind == 'cell'")
    local = info.query("kind == 'local'")

    nk = cell.shape[0]
    nl = local.shape[0]
    click.echo(f"high density spike: {np.count_nonzero(cell_to_local)}")
    click.echo(f"num: {old_nk}, {old_nl} -> {nk}, {nl}")

    spike /= spike.max(axis=1, keepdims=True)
    localt /= localt.std(axis=1, keepdims=True)

    with Progress(length=nt, label="Spatial Init", unit="frame") as prog:
        model.spike.set_val(spike)
        model.localt.set_val(localt)
        model.prepare_spatial(batch, prog=prog)

    summary_dir = obj.summary_path(**curr)
    cb = obj.callbacks("Spatial Train", summary_dir)
    model_log = model.fit_spatial(callbacks=cb, epochs=epochs, verbose=0)

    footprint = model.footprint.get_val()
    localx = model.localx.get_val()

    info.loc[cell.index, "area"] = calc_area(footprint, threshold_region)
    info.loc[local.index, "area"] = calc_area(localx, threshold_region)

    no_seg = normalize_footprint(footprint)
    nkr = np.count_nonzero(no_seg)
    footprint = footprint[~no_seg]
    spike = spike[~no_seg]
    clean_footprint(footprint, mask)
    info.loc[cell.index[no_seg], "kind"] = "no_seg"
    cell = info.query("kind == 'cell'")

    no_seg = normalize_footprint(localx)
    nlr = np.count_nonzero(no_seg)
    localx = localx[~no_seg]
    localt = localt[~no_seg]
    clean_footprint(localx, mask)
    info.loc[local.index[no_seg], "kind"] = "no_seg"
    local = info.query("kind == 'local'")

    click.echo(f"no seg: {nkr}, {nlr}")
    click.echo(f"num: {nk}, {nl} -> {nk - nkr}, {nl - nlr}")
    nk = cell.shape[0]
    nl = local.shape[0]

    info["firmness"] = np.nan
    info["radius"] = np.nan
    with Progress(length=nk, label="Eval Cell", unit="footprint") as prog:
        out = evaluate_footprint(footprint, mask, radius, batch, prog)
        for k, v in out.items():
            info.loc[cell.index, k] = v
    with Progress(length=nl, label="Eval Local", unit="footprint") as prog:
        out = evaluate_footprint(localx, mask, radius, batch, prog)
        for k, v in out.items():
            info.loc[local.index, k] = v

    small_r = info.loc[cell.index, "radius"] == radius[0]
    nkr = np.count_nonzero(small_r)
    footprint = footprint[~small_r]
    spike = spike[~small_r]
    info.loc[cell.index[small_r], "kind"] = "small_r"
    cell = info.query("kind == 'cell'")

    small_r = info.loc[local.index, "radius"] == radius[0]
    nlr = np.count_nonzero(small_r)
    localx = localx[~small_r]
    localt = localt[~small_r]
    info.loc[local.index[small_r], "kind"] = "small_r"
    local = info.query("kind == 'local'")

    click.echo(f"small radius: {nkr}, {nlr}")
    click.echo(f"num: {nk}, {nl} -> {nk - nkr}, {nl - nlr}")
    nk = cell.shape[0]
    nl = local.shape[0]

    info["scale"] = np.nan
    scale = footprint.max(axis=1, keepdims=True)
    info.loc[cell.index, "scale"] = scale
    footprint /= scale
    spike *= scale
    scale = localx.max(axis=1, keepdims=True)
    info.loc[local.index, "scale"] = scale
    localx /= scale
    localt *= scale

    binary = footprint > threshold_region
    info.loc[cell.index, "overwrap"] = calc_overwrap(binary)
    info.loc[local.index, "overwrap"] = np.nan

    info["id"] = -1
    info.loc[cell.index, "id"] = np.arange(cell.shape[0])
    info.loc[local.index, "id"] = np.arange(local.shape[0])

    obj.save_csv(info, **curr, name="info")
    obj.save_numpy(footprint, **curr, name="footprint")
    obj.save_numpy(localx, **curr, name="localx")
    obj.save_numpy(spike, **curr, name="spike") 
    obj.save_numpy(localt, **curr, name="localt")

    log = dict(
        data_tag=data_tag,
        temporal_tag=temporal_tag,
        temporal_stage=temporal_stage,
        history=model_log.history,
    )
    return log, tag, stage, "1spatial"
