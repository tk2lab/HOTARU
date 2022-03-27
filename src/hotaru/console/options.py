import os

from cleo import option

from ..util.pickle import load_pickle


p = dict(
    data_tag='default',
    peak_tag='default',
    init_tag='default',
    tag='default',
    imgs_path='imgs.tif',
    mask_type='0.pad',
    radius_kind='log',
    radius_min=2.0,
    radius_max=24.0,
    radius_num=13,
    radius_elem=None,
    shard=1,
    distance=1.6,
    thr_area_abs=100.0,
    thr_area_rel=2.0,
    hz=20.0,
    tau_rise=0.08,
    tau_fall=0.16,
    tau_scale=6.0,
    lu=30.0,
    la=20.0,
    bt=0.1,
    bx=0.1,
    lr=0.01,
    tol=0.001,
    epoch=100,
    step=100,
    window=100,
    batch=100,
)
p = {k.replace('_', '-'): v for k, v in p.items()}

if os.path.exists('hotaru/config.pickle'):
    p.update(load_pickle('hotaru/config.pickle'))


short = dict(
    data_tag=None,
    peak_tag=None,
    init_tag=None,
    tag=None,
    imgs_path=None,
    mask_type=None,
    radius_kind=None,
    radius_min=None,
    radius_max=None,
    radius_num=None,
    radius_elem='r',
    shard=None,
    distance=None,
    thr_area_abs=None,
    thr_area_rel=None,
    hz=None,
    tau_rise=None,
    tau_fall=None,
    tau_scale=None,
    lu=None,
    la=None,
    bt=None,
    bx=None,
    lr=None,
    tol=None,
    epoch=None,
    step=None,
    window=None,
    batch=None,
)
short = {k.replace('_', '-'): v for k, v in short.items()}


desc = dict(
    data_tag='',
    peak_tag='',
    init_tag='',
    tag='',
    imgs_path='',
    mask_type='',
    radius_kind='',
    radius_min='',
    radius_max='',
    radius_num='',
    radius_elem='',
    shard='',
    distance='',
    thr_area_abs='',
    thr_area_rel='',
    hz='',
    tau_rise='',
    tau_fall='',
    tau_scale='',
    lu='',
    la='',
    bt='',
    bx='',
    lr='',
    tol='',
    epoch='',
    step='',
    window='',
    batch='',
)
desc = {k.replace('_', '-'): v for k, v in desc.items()}


option_type = dict(
    data_tag=str,
    peak_tag=str,
    init_tag=str,
    tag=str,
    imgs_path=str,
    mask_type=str,
    radius_kind=str,
    radius_min=float,
    radius_max=float,
    radius_num=int,
    radius_elem=lambda x: [float(v) for v in x],
    shard=int,
    distance=float,
    thr_area_abs=float,
    thr_area_rel=float,
    hz=float,
    tau_rise=float,
    tau_fall=float,
    tau_scale=float,
    lu=float,
    la=float,
    bt=float,
    bx=float,
    lr=float,
    tol=float,
    epoch=int,
    step=int,
    window=int,
    batch=int,
)
option_type = {k.replace('_', '-'): v for k, v in option_type.items()}


tag_options = {
    'spike-tag': option('spike-tag', 'P', '', False, False, False, p['tag']),
    'footprint-tag': option('footprint-tag', 'P', '', False, False, False, p['tag']),
    'start': option('start', None, '', False, False, False, 0),
    'goal': option('goal', None, '', False, False, False, 10),
}


options = {
    k: option(k, short[k], desc[k], False, p[k] is None, p[k] is None, p[k])
    for k in p.keys()
}


radius_options = [options[k] for k in [
    f'radius{p}' for p in ['-kind', '-min', '-max', '-num', '-elem']
]]


model_options = [options[k] for k in [
    'hz', 'tau-rise', 'tau-fall', 'tau-scale', 'lu', 'la', 'bt', 'bx',
]]


optimizer_options = [options[k] for k in [
    'lr', 'tol', 'epoch', 'step',
]]
