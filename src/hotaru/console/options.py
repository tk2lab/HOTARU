import os

from cleo import option

from ..util.pickle import load_pickle


p = dict(
    tag='default',
    imgs_path='imgs.tif',
    mask_type='0.pad',
    radius_kind='log',
    radius_min=2.0,
    radius_max=24.0,
    radius_num=13,
    radius=None,
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

if os.path.exists('hotaru/config.pickle'):
    p.update(load_pickle('hotaru/config.pickle'))


short = dict(
    tag=None,
    imgs_path=None,
    mask_type=None,
    radius_kind=None,
    radius_min=None,
    radius_max=None,
    radius_num=None,
    radius='r',
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


desc = dict(
    tag='',
    imgs_path='',
    mask_type='',
    radius_kind='',
    radius_min='',
    radius_max='',
    radius_num='',
    radius='',
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


tag_options = dict(
    data_tag=option('data-tag', 'D', '', False, False, False, p['tag']),
    peak_tag=option('peak-tag', 'P', '', False, False, False, p['tag']),
    spike_tag=option('spike-tag', 'P', '', False, False, False, p['tag']),
    footprint_tag=option('footprint-tag', 'P', '', False, False, False, p['tag']),
)


options = {
    k: option(k.replace('_', '-'), 
              short[k], desc[k], False, p[k] is None, p[k] is None, p[k])
    for k in p.keys()
}


radius_options = [options[k] for k in [
    f'radius{p}' for p in ['_kind', '_min', '_max', '_num', '']
]]


model_options = [options[k] for k in [
    'hz', 'tau_rise', 'tau_fall', 'tau_scale', 'lu', 'la', 'bt', 'bx',
]]


optimizer_options = [options[k] for k in [
    'lr', 'tol', 'epoch', 'step',
]]
