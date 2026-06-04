import base64
import hashlib
import inspect
import json
from functools import wraps

import numpy as np
from keras import ops
from keras.saving import load_model
from keras.saving import save_model
from matplotlib.pyplot import get_cmap
from tqdm import trange

from .data.io import to_movie
from .random import Generator
from .spatial.simulation import sim_footprints
from .spatial.simulation import sim_neuropil_basis
from .temporal import DoubleExpKernel
from .temporal import ExpKernel
from .temporal.simulation import sim_dendrites
from .temporal.simulation import sim_neuropil_traces
from .temporal.simulation import sim_traces


def to_str(x):
    match x:
        case dict():
            return {k: to_str(v) for k, v in x.items()}
        case list() | tuple():
            return [to_str(v) for v in x.items()]
        case _:
            return str(x)


def auto_save_config(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        sig = inspect.signature(func)
        bound_args = sig.bind(*args, **kwargs)
        bound_args.apply_defaults()
        config = dict(bound_args.arguments)
        path = config.pop('path')
        force = config.pop('force', False)
        config_str = json.dumps(to_str(config), sort_keys=True)
        short_bytes = hashlib.sha256(config_str.encode('utf-8')).digest()[:6]
        config_hash = base64.urlsafe_b64encode(short_bytes).decode('utf-8').rstrip('=')
        path = path / config_hash
        print(path)
        config_path = path / 'config.json'
        if force or not config_path.exists():
            path.mkdir(exist_ok=True, parents=True)
            func(path, **config)
            config_path.write_text(config_str)
        return path

    return wrapper


def make_link(dst_path, dst, target):
    target = target.absolute().relative_to(dst_path.absolute(), walk_up=True)
    dst_path = dst_path / dst
    dst_path.unlink(missing_ok=True)
    dst_path.symlink_to(target, target_is_directory=True)


def make_imgs(fps, trs, path):
    fps_val = fps.segs.numpy()
    trs_val = trs.obs.numpy()
    _nk, h, w = fps_val.shape
    _nk, nt = trs_val.shape
    out = np.lib.format.open_memmap(path / 'imgs.npy', 'w+', 'float32', (nt, h, w))
    for t in trange(nt, ncols=150, desc='mix fps and trs'):
        out[t] = np.einsum('kyx,k->yx', fps_val, trs_val[:, t])
    return out


def make_movie(imgs, hz, path, vmin=None, vmax=None, cmap='Greens'):
    if vmin is None:
        vmin = imgs.min()
    if vmax is None:
        vmax = imgs.max()
    cmap = get_cmap(cmap)

    def img_iter():
        for o in imgs:
            v = (o - vmin) / (vmax - vmin)
            yield (255 * cmap(v)).astype('uint8')

    to_movie(path / 'imgs.mp4', img_iter(), imgs.shape, hz)


@auto_save_config
def make_cell(
    path,
    num_frames,
    height,
    width,
    hz,
    num,
    intensity_s,
    radius_m,
    radius_s,
    overwrap,
    noise,
    isi_mm,
    isi_ms,
    isi_sm,
    isi_ss,
    tau1,
    tau2,
    upsample_factor,
    seed,
):
    rng = Generator(seed)

    r_cell = [radius_m] * num
    s_cell = [radius_s] * num
    fps = sim_footprints(height, width, r_cell, r_cell, s_cell, noise, overwrap, rng)
    save_model(fps, path / 'fps.keras')

    kernel = DoubleExpKernel(tau1, tau2, hz=hz * upsample_factor)
    isi_mm = [isi_mm] * num
    isi_ms = [isi_ms] * num
    isi_sm = [isi_sm] * num
    isi_ss = [isi_ss] * num
    g_s = [intensity_s] * num
    trs = sim_traces(num_frames, kernel, isi_mm, isi_ms, isi_sm, isi_ss, g_s, rng)
    save_model(trs, path / 'trs.keras')

    imgs = make_imgs(fps, trs, path)
    make_movie(imgs, hz, path)

    return path


@auto_save_config
def make_dendrite(
    path,
    cell_path,
    num_frames,
    height,
    width,
    hz,
    num,
    radius_m1,
    radius_m2,
    radius_s,
    tau,
    beta,
    num_mix,
    noise,
    seed,
):
    _ = num_frames
    rng = Generator(seed)

    cell_fps = load_model(cell_path / 'fps.keras')
    cell_trs = load_model(cell_path / 'trs.keras')

    kernel = ExpKernel(tau, hz=20.0)
    r1_dend_n = [radius_m1] * num
    r2_dend_n = [radius_m2] * num
    s_dend_n = [radius_s] * num
    fps = sim_footprints(height, width, r1_dend_n, r2_dend_n, s_dend_n, noise, 2.0, rng)
    save_model(fps, path / 'fps.keras')

    yc, xc = cell_fps.peaks.T
    yd, xd = fps.peaks.T[:, :, None]
    dist_mat = ops.hypot(yc - yd, xc - xd)
    trs = sim_dendrites(
        dist_mat,
        cell_trs.obs.value,
        kernel.kernel(),
        beta,
        num_mix,
        rng=rng,
    )
    save_model(trs, path / 'trs.keras')

    imgs = make_imgs(fps, trs, path)
    make_movie(imgs, hz, path)


@auto_save_config
def make_neuropil(path, num_frames, height, width, hz, scale, n_basis, tau, seed):
    rng = Generator(seed)

    fps = sim_neuropil_basis(height, width, scale, n_basis)
    save_model(fps, path / 'neuropil_x.keras')

    trs = sim_neuropil_traces(fps.shape[0], num_frames, tau, rng, hz=hz)
    save_model(trs, path / 'neuropil_t.keras')

    imgs = make_imgs(fps, trs, path)

    vmax = np.max(np.abs(imgs))
    make_movie(imgs, hz, path, -vmax, vmax, 'bwr')


@auto_save_config
def make_sim(path, cell, dendrite, neuropil, hz, w_dendrite, w_neuropil, base, noise, seed):
    cell = np.load(cell / 'imgs.npy')
    dendrite = np.load(dendrite / 'imgs.npy')
    neuropil = np.load(neuropil / 'imgs.npy')
    nt, h, w = cell.shape

    imgs = np.lib.format.open_memmap(path / 'imgs.npy', 'w+', 'float32', cell.shape)
    for t in trange(nt, ncols=150, desc='mix'):
        imgs[t] = cell[t] + w_dendrite * dendrite[t] + w_neuropil * neuropil[t]

    rng = Generator(seed)
    vmin = imgs.min() - base
    for t in trange(nt, ncols=150, desc='add noise'):
        v = imgs[t] - vmin
        v += ops.convert_to_numpy(rng.normal(shape=(h, w))) * np.sqrt(noise * v)
        imgs[t] = v

    make_movie(imgs, hz, path)


def greedy_matching_r2(r2_mat):
    """
    r2_mat: shape (K, L) の行列
            K: Result (検出数, e.g., 717)
            L: Ground Truth (真の細胞数)
    """
    num_results, num_gts = r2_mat.shape
    max_possible_pairs = min(num_results, num_gts)

    matched_pairs = []
    matched_r2_values = []

    used_results = set()
    used_gts = set()

    flat_indices = np.argsort(-r2_mat.ravel())
    for flat_idx in flat_indices:
        i, j = flat_idx // num_gts, flat_idx % num_gts
        if (i in used_results) or (j in used_gts):
            continue

        used_results.add(i)
        used_gts.add(j)
        matched_pairs.append((i, j))
        matched_r2_values.append(r2_mat[i, j])

        if len(matched_pairs) == max_possible_pairs:
            break

    matched_r2_values = np.array(matched_r2_values)

    print('--- 厳密評価 (Greedy Matching) ---')
    print(f'検出数 (K): {num_results}, 真の細胞数 (L): {num_gts}')
    print(f'成立したペア数: {len(matched_pairs)}')
    print(f'検出全体の平均 R^2: {matched_r2_values.sum() / num_results:.4f}')

    return matched_pairs, matched_r2_values
