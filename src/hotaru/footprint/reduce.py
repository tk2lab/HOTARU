import multiprocessing as mp

import numpy as np
import click

def label_out_of_range(peaks, radius_min, radius_max):
    r = peaks['radius'].values
    peaks['accept'] = 'yes'
    peaks.loc[r == radius_min, 'accept'] = 'small_r'
    peaks.loc[r == radius_max, 'accept'] = 'large_r'
    return peaks


def reduce_peak_idx_mp(peaks, thr_distance, size, verbose=1):
    ind = peaks.index.values
    rs = peaks['radius'].values
    ys = peaks['y'].values
    xs = peaks['x'].values
    xmax = xs.max()
    ymax = ys.max()
    margin = int(np.ceil(thr_distance * rs.max()))

    data = []
    for x in range(0, xmax - margin, size):
        x0 = max(x - margin, 0)
        x1 = min(x + size + margin, xmax)
        for y in range(0, ymax - margin, size):
            y0 = max(y - margin, 0)
            y1 = min(y + size + margin, ymax)
            cond = (x0 <= xs) & (xs <= x1) & (y0 <= ys) & (ys <= y1)
            index = ind[cond]
            xtmp = xs[cond]
            ytmp = ys[cond]
            rtmp = rs[cond]
            data.append([x, y, index, ytmp, xtmp, rtmp, size, thr_distance])

    total = int(np.ceil((xmax - margin) / size) * np.ceil((ymax - margin) / size))
    with mp.Pool() as pool:
        with click.progressbar(length=total, label='Reduce') as prog:
            ind = []
            for x in pool.imap_unordered(_reduce_peak_idx_mp_local, data):
                ind.append(x)
                prog.update(1)
    return np.unique(np.concatenate(ind, 0))


def _reduce_peak_idx_mp_local(data):
    x, y, index, ytmp, xtmp, rtmp, size, thr_distance = data
    idx = _reduce_peak_idx(ytmp, xtmp, rtmp, thr_distance, verbose=0)
    xtmp = xtmp[idx]
    ytmp = ytmp[idx]
    cond = (x <= xtmp) & (xtmp < x + size) & (y <= ytmp) & (ytmp < y + size)
    if np.any(cond):
        return index[np.array(idx)[cond]]
    else:
        return np.array([], np.int32)


def _reduce_peak_idx(ys, xs, rs, thr_distance, verbose=1):
    flg = np.arange(rs.size, dtype=np.int32)
    idx = []
    while flg.size > 0:
        i, j = flg[0], flg[1:]
        y0, x0 = ys[i], xs[i]
        y1, x1 = ys[j], xs[j]
        ya, xa = ys[idx], xs[idx]
        thr = np.square(thr_distance * rs[i])
        if not idx or np.all(np.square(ya - y0) + np.square(xa - x0) >= thr):
            cond = np.square(y1 - y0) + np.square(x1 - x0) >= thr
            flg = j[cond]
            idx.append(i)
        else:
            flg = j
    return idx


def reduce_peak(peaks, radius, thr_distance, verbose=1):
    idx = reduce_peak_idx(peaks, radius, thr_distance, verbose)
    return tuple(v[idx] for v in peaks)


def reduce_peak_idx(peaks, radius, thr_distance, verbose=1):
    ts, rs, ys, xs = peaks[:, 0], peaks[:, 1], peaks[:, 2], peaks[:, 3]
    rs = np.array(radius)[rs]
    idx = _reduce_peak_idx(ys, xs, rs, thr_distance, verbose)
    return np.array(idx, np.int32)
