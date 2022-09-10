import multiprocessing as mp

import numpy as np


def label_out_of_range(peaks, radius_min, radius_max):
    r = peaks["radius"].values

    peaks["accept"] = "yes"
    peaks.loc[(r == radius_min) | (r == radius_max), "accept"] = "no"

    peaks["reason"] = "-"
    peaks.loc[r == radius_min, "reason"] = "small_r"
    peaks.loc[r == radius_max, "reason"] = "large_r"
    return peaks


def reduce_peak_mask(peaks, thr_distance):
    rs = peaks["radius"].values
    ys = peaks["y"].values
    xs = peaks["x"].values
    idx = _reduce_peak_idx(ys, xs, rs, thr_distance)
    out = np.zeros_like(rs)
    out[idx] = True
    return out


def reduce_peak_idx_data(peaks, thr_distance, size):
    ind = peaks.index.values
    rs = peaks["radius"].values
    ys = peaks["y"].values
    xs = peaks["x"].values
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
    return data


def reduce_peak_idx_finish(data):
    with mp.Pool() as pool:
        imap = pool.imap_unordered(_reduce_peak_idx_local, data)
        ind = [x for x in imap]
    return np.unique(np.concatenate(ind, 0))


def _reduce_peak_idx_local(data):
    x, y, index, ytmp, xtmp, rtmp, size, thr_distance = data
    idx = _reduce_peak_idx(ytmp, xtmp, rtmp, thr_distance)
    xtmp = xtmp[idx]
    ytmp = ytmp[idx]
    cond = (x <= xtmp) & (xtmp < x + size) & (y <= ytmp) & (ytmp < y + size)
    if np.any(cond):
        return index[np.array(idx)[cond]]
    else:
        return np.array([], np.int32)


def _reduce_peak_idx(ys, xs, rs, thr_distance):
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
