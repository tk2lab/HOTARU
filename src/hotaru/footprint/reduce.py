import multiprocessing as mp

import numpy as np

from ..util.progress import Progress


def reduce_peak(info, thr_distance, block_size):
    ind = info.index.to_numpy()
    rs = info.radius.to_numpy()
    ys = info.y.to_numpy()
    xs = info.x.to_numpy()
    xmax = xs.max()
    ymax = ys.max()
    margin = int(np.ceil(thr_distance * rs.max()))

    data = []
    for x in range(0, xmax - margin, block_size):
        x0 = max(x - margin, 0)
        x1 = min(x + block_size + margin, xmax)
        for y in range(0, ymax - margin, block_size):
            y0 = max(y - margin, 0)
            y1 = min(y + block_size + margin, ymax)
            cond = (x0 <= xs) & (xs <= x1) & (y0 <= ys) & (ys <= y1)
            index = ind[cond]
            xtmp = xs[cond]
            ytmp = ys[cond]
            rtmp = rs[cond]
            data.append([x, y, index, ytmp, xtmp, rtmp, thr_distance, block_size])

    #data = Progress(data, "reduce peak", unit="block")
    with mp.Pool() as pool:
        imap = pool.imap_unordered(_reduce_peak_idx_local, data)
        ind = [x for x in imap]
    index = np.sort(np.unique(np.concatenate(ind, axis=0)))
    return info.loc[index].copy()


def _reduce_peak_idx_local(data):
    x, y, index, ytmp, xtmp, rtmp, thr_distance, block_size = data
    idx = _reduce_peak_idx(ytmp, xtmp, rtmp, thr_distance)
    xtmp = xtmp[idx]
    ytmp = ytmp[idx]
    cond = (x <= xtmp) & (xtmp < x + block_size) & (y <= ytmp) & (ytmp < y + block_size)
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
