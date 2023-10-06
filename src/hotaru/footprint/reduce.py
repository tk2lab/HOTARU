import multiprocessing as mp
from logging import getLogger

import numpy as np
import pandas as pd

logger = getLogger(__name__)


def reduce_peaks_simple(
    ys, xs, rs, vs, min_radius, max_radius, min_distance_ratio, old_bg=None
):
    if old_bg is None:
        old_bg = []

    n = np.count_nonzero(np.isfinite(vs))
    flg = np.argsort(vs)[::-1][:n]
    cell = []
    bg = []
    remove = []
    while flg.size > 0:
        i, flg = flg[0], flg[1:]
        r0 = rs[i]
        if r0 >= min_radius:
            y0, x0 = ys[i], xs[i]
            if i in old_bg or r0 > max_radius:
                yb, xb = ys[bg], xs[bg]
                distance = np.hypot(xb - x0, yb - y0) / r0
                if not bg or np.all(distance >= min_distance_ratio):
                    bg.append(i)
                    logger.debug(
                        "background: id=%d old=%d r=%f dist=%s",
                        i,
                        i in old_bg,
                        r0,
                        sorted(distance)[:2],
                    )
                else:
                    remove.append(i)
                    logger.debug(
                        "removed (dup bg): id=%d old=%d r=%f dist=%s",
                        i,
                        i in old_bg,
                        r0,
                        sorted(distance)[:2],
                    )
            else:
                yc, xc = ys[cell], xs[cell]
                distance = np.hypot(xc - x0, yc - y0) / r0
                if not cell or np.all(distance >= min_distance_ratio):
                    y1, x1 = ys[flg], xs[flg]
                    dist2 = np.hypot(x1 - x0, y1 - y0) / r0
                    cond = dist2 > min_distance_ratio
                    if np.any(~cond):
                        remove += list(flg[~cond])
                        logger.debug(
                            "pre remove: %s dist=%s", list(flg[~cond]), dist2[~cond]
                        )
                    flg = flg[cond]
                    cell.append(i)
                    logger.debug("cell: id=%d dist%s", i, sorted(distance)[:2])
                else:
                    remove.append(i)
                    logger.debug(
                        "removed (dup cell): id=%d r=%f dist=%s",
                        i,
                        r0,
                        sorted(distance)[:2],
                    )
        else:
            remove.append(i)
            logger.debug("removed (small r): id=%d r=%f", i, r0)
    return cell, bg, remove


def reduce_peaks_mesh(rs, vs, *args, **kwargs):
    h, w = rs.shape
    ys, xs = np.mgrid[:h, :w]
    ys, xs, rs, vs = ys.ravel(), xs.ravel(), rs.ravel(), vs.ravel()
    cell, bg, remove = reduce_peaks_simple(ys, xs, rs, vs, *args, **kwargs)
    return ys[cell], xs[cell], ys[bg], xs[bg]


def reduce_peaks(
    peakval,
    bg_type="bg",
    min_radius=None,
    max_radius=None,
    min_distance_ratio=None,
    block_size=None,
):
    logger.info(
        "reduce_peaks: %f %f %f %d",
        min_radius,
        max_radius,
        min_distance_ratio,
        block_size,
    )
    logger.info("bg_type %s", bg_type)
    static_args = min_radius, max_radius, min_distance_ratio

    radius, ts, ri, vs = peakval
    rs = radius[ri]
    h, w = rs.shape
    margin = int(np.ceil(min_distance_ratio * rs.max()))

    args = []
    for xs in range(0, w - margin, block_size):
        x0 = max(xs - margin, 0)
        xe = xs + block_size
        x1 = min(xe + margin, w)
        for ys in range(0, h - margin, block_size):
            y0 = max(ys - margin, 0)
            ye = ys + block_size
            y1 = min(ye + margin, h)
            r = rs[y0:y1, x0:x1]
            v = vs[y0:y1, x0:x1]
            args.append(((y0, x0, ys, xs, ye, xe), (r, v, *static_args)))

    out = []
    with mp.Pool() as pool:
        for o in pool.imap_unordered(_reduce_peaks, args):
            out.append(o)
    celly, cellx, bgy, bgx = [np.concatenate(v, axis=0) for v in zip(*out)]

    def make_dataframe(y, x):
        df = pd.DataFrame(
            dict(
                y=y,
                x=x,
                t=ts[y, x],
                radius=rs[y, x],
                firmness=vs[y, x],
            )
        )
        return df.sort_values("firmness", ascending=False)

    cell = make_dataframe(celly, cellx)
    cell["kind"] = "cell"
    if bg_type == "remove":
        peaks = cell
    elif bg_type == "bg":
        bg = make_dataframe(bgy, bgx)
        bg["kind"] = "background"
        peaks = pd.concat([cell, bg], axis=0)
    else:
        raise ValueError()
    peaks = peaks.reset_index(drop=True)
    peaks.insert(0, "segid", peaks.index)
    nk = np.count_nonzero(peaks.kind == "cell")
    nb = np.count_nonzero(peaks.kind == "background")
    logger.info("num cell/bg: %d/%d", nk, nb)
    return peaks[["segid", "kind", "y", "x", "t", "radius", "firmness"]]


def _reduce_peaks(args):
    def fix(y, x):
        y += y0
        x += x0
        cond = (ys <= y) & (y < ye) & (xs <= x) & (x < xe)
        return y[cond], x[cond]

    y0, x0, ys, xs, ye, xe = args[0]
    celly, cellx, bgy, bgx = reduce_peaks_mesh(*args[1])
    return *fix(celly, cellx), *fix(bgy, bgx)
