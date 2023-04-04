import numpy as np
import pandas as pd


def reduce_peak(vs, ts, rs, rmin, rmax, thr_distance):
    h, w = vs.shape
    idx = np.argsort(vs.ravel())[::-1]
    ys, xs = np.divmod(idx, w)
    ts = ts[ys, xs]
    rs = rs[ys, xs]
    flg = np.arange(h * w)
    out = []
    while flg.size > 0:
        i, j = flg[0], flg[1:]
        y0, x0 = ys[i], xs[i]
        y1, x1 = ys[j], xs[j]
        yo, xo = ys[out], xs[out]
        thr = thr_distance * rs[i]
        if (rmin < rs[i] < rmax) and (
            not out or np.all(np.hypot(xo - x0, yo - y0) >= thr)
        ):
            cond = np.hypot(x1 - x0, y1 - y0) >= thr
            flg = j[cond]
            out.append(i)
        else:
            flg = j
    return ys[out], xs[out]


def reduce_peak_block(peakval, rmin, rmax, thr_distance, block_size):
    vs, ts, rs = (np.array(o) for o in peakval)
    h, w = vs.shape
    margin = int(np.ceil(thr_distance * rs.max()))
    yout, xout = [], []
    for xs in range(0, w - margin, block_size):
        x0 = max(xs - margin, 0)
        xe = xs + block_size
        x1 = min(xe + margin, w)
        for ys in range(0, h - margin, block_size):
            y0 = max(ys - margin, 0)
            ye = ys + block_size
            y1 = min(ye + margin, h)
            v = vs[y0:y1, x0:x1]
            t = ts[y0:y1, x0:x1]
            r = rs[y0:y1, x0:x1]
            y, x = reduce_peak(v, t, r, rmin, rmax, thr_distance)
            y += y0
            x += x0
            cond = (ys <= y) & (y < ye) & (xs <= x) & (x < xe)
            yout.append(y[cond])
            xout.append(x[cond])
    yout = np.concatenate(yout, axis=0)
    xout = np.concatenate(xout, axis=0)
    tout = ts[yout, xout]
    rout = rs[yout, xout]
    vout = vs[yout, xout]
    return pd.DataFrame(dict(t=tout, r=rout, y=yout, x=xout, v=vout))
