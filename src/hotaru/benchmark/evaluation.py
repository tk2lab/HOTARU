from math import nan

import numpy as np
import polars as pl
from scipy.stats import zscore


def evaluate(fps, trs, fps_true, trs_true, stats_true=None):
    nk, h, w = fps.shape
    ng, nt = trs_true.shape
    nx = h * w

    fpt_z = zscore(fps_true.reshape(ng, nx), axis=1)
    trt_z = zscore(trs_true, axis=1)
    fps_z = zscore(fps.reshape(nk, nx), axis=1)
    trs_z = zscore(trs, axis=1)

    fps_cor = fps_z @ fpt_z.T / nx
    trs_cor = trs_z @ trt_z.T / nt
    c = (fps_cor < 0) & (trs_cor < 0)
    fps_cor[c] *= -1
    trs_cor[c] *= -1
    cor = fps_cor * trs_cor
    i, j = greedy_matching(cor)

    match_gt = np.full(nk, -1, dtype='int32')
    match_gt[i] = j

    fp_score = np.full(nk, nan, dtype='float32')
    fp_score[i] = fps_cor[i, j]

    tr_score = np.full(nk, nan, dtype='float32')
    tr_score[i] = trs_cor[i, j]

    df = pl.DataFrame({'gt': match_gt, 'fp_score': fp_score, 'tr_score': tr_score})
    if stats_true is not None:
        df = pl.concat((df, stats_true[j]), how='horizontal')
    return df


def greedy_matching(mat):
    num_results, num_gts = mat.shape
    max_possible_pairs = min(num_results, num_gts)
    flat_indices = np.flip(np.argsort(mat.ravel()))

    matched_pairs = []
    used_results = set()
    used_gts = set()
    for flat_idx in flat_indices:
        i, j = flat_idx // num_gts, flat_idx % num_gts
        if (i in used_results) or (j in used_gts):
            continue

        matched_pairs.append((i, j))
        used_results.add(i)
        used_gts.add(j)
        if len(matched_pairs) == max_possible_pairs:
            break
    return np.array(matched_pairs, 'int32').T
