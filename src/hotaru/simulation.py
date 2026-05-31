import numpy as np


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
