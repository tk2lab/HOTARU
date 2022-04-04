import numpy as np


def calc_sim(x0, x1):
    x0n = x0 / np.sqrt((x0 ** 2).sum(axis=1, keepdims=True))
    x1n = x1 / np.sqrt((x1 ** 2).sum(axis=1, keepdims=True))
    return np.dot(x0n, x1n.T)


def eval_sim(a0, v0, a1, v1):
    n, h, w = a0.shape
    m = a1.shape[0]

    a0 = a0.reshape(n, h * w)
    a1 = a1.reshape(m, h * w)
    sim = calc_sim(a0, a1)

    c0 = 0
    ci = 0
    pair = []
    for i in range(n):
        j = np.argmax(sim[i])
        k = np.argmax(sim[:, j])
        if i == k:
            pair.append((i, j))
            c0 += 1
        else:
            #print('x', i, j, sim[i, j])
            #print('x', k, j, sim[k, j])
            ci += 1
            '''
            if sim[i, j] < sim[k, j]:
                pair.append((k, j))
            else:
                pair.append((i, j))
            '''

    cj = 0
    for j in range(m):
        i = np.argmax(sim[:, j])
        k = np.argmax(sim[i])
        if j != k:
            #print('y', i, j, sim[i, j])
            #print('y', i, k, sim[i, k])
            cj += 1

    sim = calc_sim(v0, v1)
    vsim = np.array([sim[i,j] for i, j in pair])
    return c0, ci, cj, vsim
