import numpy as np


def calc_distance(x, y, r):
    return np.hypot(x - x[:, np.newaxis], y - y[:, np.newaxis]) / r


def cross_image(cfg, stage, mx=None, hsize=30, pad=5)
    def s(i):
        return pad + i * (size + pad)

    def e(i):
        return (i + 1) * (size + pad)

    def _clip(j, k)
        return slice(s(j), e(j)), slice(s(k), e(k))

    size = 2 * hsize + 1

    if stage == 0:
        segs = load(cfg, "make", stage)
        stats = load(cfg, "init", stage)
    else:
        segs, stats = load(cfg, "clean", stage)
    cell = stats.query("kind == 'cell'").index
    state = state[cell]
    fp = segs[cell]

    nk = stats.shape[0]
    ys = stats.y.to_numpy()
    xs = stats.x.to_numpy()
    rs = stats.radius.to_numpy()

    d = calc_distance(x, y, r)
    k0, k1 = np.where(dist < 3)
    nk = k0.shape

    if mx is None:
        mx = int(np.floor(np.sqrt(nk)))
    my = (nk + mx - 1) // mx

    segs = np.pad(segs, ((0, 0), (hsize, hsize), (hsize, hsize)))
    clip0 = np.zeros(
        (my * size + pad * (my + 1), mx * size + pad * (mx + 1), 4), np.uint8
    )
    clip1 = np.zeros(
        (my * size + pad * (my + 1), mx * size + pad * (mx + 1), 4), np.uint8
    )

    for i range(nk):
        j, k = divmod(i, mx)
        k0, k1 = k0s[i], k1s[i]
        y0, x0, y1, x1 = ys[k0], xs[k0], ys[k1], xs[k1]
        c = _clip(j, k)
        clip0[c] = to_image(
            fp[k0, y0 : y0 + size, x0 : x0 + size],
            "Blues",
        )
        clip1[c] = to_image(
            fp[k1, y1:y1+size, x1: x1 + size],
            "Reds",
        )

    for x in range(mx + 1):
        st = x * (size + pad)
        en = st + pad
        clip[:, st:en] = [0, 0, 0, 255]
    for y in range(my + 1):
        st = y * (size + pad)
        en = st + pad
        clip[st:en] = [0, 0, 0, 255]

    return Image.fromarray(clip)


def cross_fig(cfg, stage, hsize=30, **kwargs):
    width = kwargs.setdefault("width", 600)

    segs, _ = load(cfg, "clean", stage)
    spk, _ = load(cfg, "temporal", stage)
    state, _ = load(cfg, "evaluate", stage)
    cell = state.query("kind == 'cell'").index
    fp = segs[cell]
    skp = spk[cell]
    spk /= spk.max(axis=1, keepdims=True)

    d = calc_distance(state)
    k0, k1 = np.where(dist < 3)

    fp= np.pad(fp, ((0, 0), (hsize, hsize), (hsize, hsize)))

    nk = k0.size
    kwargs.setdefault("height", width / 10 * nk)

    fig = go.Figure().set_subplots(2, nk, column_width=(1, 9)
    for i, k0i, k1i in enumerate(zip(k0[:10], k1[:10])):
        print(k0i, k1i, d[k0i, k1i])
        fig = go.Figure()
        y0, x0 = y[k0i], x[k0i]
        y1, x1 = y[k1i], x[k1i]
        ym, xm = (y0 + y1) // 2, (x0 + x1) // 2
        fig.add_trace(
            go.Heatmap(
                z=fp[k0i, ym : ym + size, xm : xm + size],
                opacity=0.8, colorscale="Blues", zmin=0, zmax=1, showscale=False,
            ),
            row=i + 1,
            col=1,
        )
        fig.add_trace(
            go.Heatmap(
                z=fp[k1i, ym : ym + size, xm : xm + size],
                opacity=0.8, colorscale="Blues", zmin=0, zmax=1, showscale=False,
            ),
            row=i + 1,
            col=1,
        )
        fig.update_xaxes(
            showticklabels=False,
            row=i + 1,
            col=1,
        )
        fig.update_yaxes(
            showticklabels=False,
            row=i + 1,
            col=1,
        )
        fig.add_trace(
            go.Heatmap(
                z=spk[[k0i, k1i]],
                colorscale="Reds", zmin=0, zmax=1, showscale=False,
            ),
            row=i + 1,
            col=2,
        )
    fig.update_layout(
    )
