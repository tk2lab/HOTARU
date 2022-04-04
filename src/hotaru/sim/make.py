import random

import tensorflow as tf

import numpy as np
import scipy.stats as st

import pandas as pd
import tifffile

from hotaru.train.dynamics import SpikeToCalcium


def make_sim(nv=1000, h=200, w=200, hz=20.0, nk=1000, sig_min=0.5, sig_max=2.0, min_dist=1.8, seed=None):
    np.random.seed(seed)

    gamma = SpikeToCalcium()
    gamma.set_double_exp(hz, 0.08, 0.16, 6.0)
    nu = nv + gamma.pad

    a_true = []
    u_true = []

    y, x = np.mgrid[:h, :w]
    ylist, xlist = y.flatten(), x.flatten()
    cond = (xlist > 10) & (xlist < w-10) & (ylist > 10) & (ylist < h-10)
    ylist = ylist[cond]
    xlist = xlist[cond]
    xs = []
    ys = []
    rs = []
    ts = []
    ss = []
    for k in range(nk):
        if ylist.size  == 0:
            break
        for i in range(10):
            radius = st.uniform(4.0, 4.0).rvs()
            r = np.random.randint(ylist.size)
            y0, x0 = ylist[r], xlist[r]
            if (len(xs) == 0) or np.all((np.array(xs) - x0)**2 + (np.array(ys) - y0)**2 > (min_dist*radius)**2):
                break
        if i == 9:
            break
        rate = st.uniform(0.2 / hz, 2.0 / hz).rvs()
        ui = st.bernoulli(rate).rvs(nu)
        signal = st.uniform(sig_min, sig_max).rvs()
        cond = (xlist - x0)**2 + (ylist - y0)**2 > (min_dist*radius)**2
        ylist = ylist[cond]
        xlist = xlist[cond]
        xs.append(x0)
        ys.append(y0)
        rs.append(radius)
        ts.append(ui.sum())
        ss.append(signal)
        a_true.append(np.exp(-0.5 * ((x - x0)**2 + (y - y0)**2) / radius**2))
        u_true.append(signal * ui)
        print(k, y0, x0, radius, ui.sum(), signal)

    pd.DataFrame(dict(x=xs, y=ys, radius=rs, nspike=ts, signal=ss)).to_csv('cells.csv')
    nk = len(a_true)

    a_t = tf.reshape(tf.convert_to_tensor(a_true, tf.float32), (nk, -1))
    u_t = tf.convert_to_tensor(u_true, tf.float32)
    v_t = gamma(u_t)
    f_t = tf.matmul(v_t, a_t, True, False)
    a_t = tf.reshape(a_t, (nk, h, w))
    f_t = tf.reshape(f_t, (nv, h, w))
    n_t = tf.random.normal((nv, h, w))

    wh = w / 2
    hh = h / 2

    base_t = 0.5 * np.cos(np.linspace(0,4*np.pi,nv))[:,np.newaxis,np.newaxis]
    base_x = -5.0 * (((x - hh) / hh)**2 + ((y - wh) / wh)**2)
    imgs = (f_t + n_t + base_t + base_x).numpy()

    np.save('./a0.npy', a_t.numpy())
    np.save('./u0.npy', u_t.numpy())
    np.save('./v0.npy', v_t.numpy())
    np.save('./f0.npy', imgs)
    tifffile.imwrite('./imgs.tif', imgs)

    imgs -= imgs.mean(axis=0, keepdims=True)
    imgs -= imgs.mean(axis=(1, 2), keepdims=True)
    imgs /= imgs.std()
    tifffile.imwrite('./norm.tif', imgs)


if __name__ == '__main__':
    make_sim(nk=100, sig_min=1.0)
