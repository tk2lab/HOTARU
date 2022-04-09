import random
import os

import tensorflow as tf

import numpy as np
import scipy.stats as st

import pandas as pd
import tifffile

from hotaru.train.dynamics import SpikeToCalcium


def make_sim(
    outdir='sample',
    frames=1000, height=200, width=200, hz=20.0, num_neurons=1000,
    intensity_min=0.5, intensity_max=2.0, radius_min=4.0, radius_max=8.0,
    firingrate_min=0.2, firingrate_max=2.2, distance=1.8,
    tau_rise=0.08, tau_fall=0.16, seed=None,
):
    os.makedirs(f'{outdir}/truth', exist_ok=True)
    w, h, nv = width, height, frames
    nk = num_neurons
    sig_min, sig_max = intensity_min, intensity_max
    rad_min, rad_max = radius_min, radius_max
    fr_min, fr_max = firingrate_min, firingrate_max
    min_dist = distance
    np.random.seed(seed)

    gamma = SpikeToCalcium()
    gamma.set_double_exp(hz, tau_rise, tau_fall, 6.0)
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
            radius = st.uniform(rad_min, rad_max - rad_min).rvs()
            r = np.random.randint(ylist.size)
            y0, x0 = ylist[r], xlist[r]
            if (len(xs) == 0) or np.all((np.array(xs) - x0)**2 + (np.array(ys) - y0)**2 > (min_dist*radius)**2):
                break
        if i == 9:
            break
        rate = st.uniform(fr_min / hz, (fr_max - fr_min) / hz).rvs()
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

    np.save(f'{outdir}/truth/a0.npy', a_t.numpy())
    np.save(f'{outdir}/truth/u0.npy', u_t.numpy())
    np.save(f'{outdir}/truth/v0.npy', v_t.numpy())
    tifffile.imwrite(f'{outdir}/imgs.tif', imgs)
