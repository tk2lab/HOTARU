# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division

import tensorflow.compat.v2 as tf
import numpy as np

from bokeh.models import ColumnDataSource, HoverTool
from bokeh.plotting import figure

from skimage.morphology import label, convex_hull_image
from skimage.measure import find_contours


def fig_peaks(self):
    mask = self.mask.numpy()
    h, w = tf.shape(mask).numpy()
    t, y, x, r, g, v, m = [tmp.numpy() for tmp in self._peak]
    nk = g.size
    plot = figure(
        title='peaks',
        plot_width=w, plot_height=h,
        x_range=(0,w), y_range=(h,0),
    )
    source = ColumnDataSource(dict(x=x, y=y, r=r, idx=np.arange(nk), g=g))
    plot.circle(x='x', y='y', radius='r', fill_alpha='g', fill_color='red', source=source)
    plot.add_tools(HoverTool(tooltips=[('ID', '@idx'), ('gl', '@g')]))
    return plot


def fig_cells(self, ids=None, thr=0.0):
    box = np.ones((3,3), bool)

    h, w = tf.shape(self.imgs.mask).numpy()

    sc = self.footprint.gs.numpy()
    size, pos, val = self.footprint.size, self.footprint.pos, self.footprint.val

    nk = tf.size(size).numpy()
    pos = tf.RaggedTensor.from_row_lengths(pos, size)
    val = tf.RaggedTensor.from_row_lengths(val, size)

    if ids is None:
        ids = self.footprint.ids.numpy()
    rmap = {j: i for i, j in enumerate(self.footprint.ids.numpy())}

    xs, ys = [], []
    for k in ids:
        i = rmap[k]
        pk, vk = pos[i], val[i]
        img = tf.scatter_nd(pk, vk, (h,w)).numpy()
        peak = np.where(img == img.max())
        y, x = peak[0][0], peak[1][0]
        img = img > thr
        lbl = label(img, connectivity=1)
        img = lbl == lbl[y,x]
        img = convex_hull_image(img)
        cs = find_contours(img, 0.5)
        if len(cs) > 1:
            print(y,x)
            plt.imshow(img[y-50:y+50,x-50:x+50])
            plt.show()
        for c in cs:
            xs.append(c[:,1])
            ys.append(c[:,0])

    plot = figure(
        title='cells',
        plot_width=w, plot_height=h,
        x_range=(0,w), y_range=(h,0),
    )
    source = ColumnDataSource(dict(xs=xs, ys=ys, ids=ids, sc=[sc[rmap[i]] for i in ids]))
    plot.patches(xs='xs', ys='ys', source=source,
                 fill_color='green', fill_alpha='sc', line_color='black',
                 hover_fill_color='green', hover_fill_alpha='sc', hover_line_color='red')
    plot.add_tools(HoverTool(tooltips=[('ID', '@ids'), ('score', '@sc')]))
    return plot


def fig_spikes(self, ids=None):
    gamma = self.gamma

    pad = gamma.pad.numpy()
    hz = self.gamma.hz.numpy()

    u = self.temporal.uval.numpy()
    v = gamma.u_to_v(self.temporal.uval).numpy()
    e = self.temporal.vobs.numpy()

    u = (u / tf.reduce_max(u, axis=1, keepdims=True)).numpy()
    e -= np.median(e, axis=1, keepdims=True)
    e /= e.max()
    v /= v.max()

    if ids is None:
        ids = self.temporal.ids.numpy()
    rmap = {j: i for i, j in enumerate(self.temporal.ids.numpy())}
    nt = u.shape[1]
    t = np.arange(u.shape[1]) / hz

    plot = figure(
        title='spikes', y_range=[str(i) for i in ids[::-1]],
        plot_width=600, plot_height=600,
    )
    for n, k in enumerate(ids[::-1]):
        i = rmap[k]
        plot.vbar(x=t, top=n+1, bottom=n, width=u[i]/hz, line_color=None, fill_color='#ff0000')
        plot.line(x=t[pad:], y=e[i]+n, line_color='#333333', line_alpha=0.5, line_width=2.0)
        plot.line(x=t[pad:], y=v[i]+n, line_color='#0000ff', line_alpha=0.5, line_width=2.0)
    return plot
