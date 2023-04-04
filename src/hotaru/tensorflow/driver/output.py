import matplotlib.pyplot as plt
import numpy as np

from ..evaluate.footprint import (
    plot_contour,
    plot_maximum,
)
from ..evaluate.make_mpeg import make_mpeg
from ..evaluate.peaks import (
    plot_circle,
    plot_radius,
)


class HotaruOutputMixin:
    """"""

    def summary(self):
        for kind in ["cell", "local"]:
            print(self.info[self.info.kind == kind])

    def plot_peak_props(self, *args, **kwargs):
        plot_radius(self, *args, **kwargs)
        plt.show()

    def plot_circle(self, *args, **kwargs):
        plot_circle(self, *args, **kwargs)
        plt.show()

    def plot_contour(self, *args, **kwargs):
        mask = self.mask
        val = self.footprint.val
        h, w = mask.shape
        nk = val.shape[0]
        a = np.zeros((nk, h, w))
        a[:, mask] = val
        plot_contour(a, *args, **kwargs)
        val = self.localx.val
        h, w = mask.shape
        nk = val.shape[0]
        a = np.zeros((nk, h, w))
        a[:, mask] = val
        plot_contour(a, *args, **kwargs)
        plt.show()

    def plot_maximum(self, *args, **kwargs):
        plot_maximum(self.footprint.val, self.localx.val, self.mask, *args, **kwargs)
        plt.show()

    def make_mpeg(self, *args, **kwargs):
        make_mpeg(self, *args, **kwargs)
        plt.show()
