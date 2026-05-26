from logging import getLogger

import matplotlib.pyplot as plt

from ..saving import Data
from ..typing import Array
from ..typing import Shape
from .dynamics import Kernel

logger = getLogger(__name__)


class Traces(Data):
    obs: Array

    @property
    def shape(self) -> Shape:
        return self.obs.shape

    def plot_obs(self, ax=None, **kwargs):
        if ax is None:
            _fig, ax = plt.subplots(**kwargs)
        num, _nt = self.obs.shape
        vmax = self.obs.max()
        for i, xi in enumerate(self.obs):
            ax.plot(i - xi, c='blue', lw=0.3)
        ax.set_ylim(num + 0.1, -vmax)
        ax.set_yticks([])
        return ax


    def plot_core(self, kernel: Kernel):
        pass
