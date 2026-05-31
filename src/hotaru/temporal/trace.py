from logging import getLogger

import matplotlib.pyplot as plt

from ..models import Layer
from ..saving import Config
from ..typing import Shape
from .dynamics import Kernel

logger = getLogger(__name__)


class Traces(Layer):
    def __init__(self, obs_or_shape, **kwargs):
        super().__init__(**kwargs)
        match obs_or_shape:
            case (int(), int()) as shape:
                self.obs = self.add_weight(shape, name='segs')
            case obs:
                self.obs = self.add_weight(obs.shape, initializer=obs, name='obs')
        self._build_at_init()

    def get_config(self) -> Config:
        return {'obs_or_shape': self.obs.shape, **super().get_config()}

    @property
    def shape(self) -> Shape:
        return self.obs.shape

    def plot_obs(self, ax=None, **kwargs):
        obs = self.obs.numpy()
        if ax is None:
            _fig, ax = plt.subplots(**kwargs)
        num, _nt = obs.shape
        vmax = self.obs.max()
        for i, xi in enumerate(obs):
            ax.plot(i - xi, c='blue', lw=0.3)
        ax.set_ylim(num + 0.1, -vmax)
        ax.set_yticks([])
        return ax

    def plot_core(self, kernel: Kernel):
        pass
