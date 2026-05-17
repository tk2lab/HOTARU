from logging import getLogger

import matplotlib.pyplot as plt

from ..saving import Data
from ..typing import Array

logger = getLogger(__name__)


class TemporalComponents(Data):
    core: Array
    obs: Array

    def plot_obs(self):
        pass

    def plot_core(self):
        pass
