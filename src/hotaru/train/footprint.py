import tensorflow.keras.backend as K
import numpy as np

from .model import BaseModel
from .callback import FootprintCallback


class FootprintModel(BaseModel):

    def __init__(self, footprint, spike, variance, **kwargs):
        super().__init__(footprint, spike, variance, **kwargs)
        self.footprint = footprint
        self.spike_get = spike.get_val
        self.spike_set = spike.set_val

    def call(self, inputs):
        footprint = self.footprint(inputs)
        loss, footprint_penalty, spike_penalty = self.call_common(footprint)
        self.add_metric(footprint_penalty, 'mean', 'penalty')
        return loss

    def fit(self, lr, batch, **kwargs):
        spike = self.spike_get()
        self.variance.start_footprint_mode(spike, batch)
        nk = spike.shape[0]
        nx = self.variance.nx
        self.footprint.val = np.zeros((nk, nx))
        self.optimizer.learning_rate = lr * 2.0 / self.variance.lipschitz_a
        self.fit_common(FootprintCallback, **kwargs)
