import tensorflow.keras.backend as K
import numpy as np

from .model import BaseModel
from .callback import SpikeCallback


class SpikeModel(BaseModel):

    def __init__(self, footprint, spike, variance, **kwargs):
        super().__init__(footprint, spike, variance, **kwargs)
        self.spike = spike
        self.footprint_get = footprint.get_val
        self.footprint_set = footprint.set_val

    def call(self, inputs):
        spike = self.spike(inputs)
        calcium = self.variance.spike_to_calcium(spike)
        loss, footprint_penalty, spike_penalty = self.call_common(calcium)
        self.add_metric(footprint_penalty, 'mean', 'penalty')
        return loss

    def fit(self, lr, batch, **kwargs):
        footprint = self.footprint_get()
        self.variance.start_spike_mode(footprint, batch)
        nk = footprint.shape[0]
        nu = self.variance.nu
        self.spike.val = np.zeros((nk, nu))
        self.optimizer.learning_rate = lr * 2.0 / self.variance.lipschitz_u
        self.fit_common(SpikeCallback, **kwargs)
