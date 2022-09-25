from .input import DynamicInputLayer
from .input import TemporalBackground
from .loss import LossLayer
from .prox import L2
from .prox import MaxNormNonNegativeL1


class HotaruVariableMixin:
    """Variable"""

    def init_variable(self, nk, nx, nt, tausize):
        nu = nt + tausize - 1

        footprint_prox = MaxNormNonNegativeL1(axis=-1, name="footprint_prox")
        footprint_layer = DynamicInputLayer(nk, nx, name="footprint")
        footprint_layer.set_regularizer(footprint_prox)

        spike_prox = MaxNormNonNegativeL1(axis=-1, name="spike_prox")
        spike_layer = DynamicInputLayer(nk, nu, name="spike")
        spike_layer.set_regularizer(spike_prox)

        localx_prox = L2(name="localx_prox")
        localx_layer = DynamicInputLayer(nk, nx, name="localx")
        localx_layer.set_regularizer(localx_prox)

        localt_prox = L2(name="localt_prox")
        localt_layer = DynamicInputLayer(nk, nt, name="localt")
        localt_layer.set_regularizer(localt_prox)

        spatial_loss_layer = LossLayer(nk, nx, nt)
        temporal_loss_layer = LossLayer(nk, nt, nx)

        self.nm = nx * nt + nx + nt
        self.footprint = footprint_layer
        self.spike = spike_layer
        self.localx = localx_layer
        self.localt = localt_layer
        self.spatial_loss = spatial_loss_layer
        self.temporal_loss = temporal_loss_layer

    def set_penalty(self, footprint, spike, localx, localt, spatial, temporal):
        self.footprint._val.regularizer.set_l(footprint / self.nm)
        self.spike._val.regularizer.set_l(spike / self.nm)
        self.localx._val.regularizer.set_l(localx / self.nm)
        self.localt._val.regularizer.set_l(localt / self.nm)
        self.spatial_loss.set_background_penalty(spatial, temporal)
        self.temporal_loss.set_background_penalty(temporal, spatial)

    def penalty(self, dummy=None):
        return (
            self.footprint.penalty(self.footprint(dummy))
            + self.spike.penalty(self.spike(dummy))
            + self.localx.penalty(self.localx(dummy))
            + self.localt.penalty(self.localt(dummy))
        )
