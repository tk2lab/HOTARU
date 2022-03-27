import numpy as np

from ..optimizer.input import MaxNormNonNegativeL1InputLayer as Input
from ..train.variance import Variance


class ModelMixin:

    def models(self, p, nk):
        data = self.data()
        mask, nt = self.data_prop()
        nx = np.count_nonzero(mask)

        tau = self.tau(p)
        regularization = self.regularization(p)

        variance = Variance(data, nk, nx, nt)
        variance.set_double_exp(**tau)

        footprint = Input(nk, nx, name='footprint')
        footprint.mask = mask

        spike = Input(nk, variance.nu, name='spike')

        model = dict(footprint=footprint, spike=spike, variance=variance)
        return tau, regularization, model

    def tau(self, p):
        return dict(
            hz=p['hz'],
            tau1=p['tau-rise'],
            tau2=p['tau-fall'],
            tscale=p['tau-scale'],
        )

    def regularization(self, p):
        return dict(
            lu=p['lu'],
            la=p['la'],
            bt=p['bt'],
            bx=p['bx'],
        )
