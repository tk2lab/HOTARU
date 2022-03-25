import numpy as np

from ..optimizer.input import MaxNormNonNegativeL1InputLayer as Input
from ..train.variance import Variance


class ModelMixin:

    def models(self, nk):
        data, mask, nt = self.data()
        nx = np.count_nonzero(mask)

        tau = self.tau()
        regularization = self.regularization()

        variance = Variance(data, nk, nx, nt)
        variance.set_double_exp(**tau)

        footprint = Input(nk, nx, name='footprint')
        footprint.mask = mask

        spike = Input(nk, variance.nu, name='spike')

        model = dict(footprint=footprint, spike=spike, variance=variance)
        return tau, regularization, model

    def tau(self):
        return dict(
            hz=self.option('hz'),
            tau1=self.option('tau-rise'),
            tau2=self.option('tau-fall'),
            tscale=self.option('tau-scale'),
        )

    def regularization(self):
        return dict(
            lu=self.option('lu'),
            la=self.option('la'),
            bt=self.option('bt'),
            bx=self.option('bx'),
        )
