from .base import CommandBase
from .model import ModelMixin
from .base import option

from ..train.footprint import FootprintModel
from ..util.numpy import load_numpy
from ..util.numpy import save_numpy
from ..util.pickle import save_pickle



class SpatialCommand(CommandBase, ModelMixin):

    name = 'spatial'
    _type = 'footprint'
    description = 'Update footprint'
    help = '''
'''

    options = CommandBase.options + [
        option('data-tag', 'd', '', False, False, False, 'default'),
        option('spike-tag', 'p', '', False, False, False, 'default'),
    ] + ModelMixin._options + CommandBase.optimizer_options

    def _handle(self, base):
        spike_tag = self.option('spike-tag')
        spike_base = f'hotaru/spike/{spike_tag}'
        spike = load_numpy(f'{spike_base}.npy')
        nk = spike.shape[0]
        tau, regularization, models = self.models(nk)

        model = FootprintModel(**models)
        model.set_penalty(**regularization)
        model.compile()
        model.fit(
            spike,
            lr=self.option('lr'),
            min_delta=self.option('tol'),
            epochs=self.option('epoch'),
            steps_per_epoch=self.option('step'),
            batch=self.option('batch'),
            verbose=self.verbose(),
            #log_dir=logs, stage=base,
        )

        footprint = model.footprint.val
        i = np.arange(footprint.shape[0])
        j = np.argpartition(-footprint, 1)
        footprint[i, j[:, 0]] = footprint[i, j[:, 1]]
        cond = footprint.max(axis=1) > 0.0
        footprint = footprint[cond]
        save_numpy(f'{base}.npy', footprint)

        save_pickle(f'{base}_log.pickle', dict(
            data=self.option('data-tag'),
            footprint=self.option('spike-tag'),
            **tau, **regularization,
        ))
