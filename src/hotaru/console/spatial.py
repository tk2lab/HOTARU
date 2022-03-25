from .base import CommandBase
from .model import ModelMixin
from .options import tag_options
from .options import options
from .options import model_options
from .options import optimizer_options

from ..train.footprint import FootprintModel
from ..util.numpy import load_numpy
from ..util.numpy import save_numpy
from ..util.pickle import save_pickle


class SpatialCommand(CommandBase, ModelMixin):

    name = 'spatial'
    _suff = '_orig'
    _type = 'footprint'
    description = 'Update footprint'
    help = '''
'''

    options = CommandBase.options + [
        options['data_tag'],
        tag_options['spike_tag'],
    ] + model_options + optimizer_options + [
        options['batch'],
    ]

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
        footprint = model.get_footprints()
        save_numpy(f'{base}.npy', footprint)

        save_pickle(f'{base}_log.pickle', dict(
            data=self.option('data-tag'),
            footprint=self.option('spike-tag'),
            mask=self.mask(), **tau, **regularization,
        ))
