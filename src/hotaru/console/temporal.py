from .base import CommandBase
from .model import ModelMixin
from .options import tag_options
from .options import options
from .options import model_options
from .options import optimizer_options

from ..train.spike import SpikeModel
from ..util.numpy import load_numpy
from ..util.numpy import save_numpy
from ..util.pickle import save_pickle



class TemporalCommand(CommandBase, ModelMixin):

    name = 'temporal'
    _type = 'spike'
    description = 'Update spike'
    help = '''
'''

    options = CommandBase.options + [
        options['data_tag'],
        tag_options['footprint_tag'],
    ] + model_options + optimizer_options + [
        options['batch'],
    ]

    def _handle(self, base):
        footprint_tag = self.option('footprint-tag')
        footprint_base = f'hotaru/footprint/{footprint_tag}'
        footprint = load_numpy(f'{footprint_base}.npy')
        nk = footprint.shape[0]
        tau, regularization, models = self.models(nk)

        model = SpikeModel(**models)
        model.set_penalty(**regularization)
        model.compile()
        model.fit(
            footprint,
            lr=self.option('lr'),
            min_delta=self.option('tol'),
            epochs=self.option('epoch'),
            steps_per_epoch=self.option('step'),
            batch=self.option('batch'),
            verbose=self.verbose(),
            #log_dir=logs, stage=base,
        )
        save_numpy(f'{base}.npy', model.spike.val)

        save_pickle(f'{base}_log.pickle', dict(
            data=self.option('data-tag'),
            footprint=self.option('footprint-tag'),
            **tau, **regularization,
        ))
