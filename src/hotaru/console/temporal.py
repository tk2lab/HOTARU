from .base import CommandBase
from .model import ModelMixin
from .options import options
from .options import tag_options
from .options import model_options
from .options import optimizer_options

from ..train.spike import SpikeModel
from ..util.numpy import load_numpy
from ..util.numpy import save_numpy


class TemporalCommand(CommandBase, ModelMixin):

    name = 'temporal'
    _type = 'spike'
    description = 'Update spike'
    help = '''
'''

    options = CommandBase.base_options('work') + [
        options['data-tag'],
        tag_options['footprint-tag'],
    ] + model_options + optimizer_options + [
        options['batch'],
    ]

    def _handle(self, base, p):
        footprint_tag = p['footprint-tag']
        footprint = load_numpy(f'hotaru/footprint/{footprint_tag}.npy')
        nk = footprint.shape[0]

        tau, regularization, models = self.models(p, nk)
        model = SpikeModel(**models)
        model.set_penalty(**regularization)
        model.compile()
        log = model.fit(
            footprint, lr=p['lr'], min_delta=p['tol'],
            epochs=p['epoch'], steps_per_epoch=p['step'],
            batch=p['batch'], verbose=p['verbose'],
            #log_dir=logs, stage=base,
        )
        save_numpy(f'{base}.npy', model.spike.val)

        mask, nt = self.data_prop()
        p.update(dict(mask=mask, nk=nk, nt=nt, log=log.history))
