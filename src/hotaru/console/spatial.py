from .base import CommandBase
from .model import ModelMixin
from .options import options
from .options import tag_options
from .options import model_options
from .options import optimizer_options

from ..train.footprint import FootprintModel
from ..util.numpy import load_numpy
from ..util.numpy import save_numpy


class SpatialCommand(CommandBase, ModelMixin):

    name = 'spatial'
    _suff = '_orig'
    _type = 'footprint'
    description = 'Update footprint'
    help = '''
'''

    options = CommandBase.base_options('work') + [
        options['data-tag'],
        tag_options['spike-tag'],
    ] + model_options + optimizer_options + [
        options['batch'],
    ]

    def _handle(self, base, p):
        mask, nt = self.data_prop()

        spike_tag = self.option('spike-tag')
        spike = load_numpy(f'hotaru/spike/{spike_tag}.npy')
        nk = spike.shape[0]

        tau, regularization, models = self.models(p, nk)
        model = FootprintModel(**models)
        model.set_penalty(**regularization)
        model.compile()
        log = model.fit(
            spike, lr=p['lr'], min_delta=p['tol'],
            epochs=p['epoch'], steps_per_epoch=p['step'],
            batch=p['batch'], verbose=p['verbose'],
            #log_dir=logs, stage=base,
        )
        footprint = model.get_footprints()
        save_numpy(f'{base}.npy', footprint)

        p.update(dict(mask=mask, nt=nt, nk=nk, log=log))
