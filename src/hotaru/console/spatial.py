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
    _type = 'footprint'
    _suff = '_orig'
    description = 'Update footprint'
    help = '''
'''

    options = CommandBase.base_options() + [
        options['data-tag'],
        tag_options['spike-tag'],
        options['stage'],
    ] + model_options + optimizer_options + [
        options['batch'],
    ]

    def log_path(self):
        tag = self.p('tag')
        stage = self.p('stage')
        if stage < 0:
            curr = ''
        else:
            curr = f'_{stage:03}'
        return f'hotaru/{self._type}/{tag}{curr}_orig_log.pickle'

    def _handle(self, p):
        mask, nt = self.data_prop()

        stage = p['stage']
        if stage < 0:
            prev = ''
        elif stage == 0:
            prev = '_init'
        else:
            prev = f'_{stage-1:03}'
        curr = '' if stage < 0 else f'_{stage:03}'

        spike_tag = self.option('spike-tag')
        spike = load_numpy(f'hotaru/spike/{spike_tag}{prev}.npy')
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

        tag = p['tag']
        save_numpy(f'hotaru/footprint/{tag}{curr}_orig.npy', footprint)
        if stage >= 0:
            save_numpy(f'hotaru/footprint/{tag}_orig.npy', footprint)

        p.update(dict(mask=mask, nt=nt, nk=nk, log=log.history))
