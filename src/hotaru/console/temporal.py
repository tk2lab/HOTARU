from .base import CommandBase
from .model import ModelMixin
from .options import options
from .options import tag_options
from .options import model_options
from .options import optimizer_options

from ..train.spike import SpikeModel
from ..util.csv import load_csv
from ..util.numpy import load_numpy
from ..util.csv import save_csv
from ..util.numpy import save_numpy


class TemporalCommand(CommandBase, ModelMixin):

    name = 'temporal'
    _type = 'spike'
    description = 'Update spike'
    help = '''
'''

    options = CommandBase.base_options() + [
        options['data-tag'],
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
        return f'hotaru/{self._type}/{tag}{curr}_log.pickle'

    def _handle(self, p):
        tag = p['tag']
        stage = p['stage']
        curr = '' if stage < 0 else f'_{stage:03}'
        self.line(f'{tag}, {stage}')

        footprint = load_numpy(f'hotaru/footprint/{tag}{curr}.npy')
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
        save_numpy(f'hotaru/spike/{tag}{curr}.npy', model.spike.val)
        if stage >= 0:
            save_numpy(f'hotaru/spike/{tag}.npy', model.spike.val)

        index = load_pickle(f'hotaru/footprint/{tag}{curr}_log.pickle')['peaks'].query('accetp == "yes"').index
        p.update(dict(index=index, log=log.history))
