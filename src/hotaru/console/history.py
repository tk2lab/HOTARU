import pickle

from .base import Command, _option


class HistoryCommand(Command):

    name = 'history'
    description = 'Show History'
    help = '''
'''

    options = [
        _option('job-dir'),
    ]

    def handle(self):
        self.set_job_dir()
        history = self.status.history
        for k, hs in history.items():
            self.line(f'name: {k}')
            for s, h in enumerate(hs):
                if s == 0:
                    self.line(f'data ({s}): {h[0]}, {h[1]}')
                elif s == 1:
                    self.line(f'peak ({s}): gamma={h[0]}, thr-intensity{h[2]}, shard={h[3]}')
                    self.line('    radius=(' + ', '.join(f'{r:.2f}' for r in h[1]) + ')')
                elif s == 2:
                    self.line(f'segment ({s}): thr-distance{h[0]}')
                elif s % 3 == 0:
                    self.line(f'spike ({s}): tau={h[0]}, lu={h[1]}, bx={h[2]}, bt={h[3]}')
                elif s % 3 == 1:
                    self.line(f'footprint ({s}): la={h[0]}, bx={h[1]}, bt={h[2]}')
                elif s % 3 == 2:
                    self.line(f'{h}')
                    self.line(f'clean ({s}): gamma={h[0]}, thr-firmness={h[2]}, thr-sim-area={h[3]}, thr-similarity={h[4]}')
                    self.line('    radius=(' + ', '.join(f'{r:.2f}' for r in h[1]) + ')')
