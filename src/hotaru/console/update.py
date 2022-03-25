from cleo import Command

from .options import options


class UpdateCommand(Command):

    name = 'update'
    description = 'Update footprint and spike'
    help = '''
'''

    options = [
        options['tag'],
    ]

    def handle(self):
        tag = self.option('tag')
        self.call('data')
        self.call('find')
        self.call('reduce')
        self.call('init')
        self.call('temporal')
        self.call('spatial', '-f')
        self.call('clean', '-f')
        self.call('temporal', '-f')
