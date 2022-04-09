import pkgutil

import click


@click.command()
def config():
    '''Make hotaru.ini'''

    data = pkgutil.get_data('hotaru.console', 'hotaru.ini').decode('utf-8')
    with open('hotaru.ini', 'w') as f:
        f.write(data)
