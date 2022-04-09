import click

from hotaru.sim.mpeg import make_mpeg


@click.command()
@click.option('--stage', '-s', type=int)
@click.option('--has-truth', is_flag=True)
@click.pass_obj
def mpeg(obj, stage, has_truth):
    '''Make Mp4'''

    if stage is None:
        stage = 'curr'
    else:
        stage = f'{stage:03}'
    make_mpeg(obj.data_tag, obj.tag, stage, has_truth)
