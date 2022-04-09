import click

from hotaru.sim.make import make_sim


@click.command()
@click.option('--outdir', type=click.Path(), default='sample', show_default=True)
@click.option('--width', type=int, default=200, show_default=True)
@click.option('--height', type=int, default=200, show_default=True)
@click.option('--frames', type=int, default=1000, show_default=True)
@click.option('--hz', type=float, default=20.0, show_default=True)
@click.option('--num-neurons', type=int, default=1000, show_default=True)
@click.option('--intensity-min', type=float, default=0.5, show_default=True)
@click.option('--intensity-max', type=float, default=2.0, show_default=True)
@click.option('--radius-min', type=float, default=4.0, show_default=True)
@click.option('--radius-max', type=float, default=8.0, show_default=True)
@click.option('--radius-min', type=float, default=4.0, show_default=True)
@click.option('--distance', type=float, default=1.8, show_default=True)
@click.option('--firingrate-min', type=float, default=0.2, show_default=True)
@click.option('--firingrate-max', type=float, default=2.2, show_default=True)
@click.option('--tau_rise', type=float, default=0.08, show_default=True)
@click.option('--tau_fall', type=float, default=0.18, show_default=True)
@click.option('--seed', type=int)
def sample(**args):
    '''Make Sample'''

    make_sim(**args)
