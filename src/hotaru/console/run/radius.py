from functools import wraps

import numpy as np
import click


def radius_options(func):
    return click.option(
        '--radius-type',
        type=click.Choice(['log', 'linear']),
        default='log',
        show_default=True,
    )(click.option(
        '--radius-min',
        type=float,
        default=2.0,
        show_default=True,
    )(click.option(
        '--radius-max',
        type=float,
        default=20.0,
        show_default=True,
    )(click.option(
        '--radius-num',
        type=int,
        default=13,
        show_default=True,
    )(func))))


def radius_wrap(func):

    @wraps(func)
    def new_func(radius_type, radius_min, radius_max, radius_num, **args):
        if radius_type == 'linear':
            radius = np.linspace(radius_min, radius_max, radius_num)
        elif radius_type == 'log':
            radius_min = np.log10(radius_min)
            radius_max = np.log10(radius_max)
            radius = np.logspace(radius_min, radius_max, radius_num)
        return func(radius=radius, **args)

    return new_func
