from functools import wraps

import click


def tau_options(func):
    hz = click.option(
        '--hz',
        type=float,
        default=20.0,
        show_default=True,
    )
    rise = click.option(
        '--tau-rise',
        type=float,
        default=0.08,
        show_default=True,
    )
    fall = click.option(
        '--tau-fall',
        type=float,
        default=0.16,
        show_default=True,
    )
    scale = click.option(
        '--tau-scale',
        type=float,
        default=6.0,
        show_default=True,
    )
    return hz(rise(fall(scale(func))))


def tau_wrap(func):

    @wraps(func)
    def new_func(hz, tau_rise, tau_fall, tau_scale, **args):
        tau = dict(hz=hz, tau1=tau_rise, tau2=tau_fall, tscale=tau_scale)
        return func(tau=tau, **args)

    return new_func
