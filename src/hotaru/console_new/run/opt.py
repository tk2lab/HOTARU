from functools import wraps

import click


def opt_options(func):
    lr = click.option(
        '--lr',
        type=float,
        default=0.01,
        show_default=True,
    )
    tol = click.option(
        '--tol',
        type=float,
        default=0.001,
        show_default=True,
    )
    epoch = click.option(
        '--epoch',
        type=int,
        default=100,
        show_default=True,
    )
    steps = click.option(
        '--steps',
        type=int,
        default=100,
        show_default=True,
    )
    return lr(tol(epoch(steps(func))))


def opt_wrap(func):

    @wraps(func)
    def new_func(lr, tol, epoch, steps, **args):
        opt = dict(lr=lr, min_delta=tol, epochs=epoch, steps_per_epoch=steps)
        return func(opt=opt, **args)

    return new_func
