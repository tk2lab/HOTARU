from functools import wraps

import click


def reg_options(func):
    la = click.option(
        '--la',
        type=float,
        default=20.0,
        show_default=True,
    )
    lu = click.option(
        '--lu',
        type=float,
        default=30.0,
        show_default=True,
    )
    bx = click.option(
        '--bx',
        type=float,
        default=0.1,
        show_default=True,
    )
    bt = click.option(
        '--bt',
        type=float,
        default=0.1,
        show_default=True,
    )
    return la(lu(bx(bt(func))))


def reg_wrap(func):

    @wraps(func)
    def new_func(la, lu, bx, bt, **args):
        return func(reg=dict(la=la, lu=lu, bx=bx, bt=bt), **args)

    return new_func
