import click


def radius_options(func):
    rtype = click.option(
        '--radius-type',
        type=click.Choice(['log', 'linear']),
        default='log',
        show_default=True,
    )
    rmin = click.option(
        '--radius-min',
        type=float,
        default=2.0,
        show_default=True,
    )
    rmax = click.option(
        '--radius-max',
        type=float,
        default=20.0,
        show_default=True,
    )
    rnum = click.option(
        '--radius-num',
        type=int,
        default=13,
        show_default=True,
    )
    return rtype(rmin(rmax(rnum(func))))


def model_options(func):
    return tau_options(reg_options(opt_options(func)))


def tau_options(func):
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
    return rise(fall(scale(func)))


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
    batch = click.option(
        '--batch',
        type=int,
        default=100,
        show_default=True,
    )
    return lr(tol(epoch(steps(batch(func)))))
