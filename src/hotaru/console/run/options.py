import click


def radius_options():
    return [
        click.Option(
            ['--radius-type'],
            type=click.Choice(['log', 'linear']),
        ),
        click.Option(
            ['--radius-min'],
            type=float,
        ),
        click.Option(
            ['--radius-max'],
            type=float,
        ),
        click.Option(
            ['--radius-num'],
            type=int,
        ),
    ]


def model_options():
    return tau_options() + reg_options() + opt_options()


def tau_options():
    return [
            click.Option(
            ['--tau-rise'],
            type=float,
        ),
        click.Option(
            ['--tau-fall'],
            type=float,
        ),
        click.Option(
            ['--tau-scale'],
            type=float,
        ),
    ]


def reg_options():
    return [
        click.Option(
            ['--la'],
            type=float,
        ),
        click.Option(
            ['--lu'],
            type=float,
        ),
        click.Option(
            ['--bx'],
            type=float,
        ),
        click.Option(
            ['--bt'],
            type=float,
        ),
    ]


def opt_options():
    return [
        click.Option(
            ['--lr'],
            type=float,
        ),
        click.Option(
            ['--tol'],
            type=float,
        ),
        click.Option(
            ['--epoch'],
            type=int,
        ),
        click.Option(
            ['--steps'],
            type=int,
        ),
        click.Option(
            ['--batch'],
            type=int,
        ),
    ]
